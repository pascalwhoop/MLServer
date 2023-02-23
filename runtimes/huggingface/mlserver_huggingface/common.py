import os
import json
from typing import Optional, Dict
from distutils.util import strtobool

import numpy as np
from pydantic import BaseSettings
from mlserver.errors import MLServerError

from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer
import transformers
from mlserver.logging import logger
from mlserver.logging import logger
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from accelerate import Accelerator
from transformers import AutoConfig

accelerator = Accelerator()

from optimum.pipelines import SUPPORTED_TASKS as SUPPORTED_OPTIMUM_TASKS


HUGGINGFACE_TASK_TAG = "task"

ENV_PREFIX_HUGGINGFACE_SETTINGS = "MLSERVER_MODEL_HUGGINGFACE_"
HUGGINGFACE_PARAMETERS_TAG = "huggingface_parameters"
PARAMETERS_ENV_NAME = "PREDICTIVE_UNIT_PARAMETERS"


class InvalidTranformerInitialisation(MLServerError):
    def __init__(self, code: int, reason: str):
        super().__init__(
            f"Huggingface server failed with {code}, {reason}",
            status_code=code,
        )


class HuggingFaceWithAccelerateSettings(BaseSettings):
    """
    Parameters that apply only to alibi huggingface models
    """

    class Config:
        env_prefix = ENV_PREFIX_HUGGINGFACE_SETTINGS

    task: str = ""
    auto_loader_name: Optional[str] = "AutoModel"
    pretrained_model: Optional[str] = None
    model_parameters: Optional[Dict] = None
    pretrained_tokenizer: Optional[str] = None


def parse_parameters_from_env() -> Dict:
    """
    TODO
    """
    parameters = json.loads(os.environ.get(PARAMETERS_ENV_NAME, "[]"))

    type_dict = {
        "INT": int,
        "FLOAT": float,
        "DOUBLE": float,
        "STRING": str,
        "BOOL": bool,
    }

    parsed_parameters = {}
    for param in parameters:
        name = param.get("name")
        value = param.get("value")
        type_ = param.get("type")
        if type_ == "BOOL":
            parsed_parameters[name] = bool(strtobool(value))
        else:
            try:
                parsed_parameters[name] = type_dict[type_](value)
            except ValueError:
                raise InvalidTranformerInitialisation(
                    "Bad model parameter: "
                    + name
                    + " with value "
                    + value
                    + " can't be parsed as a "
                    + type_,
                    reason="MICROSERVICE_BAD_PARAMETER",
                )
            except KeyError:
                raise InvalidTranformerInitialisation(
                    "Bad model parameter type: "
                    + type_
                    + " valid are INT, FLOAT, DOUBLE, STRING, BOOL",
                    reason="MICROSERVICE_BAD_PARAMETER",
                )
    return parsed_parameters

def _get_model_loader(hf_settings: HuggingFaceWithAccelerateSettings): 
    auto_loader_name = hf_settings.auto_loader_name
    loader = getattr(transformers, auto_loader_name, None)
    return loader

def load_pipeline_from_settings(hf_settings: HuggingFaceWithAccelerateSettings) -> Pipeline:
    """
    TODO
    """
    # TODO: Support URI for locally downloaded artifacts
    # uri = model_parameters.uri

    # if
    model = hf_settings.pretrained_model
    tokenizer = hf_settings.pretrained_tokenizer

    if model and not tokenizer:
        tokenizer = model
    
    # apply the tricks from huggingface accelerate

    path = snapshot_download(model)
    config = AutoConfig.from_pretrained(model, **hf_settings.model_parameters)
    with init_empty_weights():
        model = _get_model_loader(hf_settings).from_config(config)
    model = load_checkpoint_and_dispatch(model, path , 
    device_map=hf_settings.model_parameters.get("device_map", "auto"),
    offload_folder=hf_settings.model_parameters.get("offload_folder", "/tmp/offload"),
    ) 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, **hf_settings.model_parameters)

    return pipeline(
        hf_settings.task,
        model=model,
        tokenizer=tokenizer,
    )


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
