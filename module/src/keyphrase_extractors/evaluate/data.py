import numpy as np
from pydantic import BaseModel, ConfigDict, Field


TYPE_FLOAT = float | np.float_


class EvaluationSample(BaseModel):
    dataset_name: str
    id: str | int
    text: str
    keyphrase_list: list[list[str]]


class Score(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    precision: TYPE_FLOAT = Field(ge=0, le=1)
    recall: TYPE_FLOAT = Field(ge=0, le=1)
    hitrate: TYPE_FLOAT = Field(ge=0, le=1)
    lcs_by_truthset: TYPE_FLOAT = Field(ge=0, le=1)
    lcs_by_pred: TYPE_FLOAT = Field(ge=0, le=1)


class Stats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mean: TYPE_FLOAT
    std: TYPE_FLOAT
    max: TYPE_FLOAT
    min: TYPE_FLOAT
