from pydantic import BaseModel


class Inputs(BaseModel):
    docs: list[str]


class Keyphrase(BaseModel):
    phrase: str
    score: float


class Outputs(BaseModel):
    keyphrases: list[list[Keyphrase]]
