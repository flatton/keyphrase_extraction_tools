from pydantic import BaseModel

from ..io_data import Keyphrase


class ResponseSchema(BaseModel):
    keyphrases: list[Keyphrase]
