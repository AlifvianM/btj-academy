from typing import List
from pydantic import BaseModel

class PredictionParams(BaseModel):
    sepal_length: int = 0
    sepal_width: int = 0
    petal_length: int = 0
    petal_width: int = 0

class PredictionResult(BaseModel):
    message: str
    result: List[int]