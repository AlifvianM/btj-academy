from typing import List, Optional
from pydantic import BaseModel

class PredictionParams(BaseModel):
    sepal_length: int
    sepal_width: int
    petal_length: int
    petal_width: int

class PredictionResult(BaseModel):
    message: str
    result: List[int]