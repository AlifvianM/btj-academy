import pandas as pd
import os

from dotenv import load_dotenv
from typing import Union

from fastapi import FastAPI

load_dotenv(dotenv_path='.dev.env')
api_key = os.getenv("GITHUB_API")

app = FastAPI(title="TEST PROJECT 1")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(
        item_id: int, 
        q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/")
def predict(data_id: int, val: int):

    result = "Model has been predicted with result TRUE"

    return {
        "data_id":data_id,
        "val": val,
        "result":result
    }