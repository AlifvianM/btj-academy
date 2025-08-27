import pandas as pd
import os

from dotenv import load_dotenv
from typing import Union

from fastapi import FastAPI

from api.predict.views import predict_router
from api.scheduler.views import scheduler_router

load_dotenv(dotenv_path='.dev.env')

app = FastAPI(title="TEST PROJECT 1")
app.include_router(predict_router)
app.include_router(scheduler_router)

@app.get("/")
def health_check():
    return {"Hello": "World"}