from fastapi import APIRouter

from api.predict.service import Predict
from api.predict.schemas import PredictionParams, PredictionResult

predict_router = APIRouter()
tag = ["Predict"]

@predict_router.post('/predict', tags=tag)
def predict_route(
    data: PredictionParams
):
    
    predict = Predict(params=data)
    pred = predict.predict()
    return PredictionResult(
        message=pred.get('data', ""),
        result=pred.get("result")
    )