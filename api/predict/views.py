from fastapi import APIRouter, Request, HTTPException, status

from api.predict.service import Predict
from api.predict.schemas import PredictionParams, PredictionResult

predict_router = APIRouter()
tag = ["Predict"]

@predict_router.post('/predict', tags=tag)
def predict_route(
    request: Request,
    data: PredictionParams
)->PredictionResult:
    
    predict = Predict(params=data)
    pred= predict.predict()
    
    try:
        message = pred.get('data', "")
        result = pred.get("result")
        response = PredictionResult(
            message=message,
            result=result
        )
        return response
    except:
        raise HTTPException(
            status_code=404, 
            detail=PredictionResult(
                message=pred.get('data', ""),
                result=pred.get('result', []),
            ).model_dump()
        )
        
        