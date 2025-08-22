import pickle
import joblib

from api.predict.schemas import PredictionParams


IRIS_MODELS_PATH = "api/predict/models/iris_test.pkl"
SCALER_PATH = "api/predict/models/minmax_scaler.pkl"

class Predict:

    def __init__(
            self,
            params: PredictionParams
        ):
        self.params = params

    def predict(self):
        pred_data = [[self.params.sepal_length, self.params.sepal_width, self.params.petal_length, self.params.petal_width]]

        with open(SCALER_PATH, 'rb') as scaler:
            scaler_ = pickle.load(scaler)
            scaler.close()
        
        pred_scaler = scaler_.transform(pred_data)
        print("Load Scaler Success!")

        with open(IRIS_MODELS_PATH, 'rb') as file:
            loaded_model = pickle.load(file)
            print("Load Model Success!")
            file.close()


        results = loaded_model.predict(pred_scaler)            

        return {
            "result":results.tolist(),
            "data":f"Prediction process for data"
        }