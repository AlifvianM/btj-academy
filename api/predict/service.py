import pickle
import joblib
import mlflow

from api.predict.schemas import PredictionParams

from pydantic_core import ValidationError



IRIS_MODELS_PATH = "api/predict/models/iris_test.pkl"
SCALER_PATH = "api/predict/models/minmax_scaler.pkl"

class Predict:

    def __init__(
            self,
            params: PredictionParams
        ):
        self.params = params

    def predict(self):
        try:
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
                "data":f"Prediction process for data is Success!"
            }
        except ValueError as e:
            return {
                "data":f"Prediction process for data is not success because {e}"
            }

    def predict_v2(self):
        mlflow.set_tracking_uri('http://127.0.0.1:5001')


        model_name = "PipelineModelLRTest"
        model_version = "latest"

        # Load the model from the Model Registry
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)

        # Generate a new dataset for prediction and predict
        # X_new, _ = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
        X_new = [[self.params.sepal_length, self.params.sepal_width, self.params.petal_length, self.params.petal_width]]
        y_pred_new = model.predict(X_new)
        return {
            "result":y_pred_new,
            "data":f"Prediction process for data is Success!"
        }
        
def health_check():
    return{'message':"It's Okay!"}