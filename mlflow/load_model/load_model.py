import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression

mlflow.set_tracking_uri('http://127.0.0.1:5001')


model_name = "PipelineModelLRTest"
model_version = "latest"

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

# Generate a new dataset for prediction and predict
# X_new, _ = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X_new = [[5. , 3. , 1.6, 0.2]]
y_pred_new = model.predict(X_new)

print(y_pred_new)