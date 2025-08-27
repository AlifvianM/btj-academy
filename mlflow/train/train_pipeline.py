import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("iris_models_pipeline")

ARTIFACT_PATH="l2_iris_lr_pipeline"

iris = load_iris()
df = pd.DataFrame(
    iris['data'],
    columns=iris['feature_names']
)
df['target'] = iris['target']
X = df.drop('target', axis=1)
y = df['target']

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="iris_lr_pipeline_test") as run:
    params={
        "penalty":"l2"
    }
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    accuracy_score = accuracy_score(y_true=y_test, y_pred=y_pred)
    recall_score = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
    f1_score = f1_score(y_true=y_test, y_pred=y_pred, average='macro')

    mlflow.log_params(params=params)

    print(f"{params['penalty']} model result")
    print(f"Accuracy {accuracy_score}")
    print(f"Recall {recall_score}")
    print(f"F1 Score {f1_score}")

    mlflow.log_metrics(metrics={
        "accuracy_score":accuracy_score,
        "recall_score":recall_score,
        "f1_score":f1_score
    })

    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    mlflow.sklearn.log_model(sk_model=pipeline, input_example=X_test, name=ARTIFACT_PATH, registered_model_name="PipelineModelLRTest")
