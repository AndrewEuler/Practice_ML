import mlflow

mlflow.set_experiment("smoke-test")

mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.99)

print("OK")