from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def run_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Метрики на test-set:")
    print(f"  accuracy : {accuracy:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  recall   : {recall:.4f}")
    print(f"  f1       : {f1:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def main() -> None:
    project_dir = Path(__file__).resolve().parent  # беру папку откуда test_inference.py
    x_test_path = project_dir / "data" / "splits" / "X_test.csv"
    y_test_path = project_dir / "data" / "splits" / "Y_test.csv"

    preprocess_path = project_dir / "models" / "preprocess.joblib"
    onnx_path = project_dir / "models" / "model.onnx"

    # Загружаем тестовые данные
    X_test_raw = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")  # превращаем 1-колоночный DF в Series

    # Препроцессинг
    preprocess = joblib.load(preprocess_path)
    X_num = preprocess.transform(X_test_raw)

    # Загружаем ONNX и запускаем inference
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Смотрим имя входа
    input_name = sess.get_inputs()[0].name

    X_num = X_num.astype(np.float32)
    outputs = sess.run(None, {input_name: X_num})

    y_pred = np.array(outputs[0]).reshape(-1)

    # Иногда onnx отдаёт тип не int -> приведём
    y_pred = y_pred.astype(int)

    # Метрики
    run_metrics(y_test, y_pred)

    # Примеры
    print("\nSample predictions (первые 5 строк):")
    sample = X_test_raw.head(5).copy()
    sample["y_test"] = y_test.head(5).values
    sample["y_pred"] = y_pred[:5]
    print(sample[["y_test", "y_pred"]])


if __name__ == "__main__":
    main()