from __future__ import annotations  # отложенная обработка аннотаций типов

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)  # создание Flask приложения
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"

pipeline = joblib.load(MODEL_PATH)


def _to_dataframe(features: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """
        - Если пришёл dict -> делаем 1 строку
        - Если пришёл list[dict] -> делаем много строк (batch)
    """
    if isinstance(features, dict):
        return pd.DataFrame([features])
    if isinstance(features, list):
        if not all(isinstance(x, dict) for x in features):
            raise ValueError("Если features — список, то каждый элемент должен быть объектом (dict).")
        return pd.DataFrame(features)

    raise ValueError("Поле features должно быть либо объектом (dict), либо списком объектов (list[dict]).")


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict() -> Any:
    """
    Основной эндпоинт предсказания.
    Принимает JSON с ключом 'features'.
    Возвращает JSON с predictions (+ probabilities если доступны).
    """

    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body должен быть JSON"}), 400

    if "features" not in data:
        return jsonify({"error": "В JSON должен быть ключ 'features'"}), 400

    try:
        X = _to_dataframe(data["features"])

        preds = pipeline.predict(X)

        probs = None
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)

        # Готовим ответ: numpy -> list (чтобы JSON сериализовался)
        response = {
            "predictions": preds.tolist()
        }
        if probs is not None:
            response["probabilities"] = probs.tolist()

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)  # чтобы контейнер снаружи был доступен нуженн такой хост