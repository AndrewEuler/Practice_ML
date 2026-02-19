import os
import joblib
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/pipeline.joblib"
META_PATH = "models/meta.json"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def detected_id_like_columns(X: pd.DataFrame, threshold: float) -> list[str]:
    # возвращаю список столбцов, которые выглядят как id. можно вырубить поставив значение > 1.0
    n = len(X)
    if n == 0:
        return []
    id_like = []
    for col in X.columns:
        nunique = X[col].nunique(dropna=False)
        if nunique / n > threshold:
            id_like.append(col)
    return id_like


def build_pipline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # заполняем пропуски медианой столбца
        ("scaler", StandardScaler()),  # нормализуем количественные признаки
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # заполняем пропуски популярным значением
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # кодируем категориальные признаки /
        # если встретится новая категория в test — проигнорим ее
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"  # удалить все колонки, которые не указаны в transformers
    )

    model = RandomForestClassifier(
        n_estimators=600,  # количество деревьев
        max_depth=8,  # деревья могут расти свободно
        max_features="sqrt",
        class_weight="balanced",
        random_state=42
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str] | None = None,
    id_like_threshold: float = 0.95,
):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' не найдена. Колонки: {list(df.columns)}")

    if y.isna().any():
        raise ValueError("В target есть NaN. Удали/заполни пропуски в target или выбери другой target.")

    if exclude_cols:
        exclude_cols = [c for c in exclude_cols if c in X.columns]
    else:
        exclude_cols = []

    # Подготовка столбцов

    X = X.drop(columns=exclude_cols, errors="ignore")

    auto_drop = detected_id_like_columns(X, threshold=id_like_threshold)
    X = X.drop(columns=auto_drop, errors="ignore")

    if X.shape[1] == 0:
        raise ValueError("После удаления столбцов не осталось ни одного признака.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipline(X_train)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "n_classes": int(y.nunique()),
        "target": target_col,
        "excluded_manual": exclude_cols,
        "excluded_auto_id_like": auto_drop,
        "id_like_threshold": id_like_threshold,
        "n_features_used": int(X.shape[1]),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return pipeline, metrics, cm, report


def save_pipeline_with_meta(pipeline: Pipeline, model_path: str,  meta_path: str, meta: dict):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)  # сохраняем весь pipeline в файл
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)  # сохраняем мета данные в json


def load_meta(meta_path: str) -> dict | None:
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)  # загружаем мета данные из json


def load_pipeline(model_path: str):
    return joblib.load(model_path)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Не найден {DATA_PATH}")

    TARGET_COL = "Survived"
    MANUAL_EXCLUDE = ["PassengerId"]
    ID_LIKE_THRESHOLD = 0.95

    df = load_data(DATA_PATH)

    pipeline, metrics, cm, report = train_and_evaluate(df, TARGET_COL, MANUAL_EXCLUDE, ID_LIKE_THRESHOLD)

    meta_metrics = {
            "target": metrics["target"],
            "excluded_manual": metrics["excluded_manual"],
            "excluded_auto_id_like": metrics["excluded_auto_id_like"],
            "id_like_threshold": metrics["id_like_threshold"],
            "n_features_used": metrics["n_features_used"],
    }

    print("=== Метрики ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== Матрица ошибок ===")
    print(cm)

    print("\n=== Репорт о классификации ===")
    print(report)

    save_pipeline_with_meta(pipeline, MODEL_PATH, META_PATH, meta_metrics)
    print(f"\nSaved: {MODEL_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()