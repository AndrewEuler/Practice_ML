from pathlib import Path

import joblib
import pandas as pd

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    pipeline_path = project_dir / "models" / "model.joblib"

    preprocess_path = project_dir / "models" / "preprocess.joblib"
    onnx_path = project_dir / "models" / "model.onnx"

    pipeline = joblib.load(pipeline_path)

    preprocess = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Сохраняем препроцессор отдельно (joblib)
    # preprocess_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocess, preprocess_path)
    print(f"Saved: {preprocess_path}")

    # Берём X_test и прогоняем через preprocess -> получаем float матрицу
    X_test_raw = pd.read_csv(project_dir / "data" / "splits" / "X_test.csv")
    X_test_num = preprocess.transform(X_test_raw)

    if hasattr(X_test_num, "toarray"):
        X_test_num = X_test_num.toarray()  # преобразуем в обычный numpy-массив, так как данные могут быть
        # разреженной матрицей

    X_test_num = X_test_num.astype("float32")
    n_features = X_test_num.shape[1]

    # Экспортируем в ONNX только RandomForest
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx_path.write_bytes(onnx_model.SerializeToString())
    print(f"Saved: {onnx_path}")


if __name__ == "__main__":
    main()