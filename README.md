# Практика ML (задания по учебе)

**Важно! Все мои основные проекты не публичные.**

## 📂 Структура проекта

    Practice_ML/
    │
    ├── Baseline and tune ML/
    │   ├── titanic_eda_ml_grid.ipynb
    │   └── data
    │        └── dataset.scv
    ├── Docker/
    │   ├── Dockerfile
    │   ├── app.py
    │   ├── docker-compose.yml
    │   ├── requirements.txt
    │   └── models/
    │        └── model.joblib
    ├── MLflow/
    │   ├── data/
    │   │    └── dataset.csv
    │   ├── export_mlflow_report.py
    │   ├── mlflow_smoke_test.py
    │   ├── titanic_eda_ml_grid.ipynb
    │   └── train.py
    │
    ├── NLP/
    │   └── NLP_hug.ipynb
    │
    ├── ONNX/
    │   ├── data/
    │   │    ├── dataset.csv
    │   │    └── splits/
    │   │          ├── X_test.csv
    │   │          └── Y_test.csv
    │   ├── best_model.ipynb
    │   ├── export_onnx.py
    │   └── test_inference.py
    │
    ├── Streamlit app/
    │   ├── data/
    │   │    └── dataset.csv
    │   ├── README.md
    │   ├── app.py
    │   └── train.py
    │
    ├── requirements.txt
    └── README.md

## Локальный запуск проекта

### Клонирование репозитория
```bash
git clone <repository_url>
cd <name dir>
```
### Установка зависимостей
```bash
pip install -r requirements.txt
```

## Baseline and tune ML
```bash
cd <Baseline and tune ML>
jupyter notebook
```

## Docker
Для теста *app.py* без Docker.
```bash
python app.py
```
Для работы с Docker.
```bash
docker-compose up --build
```
И отправка запроса (например).
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}}'
```

## MLflow
```bash
cd <MLflow>
mlflow ui --backend-store-uri file:./mlruns
```
Запуск *titanic_eda_ml_grid.ipynb* в отдельном cmd.
```bash
cd <MLflow>
jupyter notebook
```
Выполнение всего notebook.

Запуск *train.py*
```bash
cd <MLflow>
python train.py
```

Также можно экспортировать MLflow в html
```bash
cd <MLflow>
python export_mlflow_report.py
```

## NLP
Выполнение желательно в Google Colab на GPU
Можно и в jupyter, но будет выполняться очень долго.
```bash
cd <NLP>
jupyter notebook
```

## ONNX
Сначала выполнить *best_model.ipynb* для получения модели.
```bash
cd <ONNX>
jupyter notebook
```
После выполняем остальное.
```bash
python export_onnx.py
python test_inference.py
```

## Streamlit app
```bash
cd <Streamlit app>
python train.py
streamlit run app.py
```

## Дополнительно был реализован CI_CD pipeline
Ссылка:
https://github.com/AndrewEuler/CI_CD_projects
