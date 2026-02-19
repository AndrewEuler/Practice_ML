# Streamlit ML Dashboard (Classification)

Интерактивное приложение для:

- просмотра датасета (почти любого)
- фильтрации данных
- визуализации
- обучения модели классификации
- получения предсказаний

---

## Функциональность

### 1. Просмотр данных
- Таблица датасета
- Типы колонок
- Количество пропусков

### 2. Фильтрация
- Фильтры по числовым признакам (range slider)
- Фильтры по категориальным признакам (multiselect)

### 3. Визуализация
- Histogram для числовых признаков
- Bar chart для категориальных

### 4. Обучение модели
- Выбор target-колонки
- Исключение ненужных признаков
- Авто-исключение ID-подобных колонок
- Не затюнингованный RandomForest + preprocessing pipeline

### 5. Предсказание
- Ввод признаков вручную
- Предсказанный класс
- Вероятности по классам

---

## Архитектура

Проект разделён на:

- app.py UI (Streamlit)
- train.py Логика обучения + pipeline
- data/ (dataset.csv котоырй необходимо загрузить заранее)
- models/ (сохранённая модель + meta.json)


Используется:
- `Pipeline`
- `ColumnTransformer`
- `OneHotEncoder`
- `StandardScaler`
- `LogisticRegression`

---

## Результаты модели

- **Target:** `{{ TARGET_NAME }}`  
- **Accuracy:** `{{ Accuracy }}`  
- **F1 macro:** `{{ F1 }}`
- **Precision:** `{{ Precision }}`
- **Recall:** `{{ Recall }}`

---

## Установка

```bash
pip install streamlit pandas matplotlib scikit-learn joblib

