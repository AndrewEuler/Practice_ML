import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from train import train_and_evaluate, save_pipeline_with_meta, load_meta, load_pipeline

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/pipeline.joblib"
META_PATH = "models/meta.json"  # под какой target обучено и что исключали


st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("Универсальный Dashboard + обучение модели")

if not os.path.exists(DATA_PATH):
    st.error(f"Не найден {DATA_PATH}. Положи dataset.csv в папку data/.")
    st.stop()


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


df = load_data(DATA_PATH)


st.sidebar.header("Настройки обучения")

target_col = st.sidebar.selectbox(
    "Выбери target (что предсказываем)",
    options=df.columns.tolist()
)

feature_cols = [c for c in df.columns if c != target_col]

exclude_cols = st.sidebar.multiselect(
    "Исключить колонки из обучения (например id/имя/коммент)",
    options=feature_cols,
    default=[]
)

id_like_threshold = st.sidebar.slider(
    "Порог авто-исключения ID-подобных колонок (1.0 чтобы отключить авто-исключение)",
    min_value=0.80, max_value=1.0, value=0.95, step=0.01
)

train_btn = st.sidebar.button("Train model", type="primary")

st.sidebar.divider()
st.sidebar.subheader("Сохранённая модель")
meta = load_meta(META_PATH)
if meta:
    st.sidebar.json(meta)
else:
    st.sidebar.info("Модель ещё не обучалась.")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Превью данных")
    st.write(f"Строк: {len(df)} • Колонок: {df.shape[1]}")
    st.dataframe(df.head(30), use_container_width=True)

with col2:
    st.subheader("Сводка")
    st.write("Типы колонок:")
    st.dataframe(df.dtypes.astype(str).to_frame("dtype"), use_container_width=True)
    st.write("Пропуски:")
    st.dataframe(df.isna().sum().to_frame("na_count"), use_container_width=True)

# фильтры данных

st.divider()
st.header("Фильтры данных")

filtered_df = df.copy()

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

for col in numeric_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())

    if min_val != max_val:
        selected_range = st.slider(
            f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        filtered_df = filtered_df[
            (filtered_df[col] >= selected_range[0]) &
            (filtered_df[col] <= selected_range[1])
        ]


categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

for col in categorical_cols:
    unique_vals = df[col].dropna().unique().tolist()

    if len(unique_vals) <= 30:  # защита от 1000 уникальных категорий
        selected_vals = st.multiselect(
            f"{col}",
            options=unique_vals,
            default=unique_vals
        )

        if selected_vals:
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]


st.subheader("Отфильтрованные данные")
st.write(f"Осталось строк: {len(filtered_df)}")
st.dataframe(filtered_df.head(50), use_container_width=True)

st.divider()
st.header("Визуализация")

# Гистограмма
if numeric_cols:
    selected_numeric = st.selectbox("Выбери числовую колонку для распределения", numeric_cols)
    fig, ax = plt.subplots()
    ax.hist(filtered_df[selected_numeric], bins=30)
    st.pyplot(fig)

# Бары
if categorical_cols:
    selected_cat = st.selectbox("Выбери категориальную колонку", categorical_cols)
    st.bar_chart(filtered_df[selected_cat].value_counts())

st.divider()
st.header("Предсказание модели (Predict)")

if not os.path.exists(MODEL_PATH):
    st.warning("Модель не найдена. Сначала обучи модель.")
else:
    pipeline = load_pipeline(MODEL_PATH)
    meta = load_meta(META_PATH) or {}

    # Определяем признаки, которые должны быть в форме
    target_saved = meta.get("target", None)
    excluded_manual = set(meta.get("excluded_manual", []))
    excluded_auto = set(meta.get("excluded_auto_id_like", []))

    if target_saved is None:
        st.warning("meta.json не найден или без target.")
        target_saved = target_col  # fallback

    # Признаки: все колонки датасета кроме target и исключённых
    feature_cols_for_form = [
        c for c in df.columns
        if c != target_saved and c not in excluded_manual and c not in excluded_auto
    ]

    st.caption(f"Модель обучена под target: {target_saved}")
    st.write("Колонки, которые используются для предсказания:", feature_cols_for_form)

    if len(feature_cols_for_form) == 0:
        st.error("Нет признаков для формы предсказания (всё исключили).")
    else:
        with st.form("predict_form"):
            input_row = {}

            for col in feature_cols_for_form:
                series = df[col]

                # Числовые поля
                if pd.api.types.is_numeric_dtype(series):
                    # базовые дефолты
                    default = float(series.median()) if series.notna().any() else 0.0
                    min_v = float(series.min()) if series.notna().any() else 0.0
                    max_v = float(series.max()) if series.notna().any() else 1.0
                    if min_v == max_v:
                        # если константа
                        input_row[col] = st.number_input(col, value=default)
                    else:
                        input_row[col] = st.number_input(col, value=default, min_value=min_v, max_value=max_v)

                # Категориальные поля
                else:
                    options = series.dropna().unique().tolist()
                    options = options[:200]  # защита от слишком большого списка
                    if len(options) == 0:
                        input_row[col] = st.text_input(col, value="")
                    else:
                        input_row[col] = st.selectbox(col, options=options)

            submit = st.form_submit_button("Predict")

        if submit:
            X_one = pd.DataFrame([input_row])

            try:
                pred = pipeline.predict(X_one)[0]
                st.success(f"Предсказанный класс: **{pred}**")

                # вероятности (если модель умеет)
                if hasattr(pipeline, "predict_proba"):  # Есть ли у объекта атрибут с таким именем
                    proba = pipeline.predict_proba(X_one)[0]
                    classes = pipeline.named_steps["model"].classes_

                    proba_df = pd.DataFrame({
                        "class": classes,
                        "probability": proba
                    }).sort_values("probability", ascending=False)

                    st.subheader("Вероятности по классам")
                    st.dataframe(proba_df, use_container_width=True)

                    st.bar_chart(proba_df.set_index("class")["probability"])

            except Exception as e:
                st.error(f"Ошибка предсказания: {e}")


if train_btn:
    st.subheader("Результат обучения")
    try:
        pipeline, metrics, cm, report = train_and_evaluate(
            df=df,
            target_col=target_col,
            exclude_cols=exclude_cols,
            id_like_threshold=id_like_threshold
        )

        save_pipeline_with_meta(
            pipeline,
            model_path=MODEL_PATH,
            meta_path=META_PATH,
            meta={
                "target": metrics["target"],
                "excluded_manual": metrics["excluded_manual"],
                "excluded_auto_id_like": metrics["excluded_auto_id_like"],
                "id_like_threshold": metrics["id_like_threshold"],
                "n_features_used": metrics["n_features_used"],
            }
        )

        st.success("Модель обучена и сохранена")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("F1 macro", f"{metrics['f1_macro']:.4f}")
        m3.metric("Precision", f"{metrics['precision']:.4f}")
        m4.metric("Recall", f"{metrics['recall']:.4f}")
        m5.metric("Classes", metrics["n_classes"])
        m6.metric("Features used", metrics["n_features_used"])

        st.write("Исключённые вручную:", metrics["excluded_manual"])
        st.write("Исключённые автоматически (ID-like):", metrics["excluded_auto_id_like"])

        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(cm), use_container_width=True)  # сделаем из numpy.ndarray датафрейм и
        # растянем таблицу на всю ширину блока

        st.subheader("Classification Report")
        st.code(report)

        st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.sidebar.metric("F1 macro", f"{metrics['f1_macro']:.4f}")

    except Exception as e:
        st.error(f"Ошибка обучения: {e}")