# 1) Базовый Python образ (slim = меньше размер)
FROM python:3.10-slim

# 2) Чтобы логи сразу печатались в консоль, без буферизации
ENV PYTHONUNBUFFERED=1

# 3) Рабочая директория внутри контейнера
WORKDIR /app

# 4) Копируем только requirements сначала (для кеша Docker)
COPY requirements.txt .

# 5) Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# 6) Копируем код и модели
COPY app.py .
COPY models ./models

# 7) Открываем порт (документация для Docker; реально пробрасывает run/compose)
EXPOSE 5000

# 8) Команда запуска
CMD ["python", "app.py"]
