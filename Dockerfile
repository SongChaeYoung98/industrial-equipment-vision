FROM python:3.12-slim

WORKDIR /app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY app ./app
COPY configs ./configs
COPY models ./models

EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]

