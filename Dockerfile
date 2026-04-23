FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-webhook.txt /app/requirements-webhook.txt
RUN pip install --no-cache-dir -r /app/requirements-webhook.txt

COPY . /app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "examples.webhook_app:app", "--host", "0.0.0.0", "--port", "8000"]
