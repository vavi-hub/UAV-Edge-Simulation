FROM python:3.11-slim

WORKDIR /app

COPY api.py .
COPY flower_server.py .
COPY static ./static

RUN pip install fastapi uvicorn flwr websockets

CMD ["sh", "-c", "python flower_server.py & uvicorn api:app --host 0.0.0.0 --port 8000"]