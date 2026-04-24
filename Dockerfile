FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir outputs

ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONHASHSEED=42
ENV CUDA_VISIBLE_DEVICES=-1
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]