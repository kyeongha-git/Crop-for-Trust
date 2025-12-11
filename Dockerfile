FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir jinja2==3.1.4
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src /app/src
COPY utils /app/utils
COPY saved_model/yolo_cropper/yolov8s.pt /app/saved_model/yolo_cropper/yolov8s.pt
COPY data/sample /app/data/sample

COPY run_demo.sh /app/run_demo.sh
RUN chmod +x /app/run_demo.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV WANDB_MODE=disabled

ENTRYPOINT ["/app/run_demo.sh"]