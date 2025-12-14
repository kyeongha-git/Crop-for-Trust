FROM python:3.10-slim

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Prepare YOLOv5 (Build-time Clone)
RUN mkdir -p third_party && \
    git clone https://github.com/ultralytics/yolov5.git third_party/yolov5

# 3. Install Python Dependencies
RUN pip install --no-cache-dir jinja2==3.1.4
RUN pip install --no-cache-dir "numpy<2.0"
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r third_party/yolov5/requirements.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Project Files
COPY src /app/src
COPY utils /app/utils
COPY data/sample /app/data/sample
COPY run_demo.sh /app/run_demo.sh
RUN chmod +x /app/run_demo.sh

# 5. Environment Setup
ENV PYTHONPATH=/app:/app/third_party/yolov5
ENV PYTHONUNBUFFERED=1
ENV WANDB_MODE=disabled

ENTRYPOINT ["/app/run_demo.sh"]