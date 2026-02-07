FROM python:3.10-slim

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    build-essential \
    ca-certificates \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Prepare YOLOv5
RUN mkdir -p third_party && \
    git clone https://github.com/ultralytics/yolov5.git third_party/yolov5

# 3. Prepare Darknet (YOLOv2, v4) - Compile
WORKDIR /app/third_party
RUN git clone https://github.com/AlexeyAB/darknet.git && \
    cd darknet && \
    sed -i 's/GPU=1/GPU=0/' Makefile && \
    sed -i 's/CUDNN=1/CUDNN=0/' Makefile && \
    sed -i 's/OPENCV=1/OPENCV=0/' Makefile && \
    make

WORKDIR /app

# 4. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Copy Project Files (Explicitly)
COPY src /app/src
COPY utils /app/utils
COPY data /app/data
COPY app.py /app/app.py

COPY run_full_reproduction.sh /app/run_full_reproduction.sh
RUN chmod +x /app/run_full_reproduction.sh

# 6. Environment Setup
ENV PYTHONPATH=/app:/app/third_party/yolov5
ENV PYTHONUNBUFFERED=1
ENV WANDB_MODE=disabled

EXPOSE 7860

CMD ["python", "app.py"]