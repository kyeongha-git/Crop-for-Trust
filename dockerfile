FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 wget git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

RUN chmod +x setup_docker.sh && ./setup_docker.sh
RUN chmod +x docker_entrypoint.sh

ENTRYPOINT ["./docker_entrypoint.sh"]