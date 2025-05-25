FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

RUN apt update && apt install -y \
    libmysqlclient-dev \
    default-libmysqlclient-dev \
    pkg-config \
    build-essential \
 && apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace

RUN pip install -r /workspace/requirements.txt

COPY src /workspace/src
COPY start_api_server.sh /workspace

WORKDIR /workspace