FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    libeigen3-dev \
    libceres-dev \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

WORKDIR /app/engine
RUN mkdir build && cd build && cmake .. && make

WORKDIR /app/api
RUN pip3 install fastapi uvicorn numpy scipy

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
