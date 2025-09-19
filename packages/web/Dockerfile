FROM --platform=linux/arm64 python:3.12-slim

RUN mkdir -p /app

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv sync

CMD ["uv", "run", "app.py"]