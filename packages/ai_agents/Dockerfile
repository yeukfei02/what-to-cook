FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 \
    AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 \
    DOCKER_CONTAINER=1

COPY . .

RUN uv pip install . && \
    uv pip install aws-opentelemetry-distro>=0.10.1

EXPOSE 8080 8000

CMD ["opentelemetry-instrument", "python", "-m", "app"]
