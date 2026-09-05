FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/trace-map
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --upgrade pip && python -m pip install -e .

COPY tests ./tests
ENTRYPOINT ["trace-map"]
CMD ["smoke", "--config", "configs/smoke.yaml", "--output", "results/generated/docker-smoke"]
