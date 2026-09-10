# Dashboard image.
#
# The warehouse is built at IMAGE BUILD TIME from data downloaded during the build. That
# is a deliberate trade: the image is larger, but it starts instantly, needs no network at
# runtime, and shows the exact numbers that were reviewed rather than whatever TLC serves
# today.
FROM python:3.12-slim AS build
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# pyspark is excluded from the image: it needs a JVM and ~400MB, and the serving path
# never touches it. It stays a development and verification dependency.
RUN grep -v '^pyspark' requirements.txt > runtime-requirements.txt && \
    pip install --no-cache-dir --prefix=/install -r runtime-requirements.txt

COPY data/fetch.py ./data/
COPY etl/ ./etl/
COPY sql/ ./sql/
COPY src/ ./src/
ENV PYTHONPATH=/install/lib/python3.12/site-packages:/build PYTHONUNBUFFERED=1

# Three months rather than twelve: the image would otherwise carry ~700 MB of raw
# Parquet. The full pipeline is identical, and `make all` locally reproduces all twelve.
ARG MONTHS=3
RUN python data/fetch.py --year 2024 --months ${MONTHS} && \
    python -m etl.incremental && \
    python -m src.build && \
    python -m src.backtest && \
    python -m src.report && \
    rm -rf data/raw/*.parquet

# ---------------------------------------------------------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 urban
COPY --from=build /install /usr/local
WORKDIR /app

COPY dashboard/ ./dashboard/
COPY src/ ./src/
COPY --from=build /build/data/urban.duckdb ./data/
COPY --from=build /build/data/warehouse/ ./data/warehouse/
COPY --from=build /build/data/raw/taxi_zone_lookup.csv ./data/raw/
COPY --from=build /build/outputs/ ./outputs/

RUN chown -R urban:urban /app
USER urban

ENV PYTHONUNBUFFERED=1 URBAN_DB=/app/data/urban.duckdb
EXPOSE 8500
HEALTHCHECK --interval=20s --timeout=4s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8500/health',timeout=3).status==200 else 1)"

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8500"]
