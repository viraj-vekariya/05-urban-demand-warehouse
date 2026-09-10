# Urban Demand Warehouse
#
#   make setup     install dependencies
#   make data      download 12 months of NYC TLC trip records (~700 MB)
#   make etl       incremental load into partitioned Parquet
#   make build     SQL warehouse + reconciliation gate
#   make backtest  allocation policies on held-out months
#   make analyse   forecast, seasonal decomposition, consolidated results
#   make spark     the same aggregation in PySpark, verified against DuckDB
#   make serve     the dashboard on :8500
#   make all       data -> etl -> build -> backtest -> analyse -> test

PY        ?= python3
JAVA_HOME ?= $(shell /usr/libexec/java_home -v 21 2>/dev/null || echo /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home)

.PHONY: setup data etl build backtest analyse spark test serve docker clean all

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) data/fetch.py --year 2024 --months 12

etl:
	$(PY) -m etl.incremental

build:
	$(PY) -m src.build

backtest:
	$(PY) -m src.backtest

analyse:
	$(PY) -m src.report

spark:
	JAVA_HOME=$(JAVA_HOME) $(PY) -m etl.spark_job --verify --months 3

test:
	$(PY) -m pytest tests/ -q

serve:
	$(PY) -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8500

docker:
	docker compose up --build

all: data etl build backtest analyse test

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache data/urban.duckdb data/warehouse
