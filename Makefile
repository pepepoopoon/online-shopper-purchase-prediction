PYTHON ?= python

.PHONY: install smoke train evaluate predict test lint check

install:
	$(PYTHON) -m pip install -e ".[dev]"

smoke:
	PYTHONPATH=src $(PYTHON) -m online_shopper.generate_smoke_data --output data/smoke.csv

train: smoke
	PYTHONPATH=src $(PYTHON) -m online_shopper.train --data data/smoke.csv

evaluate:
	PYTHONPATH=src $(PYTHON) -m online_shopper.evaluate --data data/smoke.csv

predict:
	PYTHONPATH=src $(PYTHON) -m online_shopper.predict --data data/smoke.csv

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

check: lint test
