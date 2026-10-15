PY ?= python

.PHONY: install lint test test-all data train evaluate explorer all

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

train:
	$(PY) -m house_price train --dataset king_county --jobs 5
	$(PY) -m house_price train --dataset california --jobs 5

evaluate:
	$(PY) -m house_price evaluate --dataset king_county --jobs 5
	$(PY) -m house_price evaluate --dataset california --jobs 5

explorer:
	$(PY) scripts/build_explorer.py

all: train evaluate explorer
