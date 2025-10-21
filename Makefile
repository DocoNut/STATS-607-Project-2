VENV   := .venv
PY     := $(VENV)/bin/python3
PIP    := $(VENV)/bin/pip3
CONFIG ?= src/config.json

.PHONY: all simulate venv install clean test

all: simulate

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	@$(PIP) -q install -U pip
	@test -f requirements.txt && $(PIP) -q install -r requirements.txt || true

simulate: install
	@$(PY) src/simulation.py --config $(CONFIG)

test: install
	@$(PY) tests/test.py

clean:
	@rm results/raw/* results/figures/*
