#mcandrew

PYTHON ?= python3 -W ignore
STATES ?= 'all'

VENV_DIR := .whoseason
VENV_PYTHON := $(VENV_DIR)/bin/python -W ignore

R ?= Rscript

run: build_env format_data

build_env:
	@echo "build forecast environment"
	@$(PYTHON) -m venv ./$(VENV_DIR)
	@$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "Analysis environment installed"

format_WHO_data:
	@echo "Format WHO data"
	@$(VENV_PYTHON) ./compute_cases_per_week_per_country.py


