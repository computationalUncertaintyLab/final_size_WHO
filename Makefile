#mcandrew

PYTHON ?= python3 -W ignore

VENV_DIR := .whoseason
VENV_PYTHON := $(VENV_DIR)/bin/python -W ignore

R ?= Rscript

run: build_env format_WHO_data fig1

build_env:
	@echo "build forecast environment"
	@$(PYTHON) -m venv ./$(VENV_DIR)
	@$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "Analysis environment installed"

format_WHO_data:
	@echo "Format WHO data"
	@$(VENV_PYTHON) ./compute_cases_per_week_per_country.py

fig1:
	@echo "Build data for fig 01"
	@$(VENV_PYTHON) ./viz/showcase_data_and_idea/data_for_viz.py
	@echo "Build Fig01"
	@$(VENV_PYTHON) ./viz/showcase_data_and_idea/plot.py



