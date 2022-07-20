.PHONY: create_env install

ENVIRONMENT_FILE = environment.yaml

create_env:
	@echo ">>> Creating conda environment. Run 'conda activate $(ENVIRONMENT_FILE)' after this finishes."
	conda env create -f $(ENVIRONMENT_FILE)

install:
	@echo ">>> Updating conda environment."
	conda env update -f $(ENVIRONMENT_FILE)

lint:
	@echo ">>> Linting code with isort, black, flake8, and mypy."
	isort .
	black *.py
	pylint *.py

all: install lint
