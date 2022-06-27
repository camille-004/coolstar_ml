.PHONY: create_env install

ENVIRONMENT_FILE = environment.yml

create_env:
	@echo ">>> Creating conda environment. Run 'conda activate $(ENVIRONMENT_FILE)' after this finishes."
	conda env create -f $(ENVIRONMENT_FILE)

install:
	@echo ">>> Updating conda environment."
	conda env update -f $(ENVIRONMENT_FILE)

lint:
	isort .
	black *.py
	flake8
#	mypy .

all: install lint