#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mates
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python$(PYTHON_VERSION)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


.PHONY: dependencies
dependencies:
	poetry update


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using pylint
.PHONY: lint
lint:
	@pylint $$(git ls-files '*.py')
	@isort --check --diff --profile black mates tests 
	@black --check --config pyproject.toml mates tests

.PHONY: test
test:
	pytest

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml mates


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Run preprocessing
.PHONY: prepare
prepare:
	$(PYTHON_INTERPRETER) -m mates.modeling.prepare

.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mates.modeling.train

.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m mates.modeling.predict


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
