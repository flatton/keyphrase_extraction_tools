.PHONY: check ruff-check ruff-format ruff-format-check pyright-check install-kernel

PROJECT_NAME := "keyphrase_extraction"

check:
	$(MAKE) ruff-format $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) ruff-format-check $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) ruff-check $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) pyright-check $(filter-out $@,$(MAKECMDGOALS))

# ruff
ruff-check:
	poetry run ruff check $(filter-out $@,$(MAKECMDGOALS)) --config ./pyproject.toml

ruff-format:
	poetry run ruff format $(filter-out $@,$(MAKECMDGOALS)) --config ./pyproject.toml

ruff-format-check:
	poetry run ruff format --check $(filter-out $@,$(MAKECMDGOALS)) --config ./pyproject.toml

# pyright
pyright-check:
	poetry run pyright $(filter-out $@,$(MAKECMDGOALS)) --project ./pyproject.toml

# development
install-kernel:
	poetry install
	poetry run python -m ipykernel install --user --name $(PROJECT_NAME) --display-name "Python ($(PROJECT_NAME))"
