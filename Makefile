.PHONY: build quality style test

check_dirs := src tests

build:
	python -m build

quality:
	python -m ruff check $(check_dirs)
	python -m compileall -q src tests

style:
	python -m ruff check $(check_dirs) --fix
	python -m ruff format $(check_dirs)

test:
	python -m pytest tests/test_mvp_*.py tests/data/test_converter.py -q
