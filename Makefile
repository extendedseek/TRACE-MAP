.PHONY: install install-full test lint format smoke package

install:
	python -m pip install -e .

install-full:
	python -m pip install -e ".[train,language,taxai,dev]"

test:
	python -m unittest discover -s tests -v

lint:
	python -m compileall -q src tests
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

smoke:
	trace-map smoke --config configs/smoke.yaml --output results/generated/smoke

package:
	python -m build
