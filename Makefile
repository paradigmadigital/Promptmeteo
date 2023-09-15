setup:
	pip install --upgrade pip
	pip install '.[all]'

dev:
	pip install -e ".[dev]"
	pre-commit install

format:
	black promptmeteo/
	black tests/

docs:
	mkdocs build
	mkdocs serve
	mkdocs gh-deploy

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -f .coverage
	rm -f .coverage.*
	rm -rf ./build
	rm -rf ./.pytest_cache

test:
	pytest --cov=promptmeteo .
