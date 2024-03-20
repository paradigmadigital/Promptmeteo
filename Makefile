setup:
	pip install --upgrade pip
	pip install '.[all]'

dev:
	pip install -e ".[dev]"
	pre-commit install

format:
	black promptmeteo/
	black tests/

docsetup:
	pip install -e ".[docs]"
	pip install -e ".[aws]"
	sphinx-apidoc -f -o docs/source/ promptmeteo

html:
	$(MAKE) -C docs html

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -f .coverage
	rm -f .coverage.*
	rm -rf ./docs/build
	rm -rf ./.pytest_cache

test:
	pytest --cov=promptmeteo .
