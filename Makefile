.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

formatter:
	black seq2annotaion tests

lint: ## check style with flake8
	flake8 seq2annotation tests
	black --check seq2annotation tests

types:
	pytype --keep-going seq2annotation

test: test_install ## run tests quickly with the default Python
	py.test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source tokenizer_tools -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/tokenizer_tools.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ tokenizer_tools
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

nightly_release: nightly_dist ## package and upload a release
	twine upload dist/*

nightly_dist: clean ## builds source and wheel package
	_PKG_NAME=s2a-nightly python setup.py egg_info --tag-date --tag-build=DEV sdist
	_PKG_NAME=s2a-nightly python setup.py egg_info --tag-date --tag-build=DEV bdist_egg
	ls -l dist

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

.PHONY: test_install
test_install:
	pip install -r test_requirements.txt

.PHONY: dev_install
dev_install:
	pip install -r dev_requirements.txt

# version update

.PHONY: update_minor_version
update_minor_version:
	bumpversion minor

.PHONY: update_patch_version
update_patch_version:
	bumpversion patch

.PHONY: update_major_version
update_major_version:
	bumpversion major

# build docker image for production

.PHONY: build_docker
build_docker: build_docker_base build_docker_trainer build_docker_server

.PHONY: build_docker_base
build_docker_base:
	docker build --no-cache --force-rm --tag ner_base --file docker_v2/stable/base/Dockerfile docker_v2/stable/base/

.PHONY: build_docker_trainer
build_docker_trainer:
	docker rmi -f ner_trainer
	docker build --no-cache --force-rm --tag ner_trainer --file docker_v2/stable/trainer/Dockerfile docker_v2/stable/trainer/

.PHONY: build_docker_server
build_docker_server:
	docker rmi -f ner_server
	docker build --no-cache --force-rm --tag ner_server --file docker_v2/stable/server/Dockerfile docker_v2/stable/server/

# build docker image for testing

.PHONY: build_docker_nightly
build_docker_nightly: build_docker_nightly_trainer build_docker_nightly_server

.PHONY: build_docker_nightly_trainer
build_docker_nightly_trainer: dist
	cp -r dist docker_v2/nightly/trainer/
	docker rmi -f ner_trainer
	docker build --no-cache --force-rm --tag ner_trainer --file docker_v2/nightly/trainer/Dockerfile docker_v2/nightly/trainer/

.PHONY: build_docker_nightly_server
build_docker_nightly_server: dist
	cp -r dist docker_v2/nightly/server/
	docker rmi -f ner_server
	docker build --no-cache --force-rm --tag ner_server --file docker_v2/nightly/server/Dockerfile docker_v2/nightly/server/

.PHONY: run_docker_nightly_server
run_docker_nightly_server:
	docker run --rm -p 5000:5000 -v /home/howl/workshop/seq2annotation_keras_ner_on_ecarx/results/deliverable_model:/model ner_server

.PHONY: run_docker_nightly_trainer
run_docker_nightly_trainer:
	docker run --runtime=nvidia --rm -v /home/howl/PycharmProjects/seq2annotation:/data ner_trainer
