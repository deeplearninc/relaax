.PHONY: clean-pyc clean-docs clean-build docs test

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-docs - remove documentation artifacts"
	@echo "clean-test - remove artifacts created by test"
	@echo "clean-pyc - remove Python file artifacts"	
	@echo "clean - remove artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "sdist - package"

clean: clean-build clean-docs clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	find . -name '*.egg-info' -type d -exec rm -rf {} +
	rm -fr *.egg

clean-docs:
	# $(MAKE) -C doc clean

clean-pyc:
	find . -name '*.pyc' -type f -exec rm -f {} +
	find . -name '*.pyo' -type f -exec rm -f {} +
	find . -name '*~' -type f -exec rm -f {} +
	find . -name '__pycache__' -type d -exec rm -rf {} +

clean-test:
	rm -fr coverage/
	rm -fr .tox/

lint:
	flake8

test: claen-test
	mkdir coverage
	pytest --cov=relaax --cov-report html:coverage tests/

docs:
	$(MAKE) -C doc html

# release: clean
# 	python setup.py sdist bdist_wheel upload

sdist: clean
	python setup.py sdist
	ls -l dist