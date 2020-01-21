# Working dir
WORKON_HOME := $(abspath $(dir $(realpath $(firstword $(MAKEFILE_LIST)))))
# Path to environment
ENVPATH := $(WORKON_HOME)/env
# Option that needs to be given many times
PENV := --prefix $(ENVPATH)
BUILDSUFFIXDIR := build
BUILDDIR := $(WORKON_HOME)/$(BUILDSUFFIXDIR)

ifndef RPMVER
	# Actual value check will happen later, in a rule. It cannot happen here.
	RPMVER := 1
endif

## Setup rules
devinit: purge envsetup

# Cleans all generated files.
purge: cleanbuild
	# Actual env
	rm -rf $(ENVPATH)

cleanbuild:
	# Build droppings
	rm -rf $(BUILDDIR)

envsetup:
	# Will install all dev dependencies
	test -d $(ENVPATH) || conda env create $(PENV) --file ./conda-env.yaml
	# Will install pip-only dependencies (ptpython only for now)
	$(ENVPATH)/bin/pip install -r requirements.txt

## Tests
# Checks pep8.
pep8:
	$(ENVPATH)/bin/pycodestyle --show-source phos --max-line-length 120

# Runs unittest and coverage.
unittest:
	$(ENVPATH)/bin/coverage erase
	$(ENVPATH)/bin/coverage run --include "phos*" --omit "*test*" -m unittest discover -v
	$(ENVPATH)/bin/coverage report
	$(ENVPATH)/bin/coverage html

typing:
	# --ignore-missing-imports to avoid warnings about 3rd party modules not having stubs.
	$(ENVPATH)/bin/mypy --ignore-missing-imports $(WORKON_HOME)/phos

test: unittest pep8 typing

