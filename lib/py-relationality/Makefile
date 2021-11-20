PYTHON = "python3"

tooling_venv = "tooling-venv"
poetry = "$(tooling_venv)/bin/poetry"

lib = "src/relationality"
tests = "tests/relationality"

venv = ".venv"
pytest = ".venv/bin/pytest"
pylint = ".venv/bin/pylint"

.PHONY: all clean test lint

all: test lint

test: .dev
	$(pytest)

lint: .dev
	$(pylint) $(lib) $(tests)

.dev: .tooling pyproject.toml poetry.toml
	$(poetry) install
	touch .dev

.tooling: tooling-requirements.txt
	PYTHON=$(PYTHON) bash setup
	touch .tooling

clean-cache:
	find . -type d -name '__pycache__'   -prune -exec rm -rf "{}" \+	
	find . -type d -name '.mypy_cache'   -prune -exec rm -rf "{}" \+	
	find . -type d -name '.pytest_cache' -prune -exec rm -rf "{}" \+	
	find . -type d -name '*.egg-info'    -prune -exec rm -rf "{}" \+	

clean: clean-cache
	rm -f .tooling .dev
	rm -rf $(venv) $(tooling_venv)
