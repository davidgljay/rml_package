poetry = "tooling/venv/bin/poetry"

lib = "src/relationality"
tests = "tests/relationality"

venv = ".venv"

.PHONY: all clean test lint

all: typecheck test lint

test: .dev
	$(poetry) run pytest

lint: .dev
	$(poetry) run pylint $(lib) $(tests)

typecheck: .dev
	$(poetry) run mypy src tests

.dev: pyproject.toml poetry.toml
	$(poetry) install
	touch .dev

# Could use a rule like this
# $(poetry):
#	cd tooling && make

clean-cache:
	find . -type d -name '__pycache__'   -prune -exec rm -rf "{}" \+	
	find . -type d -name '.mypy_cache'   -prune -exec rm -rf "{}" \+	
	find . -type d -name '.pytest_cache' -prune -exec rm -rf "{}" \+	
	find . -type d -name '*.egg-info'    -prune -exec rm -rf "{}" \+	

clean: clean-cache
	rm -rf .dev $(venv)
