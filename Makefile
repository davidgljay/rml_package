PYTHON = "python3"

tooling_venv = "tooling/venv"
poetry = "$(tooling_venv)/bin/poetry"

.PHONY: all clean tooling

all: tooling

tooling:
	cd tooling && $(MAKE)

clean:
	cd tooling && $(MAKE) clean
