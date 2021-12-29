PYTHON = "python3"

tooling_venv = "venv"

.PHONY: all clean

all: .tooling

.tooling: tooling-requirements.txt
	PYTHON=$(PYTHON) bash setup
	touch .tooling

clean:
	rm -rf .tooling $(tooling_venv)
