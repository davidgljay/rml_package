# Relationality
Python/Numpy library for experimenting with ideas about relationality.

## Setup Development Environment

To setup the development environment
```
make test PYTHON=<location of a python 3.8 interpreter>
```

This will do the following:
1. Setup a virtual environment `tooling-venv` with `poetry` in it for building and package managment.
2. Use `poetry` to build a virtual environment `.venv` for the `relationality` library.
3. Run tests on this environment.

Use `./.venv/bin/ipython` to enter an ipython CLI environment with access to the library and its dependencies.
