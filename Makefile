# Shortcuts. Run from the project root with the virtual environment active (or use .venv/bin/...).
PY ?= .venv/bin/python

.PHONY: install demo notebooks app test lint

install:            ## create the environment and install everything
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pip install -e .
	$(PY) scripts/fix_lightgbm_macos.py

demo:               ## full run on data/raw/train.csv: README numbers, charts, dashboard artefacts, notebooks
	$(PY) -m src.pipeline --all
	$(MAKE) notebooks

notebooks:          ## re-execute the three notebooks on the cached results
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

app:                ## start the dashboard on http://localhost:8501
	.venv/bin/streamlit run app/streamlit_app.py

test:               ## tests (synthetic data, no download needed)
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests app scripts
