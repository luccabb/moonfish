VENV := .venv
UV := $(shell which uv)

ensure-uv:
ifndef UV
	# install uv: https://docs.astral.sh/uv/getting-started/installation/#installation-methods
	curl -LsSf https://astral.sh/uv/install.sh | sh
else
	@echo "uv already installed at $(UV)"
endif

venv: ensure-uv
	uv venv

activate:
	. $(VENV)/bin/activate || source $(VENV)/bin/activate

install: venv activate
	uv pip install -e .

requirements.txt: pyproject.toml ensure-uv
	uv pip compile pyproject.toml -o requirements.txt

clean:
	rm -rf .venv __pycache__ .mypy_cache dist *.egg-info