VENV := .venv
UV := $(shell which uv)

ensure-uv:
ifndef UV
	# install uv: https://docs.astral.sh/uv/getting-started/installation/#installation-methods
	curl -LsSf https://astral.sh/uv/install.sh | sh
else
	@echo "uv already installed at $(UV)"
endif


requirements.txt: pyproject.toml ensure-uv
	uv pip compile pyproject.toml -o requirements.txt