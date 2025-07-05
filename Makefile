.PHONY: setup

setup:
	@echo "Installing dependencies with uv..."
	@uv sync
	@echo "Project setup complete."
