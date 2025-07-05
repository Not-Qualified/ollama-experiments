# ollama experiments

## Introduction

This repository contains poc code for ollama python sdk, ollama tools, and ollama experiments.

## Implemented Features

- [x] use directory as input context
- [x] use webpage url as input context
- [ ] poc for tools
- [ ] use the lightest model as tool for navigating the actual query to different domain expert model

## Setup the Project and Dependencies

1. Install `uv` (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync project dependencies:
   ```bash
   uv sync
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Run the project:
   ```bash
   python main.py
   ```
   Or run specific experiments from the `experiments/` directory.

