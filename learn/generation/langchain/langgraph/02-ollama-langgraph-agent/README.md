# Ollama LangGraph Agent

## 1. Install Ollama

Go to [ollama.com](https://ollama.com) and install Ollama for your respective OS (we recommend running on MacOS if possible).

## 2. (Optional) Create a New Python Environment

### Via Conda

*Note: you don't need to use `conda`, feel free to use your preferred package manager.*

```
conda create -n ollama-langgraph python=3.12
conda activate ollama-langgraph
```

## 3. Install Required Packages

We first install `poetry` with:

```
pip install -qU poetry
```

Then we install the poetry package by navigating to this directory and running:

```
poetry install
```

## 4. Run the Notebook

Head to `02-ollama-langgraph-agent.ipynb` and continue from there!