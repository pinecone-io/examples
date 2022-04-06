# Generative Pseudo-Labeling

This directory contains the code notebooks explained in the [Generative Pseudo-Labeling (GPL) article](https://www.pinecone.io/learn/gpl/). Notebooks include:

* `00_download_cord_19.ipynb` shows how to download the CORD-19 dataset.
* `01_query_gen.ipynb` demonstrates the synthetic query generation data prep step.
* `02_negative_mining.ipynb` works through the second data prep step of negative mining.
* `03_ce_scoring.ipynb` details the final data prep step of pseudo-labeling.
* `04_finetune.ipynb` shows how to use the data created in the previous notebooks to fine-tune a bi-encoder using Margin MSE loss.

All of this content is part of a course called [NLP for Semantic Search](https://www.pinecone.io/learn/nlp/).