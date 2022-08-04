# Classifier Training with Vector Search

In this directory there are several examples on how to fine-tune linear layers for classification using vector search. You can find a more in-depth explanation on [classifier fine-tuning with vector search here](https://www.pinecone.io/learn/classifier-train-vector-search/). The main stream of files are:

* **00-indexer-clip.ipynb** \[[Colab](https://colab.research.google.com/github/pinecone-io/examples/blob/classifier-train-vector-search/learn/classifier-train-vector-search/00-indexer-clip.ipynb)\]: shows how to initialize an embedding model (CLIP) and use it to create a vector index from image files.
* **01-fine-tune-vector-search.ipynb** \[[Colab](https://colab.research.google.com/github/pinecone-io/examples/blob/classifier-train-vector-search/learn/classifier-train-vector-search/01-fine-tune-vector-search.ipynb)\]: demos the fine-tuning process for a linear layer for binary classification, using vector search to enhance the fine-tuning process.
* **02-classifier-test.ipynb** \[[Colab](https://colab.research.google.com/github/pinecone-io/examples/blob/classifier-train-vector-search/learn/classifier-train-vector-search/02-classifier-test.ipynb)\]: here we test the vector search trained classifier.

There are several other notebooks marked with "xx-", these are supporting materials and/or alternative examples of the fine-tuning process for different target domains.

You can find a demo of the fine-tuning process in our [semantic query trainer app](https://huggingface.co/spaces/pinecone/semantic-query-trainer).
