![Video search image](https://github.com/pinecone-io/examples/blob/master/search/semantic-search/yt-search/assets/youtube-search-0.png)

In this directory there are notebooks and scripts that demo how we built the [YouTube Search App](https://share.streamlit.io/pinecone-io/playground/yt-search/src/server.py).

* **00-data-build.ipynb** shows how we used the Kaggle YTTTS dataset with Beautiful Soup and other libraries to create the data used in the app. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/search/semantic-search/yt-search/00-data-build.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/pinecone-io/examples/blob/master/search/semantic-search/yt-search/00-data-build.ipynb)

* **01-yt-search.ipynb** demonstrates the indexing and querying used to populate and query the index. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/search/semantic-search/yt-search/01-yt-search.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/pinecone-io/examples/blob/master/search/semantic-search/yt-search/01-yt-search.ipynb)

* **app.py** is the Streamlit script powering the app itself.

You can read more about this project in our article [Making YouTube Search Better with NLP](https://pinecone.io/learn/youtube-search).
