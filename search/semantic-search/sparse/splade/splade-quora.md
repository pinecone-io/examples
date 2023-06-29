[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/pinecone/sparse/splade/splade-quora.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/pinecone-io/examples/blob/master/pinecone/sparse/splade/splade-quora.ipynb)

# Hybrid Search with Splade Sparse Vectors

[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/full-link.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/pinecone/sparse/splade/splade-vector-generation.ipynb)

## Overview

SPLADE is a class of models that produce sparse embeddings. Unlike dense embeddings which can be difficult to interpret sparse embeddings map to tokens for easier interpretability. SPLADE models have been shown to consistently outperform dense models, particularly in out-of-domain settings. 

The following guide will show you how to construct SPLADE embeddings to use with Pinecone's sparse-dense index. See the [companion guide](https://github.com/pinecone-io/examples/blob/master/pinecone/sparse/splade/splade-vector-generation.ipynb) to learn how to generate embeddings


## Prerequisites

We'll install the required libraries: the `pinecone-client` for interacting with Pinecone, the `pinecone-datasets` library that we will use for fast processing of the Quora dataset, and `numpy`.


```python
!pip install --no-color -qU \
          "pinecone-client[grpc]"==2.2.1 \
          pinecone-datasets=='0.3.1-alpha' \
          numpy==1.24.3
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m181.1/181.1 KB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.3/17.3 MB[0m [31m30.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 KB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m11.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m20.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.2/12.2 MB[0m [31m19.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m35.0/35.0 MB[0m [31m8.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m15.4/15.4 MB[0m [31m23.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m218.0/218.0 KB[0m [31m13.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m218.0/218.0 KB[0m [31m5.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m211.7/211.7 KB[0m [31m7.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m66.8/66.8 KB[0m [31m4.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.1/9.1 MB[0m [31m43.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m140.6/140.6 KB[0m [31m6.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m115.6/115.6 KB[0m [31m5.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m115.5/115.5 KB[0m [31m11.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m115.3/115.3 KB[0m [31m10.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m115.1/115.1 KB[0m [31m10.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m114.6/114.6 KB[0m [31m10.8 MB/s[0m eta [36m0:00:00[0m


## Quora Dataset

We'll load the popular Quora dataset with precomputed embeddings. Both dense and sparse embeddings have been precomputed using the following models:

* Dense: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

* Sparse: [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil)


```python
from pinecone_datasets import load_dataset

dataset = load_dataset("quora_all-MiniLM-L6-v2_Splade")

dataset.documents.head()
```





  <div id="df-09270381-3ced-4121-9082-b01837ef1fd5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>values</th>
      <th>sparse_values</th>
      <th>metadata</th>
      <th>blob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[0.4024894, -0.23425448, -0.36006898, 0.044094...</td>
      <td>{'indices': [1012, 2000, 2011, 2017, 2022, 204...</td>
      <td>None</td>
      <td>{'text': ' What is the step by step guide to i...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[0.5111937, -0.1987632, -0.32637578, 0.1264907...</td>
      <td>{'indices': [1000, 1012, 1999, 2000, 2011, 201...</td>
      <td>None</td>
      <td>{'text': ' What is the step by step guide to i...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[-0.2237151, 0.74151665, -0.18739395, 0.233195...</td>
      <td>{'indices': [1005, 1006, 1007, 1010, 1011, 101...</td>
      <td>None</td>
      <td>{'text': ' What is the story of Kohinoor (Koh-...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>[-0.37123987, 0.7097032, -0.06182622, -0.16823...</td>
      <td>{'indices': [1005, 1011, 1012, 1045, 1047, 199...</td>
      <td>None</td>
      <td>{'text': ' What would happen if the Indian gov...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>[-0.16656642, 0.21881323, -0.0023541958, 0.104...</td>
      <td>{'indices': [2006, 2017, 2064, 2076, 2078, 209...</td>
      <td>None</td>
      <td>{'text': ' How can I increase the speed of my ...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-09270381-3ced-4121-9082-b01837ef1fd5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-09270381-3ced-4121-9082-b01837ef1fd5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-09270381-3ced-4121-9082-b01837ef1fd5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




As you can see, this data is already loaded with the sparse and dense representations of each document. To learn about the generation process of this values, see [this walkthrough](https://github.com/pinecone-io/examples/blob/master/pinecone/sparse/splade/splade-vector-generation.ipynb).

## Index Creation

We first need to initialize our connection to Pinecone to create our vector index. For this, we need a [free API key](https://app.pinecone.io/). We initialize the connection like so:



```python
import os
import pinecone

api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
# find environment next to your API key in the Pinecone console
env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"

pinecone.init(api_key=api_key, environment=env)
pinecone.whoami()
```




    WhoAmIResponse(username='load', user_label='label', projectname='load-test')




```python
index_name = "splade-quora"
dimension = 384
```

We create the index like so:


```python
if index_name not in pinecone.list_indexes():
  pinecone.create_index(
      index_name,
      pod_type='s1',
      metric='dotproduct',
      dimension=dimension
  )
```

And we connect to the index like so:


```python
index = pinecone.GRPCIndex(index_name)
```

## Upsert


Now let's upsert vectors to the index. We are using async upload with batching. For more information on performance boosting, see the Pinecone documentation for [Performance Tuning](https://docs.pinecone.io/docs/performance-tuning).


```python
index.upsert_from_dataframe(dataset.documents.drop(columns="blob"))
```


    sending upsert requests:   0%|          | 0/522931 [00:00<?, ?it/s]



    collecting async responses:   0%|          | 0/1046 [00:00<?, ?it/s]





    upserted_count: 522931




```python
index.describe_index_stats()
```




    {'dimension': 384,
     'index_fullness': 0.2,
     'namespaces': {'': {'vector_count': 522931}},
     'total_vector_count': 522931}



## Query

The dataset comes with a set of prewritten queries that can be used. We view them like so:


```python
dataset.queries.head()
```





  <div id="df-9897526c-f8fd-4269-839b-73bcb944b791">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vector</th>
      <th>sparse_vector</th>
      <th>filter</th>
      <th>top_k</th>
      <th>blob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-0.07095234841108322, 0.0012621647911146283, ...</td>
      <td>{'indices': [18989, 23463, 27058, 31925, 38916...</td>
      <td>None</td>
      <td>5</td>
      <td>{'id': '318', 'text': 'How does Quora look to ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.05170859768986702, -0.024982793256640434, -...</td>
      <td>{'indices': [31604, 31925, 36513, 36821, 38049...</td>
      <td>None</td>
      <td>5</td>
      <td>{'id': '378', 'text': 'How do I refuse to chos...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.005764591973274946, 0.004137433134019375, -...</td>
      <td>{'indices': [947, 2793, 15453, 15498, 35356, 4...</td>
      <td>None</td>
      <td>5</td>
      <td>{'id': '379', 'text': 'Did Ben Affleck shine m...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.00809027161449194, -0.009231459349393845, -...</td>
      <td>{'indices': [8642, 19100, 20780, 24734, 26798,...</td>
      <td>None</td>
      <td>5</td>
      <td>{'id': '399', 'text': 'What are the effects of...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.024374842643737793, 0.07713444530963898, 0....</td>
      <td>{'indices': [1657, 13677, 33956, 43002, 57110]...</td>
      <td>None</td>
      <td>5</td>
      <td>{'id': '420', 'text': 'Why creativity is impor...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9897526c-f8fd-4269-839b-73bcb944b791')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9897526c-f8fd-4269-839b-73bcb944b791 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9897526c-f8fd-4269-839b-73bcb944b791');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Here we define a function that merges the query results with the actual texts of the documents and shows them as a dataframe.


```python
import pandas as pd

def merge_with_documents(query_response, documents_df):
    results_df = pd.DataFrame([res.to_dict() for res in query_response["matches"]])
    results_df = results_df.merge(documents_df, on="id", how="inner")
    results_df["text"] = results_df["blob"].apply(lambda b: b["text"])
    return results_df[["text", "score"]].sort_values("score", ascending=False)
```

We can load a sample query like so:


```python
sample_query = dataset.queries.iloc[14226].to_dict()
sample_query["blob"]["text"]
```




    'How can I teach my kids the alphabet?'



Now we find the similarity scores for the top `5` returned items from the index:


```python
query_response = index.query(**sample_query)
merge_with_documents(query_response, dataset.documents)
```





  <div id="df-8ae4fa9f-98f0-495e-a500-e893d16c4763">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>How do I prepare for software interviews?</td>
      <td>5.739118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>How do I prepare for programming interviews?</td>
      <td>5.189726</td>
    </tr>
    <tr>
      <th>2</th>
      <td>How should I prepare for programming interviews?</td>
      <td>5.008953</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How do I prepare for a software engineering j...</td>
      <td>4.843315</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How do I prepare for interviews?</td>
      <td>4.840115</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8ae4fa9f-98f0-495e-a500-e893d16c4763')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8ae4fa9f-98f0-495e-a500-e893d16c4763 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8ae4fa9f-98f0-495e-a500-e893d16c4763');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Because we have both dense and sparse vectors in the index, the `score` above is calculated like so:

`alpha * dense_score + (1 - alpha) * sparse_score`

The `alpha` parameter specifies the weighting of the two scores. In the following code, we explore the impact of various alpha values using a sample query.


```python
from copy import deepcopy
import numpy as np

def hybrid_weight_query(query, alpha):
  query_transformed = deepcopy(query)
  query_transformed["vector"] = list(np.array(query_transformed["vector"]) * alpha)
  query_transformed["sparse_vector"]["values"] = list(np.array(query_transformed["sparse_vector"]["values"]) * (1.0 - alpha))
  return query_transformed
```

### Only Sparse (alpha = 0)


```python
query_response = index.query(**hybrid_weight_query(sample_query, 0.0))
merge_with_documents(query_response, dataset.documents)
```





  <div id="df-529323ff-6694-43a9-8f68-0f5223c5bfa1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is an example of isolationism?</td>
      <td>0.711820</td>
    </tr>
    <tr>
      <th>1</th>
      <td>If one lived in isolation, would they know an...</td>
      <td>0.701062</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Is Donald Trump an isolationist?</td>
      <td>0.699042</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is behavioral isolation?</td>
      <td>0.697808</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What is isolationism? What are some examples?</td>
      <td>0.676771</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-529323ff-6694-43a9-8f68-0f5223c5bfa1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-529323ff-6694-43a9-8f68-0f5223c5bfa1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-529323ff-6694-43a9-8f68-0f5223c5bfa1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Hybrid (0 < alpha < 1)


```python
# alpha=0.25
query_response = index.query(**hybrid_weight_query(sample_query, 0.25))
merge_with_documents(query_response, dataset.documents)
```





  <div id="df-4ea64b2a-e135-411c-aa84-5dcd2655a5f9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is the best way to teach kids alphabets?</td>
      <td>1.354345</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is the best way to teach children the al...</td>
      <td>1.333429</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What should I teach my children after alphabets?</td>
      <td>1.308289</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How can an adult re-learn the alphabet?</td>
      <td>1.118512</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How do I teach a child to read and write?</td>
      <td>1.102335</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4ea64b2a-e135-411c-aa84-5dcd2655a5f9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4ea64b2a-e135-411c-aa84-5dcd2655a5f9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4ea64b2a-e135-411c-aa84-5dcd2655a5f9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# alpha=0.6
query_response = index.query(**hybrid_weight_query(sample_query, 0.6))
merge_with_documents(query_response, dataset.documents)
```





  <div id="df-33730588-85b4-4117-bdea-1151fdfb1c7b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is the best way to teach kids alphabets?</td>
      <td>3.250428</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is the best way to teach children the al...</td>
      <td>3.200230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What should I teach my children after alphabets?</td>
      <td>3.139894</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How can an adult re-learn the alphabet?</td>
      <td>2.684429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How do I teach a child to read and write?</td>
      <td>2.645604</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-33730588-85b4-4117-bdea-1151fdfb1c7b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-33730588-85b4-4117-bdea-1151fdfb1c7b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-33730588-85b4-4117-bdea-1151fdfb1c7b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Only Dense (alpha = 1)


```python
query_response = index.query(**hybrid_weight_query(sample_query, 1.0))
merge_with_documents(query_response, dataset.documents)
```





  <div id="df-12fbe25d-5523-4a59-bf23-9630853781e4">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is the best way to teach kids alphabets?</td>
      <td>5.417379</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is the best way to teach children the al...</td>
      <td>5.333716</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What should I teach my children after alphabets?</td>
      <td>5.233157</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How can an adult re-learn the alphabet?</td>
      <td>4.474048</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How do I teach a child to read and write?</td>
      <td>4.409339</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-12fbe25d-5523-4a59-bf23-9630853781e4')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-12fbe25d-5523-4a59-bf23-9630853781e4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-12fbe25d-5523-4a59-bf23-9630853781e4');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Once we're done, delete the index to save resources:


```python
pinecone.delete_index(index_name)
```

---
