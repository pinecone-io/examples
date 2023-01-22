# Movie Recommender System

This notebook demonstrates how Pinecone's similarity search as a service helps you build a simple Movie Recommender System. There are three parts to this recommender system:

- A dataset containing movie ratings
- Two deep learning models for embedding movies and users
- A vector index to perform similarity search on those embeddings

The architecture of our recommender system is shown below. We have two models, a user model and a movie model, which generate embedding for users and movies. The two models are trained such that the proximity between a user and a movie in the multi-dimensional vector space depends on the rating given by the user for that movie. This means if a user gives a high rating to a movie, the movie will be closer to the user in the multi-dimensional vector space and vice versa. This ultimately brings users with similar movie preferences and the movies they rated higher closer in the vector space. A similarity search in this vector space for a user would give new recommendations based on the shared movie preference with other users.

< Network Architecture Diagram >

## Install Dependencies


```python
!pip install datasets transformers pinecone-client tensorflow
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting datasets
      Downloading datasets-2.4.0-py3-none-any.whl (365 kB)
    [K     |████████████████████████████████| 365 kB 16.1 MB/s 
    [?25hCollecting transformers
      Downloading transformers-4.21.2-py3-none-any.whl (4.7 MB)
    [K     |████████████████████████████████| 4.7 MB 53.9 MB/s 
    [?25hCollecting pinecone-client
      Downloading pinecone_client-2.0.13-py3-none-any.whl (175 kB)
    [K     |████████████████████████████████| 175 kB 61.7 MB/s 
    [?25hRequirement already satisfied: tensorflow in /usr/local/lib/python3.7/dist-packages (2.8.2+zzzcolab20220719082949)
    Requirement already satisfied: pyarrow>=6.0.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (6.0.1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from datasets) (1.3.5)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (2.23.0)
    Collecting multiprocess
      Downloading multiprocess-0.70.13-py37-none-any.whl (115 kB)
    [K     |████████████████████████████████| 115 kB 59.0 MB/s 
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from datasets) (1.21.6)
    Collecting huggingface-hub<1.0.0,>=0.1.0
      Downloading huggingface_hub-0.9.1-py3-none-any.whl (120 kB)
    [K     |████████████████████████████████| 120 kB 49.5 MB/s 
    [?25hRequirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (4.64.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from datasets) (4.12.0)
    Collecting xxhash
      Downloading xxhash-3.0.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)
    [K     |████████████████████████████████| 212 kB 58.9 MB/s 
    [?25hRequirement already satisfied: aiohttp in /usr/local/lib/python3.7/dist-packages (from datasets) (3.8.1)
    Collecting responses<0.19
      Downloading responses-0.18.0-py3-none-any.whl (38 kB)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (2022.7.1)
    Requirement already satisfied: dill<0.3.6 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.3.5.1)
    Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from datasets) (21.3)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (4.1.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.8.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->datasets) (3.0.9)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2022.6.15)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (3.0.4)
    Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1
      Downloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
    [K     |████████████████████████████████| 127 kB 59.8 MB/s 
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2022.6.2)
    Collecting tokenizers!=0.11.3,<0.13,>=0.11.1
      Downloading tokenizers-0.12.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.6 MB)
    [K     |████████████████████████████████| 6.6 MB 42.9 MB/s 
    [?25hCollecting loguru>=0.5.0
      Downloading loguru-0.6.0-py3-none-any.whl (58 kB)
    [K     |████████████████████████████████| 58 kB 5.8 MB/s 
    [?25hRequirement already satisfied: python-dateutil>=2.5.3 in /usr/local/lib/python3.7/dist-packages (from pinecone-client) (2.8.2)
    Collecting dnspython>=2.0.0
      Downloading dnspython-2.2.1-py3-none-any.whl (269 kB)
    [K     |████████████████████████████████| 269 kB 50.7 MB/s 
    [?25hRequirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.5.3->pinecone-client) (1.15.0)
    Requirement already satisfied: keras<2.9,>=2.8.0rc0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.8.0)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.17.3)
    Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.14.1)
    Requirement already satisfied: gast>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.5.3)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.3.0)
    Requirement already satisfied: tensorboard<2.9,>=2.8 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.8.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from tensorflow) (57.4.0)
    Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.1.2)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.47.0)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.26.0)
    Requirement already satisfied: tensorflow-estimator<2.9,>=2.8 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.8.0)
    Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.1.0)
    Requirement already satisfied: libclang>=9.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (14.0.6)
    Requirement already satisfied: flatbuffers>=1.12 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.0)
    Requirement already satisfied: absl-py>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.2.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.1.0)
    Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.7/dist-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
    Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow) (1.5.2)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow) (1.8.1)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow) (1.35.0)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow) (0.6.1)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow) (3.4.1)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow) (1.0.1)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow) (0.4.6)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow) (0.2.8)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow) (4.2.4)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow) (1.3.1)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->datasets) (3.8.1)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow) (0.4.8)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow) (3.2.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (6.0.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (22.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (4.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.8.1)
    Requirement already satisfied: asynctest==0.13.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (0.13.0)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (2.1.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.2.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2022.2.1)
    Installing collected packages: urllib3, xxhash, tokenizers, responses, multiprocess, loguru, huggingface-hub, dnspython, transformers, pinecone-client, datasets
      Attempting uninstall: urllib3
        Found existing installation: urllib3 1.24.3
        Uninstalling urllib3-1.24.3:
          Successfully uninstalled urllib3-1.24.3
    Successfully installed datasets-2.4.0 dnspython-2.2.1 huggingface-hub-0.9.1 loguru-0.6.0 multiprocess-0.70.13 pinecone-client-2.0.13 responses-0.18.0 tokenizers-0.12.1 transformers-4.21.2 urllib3-1.25.11 xxhash-3.0.0


## Load the Dataset

We will use a subset of the [MovieLens 25M Dataset]("https://grouplens.org/datasets/movielens/25m/") in this project. This dataset contains ~1M user ratings provided by over 30k unique users for the most recent ~10k movies from the [MovieLens 25M Dataset]("https://grouplens.org/datasets/movielens/25m/"). The subset is available [here]("https://huggingface.co/datasets/pinecone/movielens-recent-ratings") on HuggingFace datasets.


```python
from datasets import load_dataset

# load the dataset into a pandas datafame
movies = load_dataset("pinecone/movielens-recent-ratings", split="train").to_pandas()
```

    Using custom data configuration default
    Reusing dataset movie_lens (/Users/jamesbriggs/.cache/huggingface/datasets/pinecone___movie_lens/default/0.0.0/0b5cf78c3c23d9db1c33d17d7d490a06b45c6d9f00a6691aa005c6fcad1c8b82)



```python
# drop duplicates to return only unique movies
unique_movies = movies.drop_duplicates(subset="imdb_id")
unique_movies.head()
```




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
      <th>imdb_id</th>
      <th>movie_id</th>
      <th>user_id</th>
      <th>rating</th>
      <th>title</th>
      <th>poster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt5027774</td>
      <td>6705</td>
      <td>4556</td>
      <td>4.0</td>
      <td>Three Billboards Outside Ebbing, Missouri (2017)</td>
      <td>https://m.media-amazon.com/images/M/MV5BMjI0OD...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt5463162</td>
      <td>7966</td>
      <td>20798</td>
      <td>3.5</td>
      <td>Deadpool 2 (2018)</td>
      <td>https://m.media-amazon.com/images/M/MV5BMDkzNm...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt4007502</td>
      <td>1614</td>
      <td>26543</td>
      <td>4.5</td>
      <td>Frozen Fever (2015)</td>
      <td>https://m.media-amazon.com/images/M/MV5BMjY3YT...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt4209788</td>
      <td>7022</td>
      <td>4106</td>
      <td>4.0</td>
      <td>Molly's Game (2017)</td>
      <td>https://m.media-amazon.com/images/M/MV5BNTkzMz...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt2948356</td>
      <td>3571</td>
      <td>15259</td>
      <td>4.0</td>
      <td>Zootopia (2016)</td>
      <td>https://m.media-amazon.com/images/M/MV5BOTMyMj...</td>
    </tr>
  </tbody>
</table>
</div>



## Initialize Embedding Models

The `user_model` and `movie_model` are trained using Tensorflow Keras. The `user_model` transforms a given `user_id` into a 32-dimensional embedding in the same vector space as the movies, representing the user’s movie preference. The movie recommendations are then fetched based on proximity to the user’s location in the multi-dimensional space.

Similarly, the `movie_model` transforms a given `movie_id` into a 32-dimensional embedding in the same vector space as other similar movies — making it possible to find movies similar to a given movie.


```python
from huggingface_hub import from_pretrained_keras

# load the user model and movie model from huggingface
user_model = from_pretrained_keras("pinecone/movie-recommender-user-model")
movie_model = from_pretrained_keras("pinecone/movie-recommender-movie-model")
```

    config.json not found in HuggingFace Hub
    WARNING:huggingface_hub.hub_mixin:config.json not found in HuggingFace Hub
    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
    config.json not found in HuggingFace Hub
    WARNING:huggingface_hub.hub_mixin:config.json not found in HuggingFace Hub
    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.


## Create Pinecone Index

To create our vector index, we first need to initialize our connection to Pinecone. For this we need a [free API key](https://app.pinecone.io/), and then we initialize the connection like so:


```python
import pinecone

# connect to pinecone environment
pinecone.init(
    api_key="<<YOUR_API_KEY>>",
    environment="us-west1-gcp"
)
```

Now we create a new index called `"movie-emb"`, what we name this isn't important.


```python
index_name = 'movie-emb'

# check if the movie-emb index exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=32,
        metric="cosine"
    )

# connect to movie-emb index we created
index = pinecone.Index(index_name)
```

## Create Movie Embeddings

We will be creating movie embeddings using the pretrained `movie_model`. All of the movie embeddings will be upserted to the new `"movie-emb"` index in Pinecone.


```python
from tqdm.auto import tqdm

# we will use batches of 64
batch_size = 64

for i in tqdm(range(0, len(unique_movies), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(unique_movies))
    # extract batch
    batch = unique_movies.iloc[i:i_end]
    # generate embeddings for batch
    emb = movie_model.predict(batch['movie_id']).tolist()
    # get metadata
    meta = batch.to_dict(orient='records')
    # create IDs
    ids = batch["imdb_id"].values.tolist()
    # add all to upsert list
    to_upsert = list(zip(ids, emb, meta))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

# check that we have all vectors in index
index.describe_index_stats()
```


      0%|          | 0/161 [00:00<?, ?it/s]





    {'dimension': 32,
     'index_fullness': 0.0,
     'namespaces': {'': {'vector_count': 10269}},
     'total_vector_count': 10269}



## Get Recommendations

We now have movie embeddings stored in Pinecone. To get recommendations we can do two things:

1. Get a user embedding via a user embedding model and our `user_id`s, and retrieve movie embeddings (from Pinecone) that are most similar.
2. Use an existing movie embedding to retrieve other similar movies.

Both of these use the same approach, the only difference is the source of data (user vs. movie) and the embedding model (user vs. movie).

We will start with task **1**.


```python
# we do this to display movie posters in this notebook
from IPython.core.display import HTML
```

We will start by looking at a users top rated movies, we can find this information inside the `movies` dataframe by filtering for movie ratings by a specific user (as per their `user_id`), and ordering these by the rating score.


```python
def top_movies_user_rated(user):
    # get list of movies that the user has rated
    user_movies = movies[movies["user_id"] == user]
    # order by their top rated movies
    top_rated = user_movies.sort_values(by=['rating'], ascending=False)
    # return the top 14 movies
    return top_rated['poster'].tolist()[:14], top_rated['rating'].tolist()[:14]
```

After this, we can define a function called `display_posters` that will take a list of movie posters (like those returned by `top_movies_user_rated`) and display them in the notebook.


```python
def display_posters(posters):
    figures = []
    for poster in posters:
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{poster}" style="width: 120px; height: 150px" >
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')
```

Let's take a look at user `3`s top rated movies:


```python
user = 3
top_rated, scores = top_movies_user_rated(user)
display_posters(top_rated)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMDliOTIzNmUtOTllOC00NDU3LWFiNjYtMGM0NDc1YTMxNjYxXkEyXkFqcGdeQXVyNTM3NzExMDQ@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjQ0MTgyNjAxMV5BMl5BanBnXkFtZTgwNjUzMDkyODE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTM4OGJmNWMtOTM4Ni00NTE3LTg3MDItZmQxYjc4N2JhNmUxXkEyXkFqcGdeQXVyNTgzMDMzMTg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTExMzU0ODcxNDheQTJeQWpwZ15BbWU4MDE1OTI4MzAy._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTc2MTQ3MDA1Nl5BMl5BanBnXkFtZTgwODA3OTI4NjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>





```python
print(scores)
```

    [4.5, 4.0, 4.0, 2.5, 2.5]


User `3` has rated these five movies, with *Big Hero 6*, *Civil War*, and *Avengers* being given good scores. They seem less enthusiastic about more sci-fi films like *Arrival* and *The Martian*.

Now let's see how to make some movie recommendations for this user.

Start by defining the `get_recommendations` function. Given a specific `user_id`, this uses the `user_model` to create a user embedding (`xq`). It then retrieves the most similar movie vectors from Pinecone (`xc`), and extracts the relevant movie posters so we can display them later.


```python
def get_recommendations(user):
    # generate embeddings for the user
    xq = user_model([user]).numpy().tolist()
    # compute cosine similarity between user and movie vectors and return top k movies
    xc = index.query(xq, top_k=14,
                    include_metadata=True)
    result = []
    # iterate through results and extract movie posters
    for match in xc['matches']:
        poster = match['metadata']['poster']
        result.append(poster)
    return result
```

## Recommendations for User


```python
urls = get_recommendations(user)
display_posters(urls)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMDliOTIzNmUtOTllOC00NDU3LWFiNjYtMGM0NDc1YTMxNjYxXkEyXkFqcGdeQXVyNTM3NzExMDQ@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjQ0MTgyNjAxMV5BMl5BanBnXkFtZTgwNjUzMDkyODE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTM4OGJmNWMtOTM4Ni00NTE3LTg3MDItZmQxYjc4N2JhNmUxXkEyXkFqcGdeQXVyNTgzMDMzMTg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMGQzN2Y0NDYtOGNlOS00OTVjLTkzMGUtZjYzNzdlMjQxMzgzXkEyXkFqcGdeQXVyNTY4NTYzMDM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjMxNjY2MDU1OV5BMl5BanBnXkFtZTgwNzY1MTUwNTM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNTk4ODQ1MzgzNl5BMl5BanBnXkFtZTgwMTMyMzM4MTI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjE0ODYxMzI2M15BMl5BanBnXkFtZTgwMDczODA2MDE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNjM0NTc0NzItM2FlYS00YzEwLWE0YmUtNTA2ZWIzODc2OTgxXkEyXkFqcGdeQXVyNTgwNzIyNzg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTU0NDE0Nzk1NF5BMl5BanBnXkFtZTgwMTY1NTAxMzE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMGZlNTY1ZWUtYTMzNC00ZjUyLWE0MjQtMTMxN2E3ODYxMWVmXkEyXkFqcGdeQXVyMDM2NDM2MQ@@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTAwMjU5OTgxNjZeQTJeQWpwZ15BbWU4MDUxNDYxODEx._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjMyNDkzMzI1OF5BMl5BanBnXkFtZTgwODcxODg5MjI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTc5MDE2ODcwNV5BMl5BanBnXkFtZTgwMzI2NzQ2NzM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYjhlNDljNTgtZjc4My00NmZmLTk2YzAtYWE5MDYwYjM4MTkzXkEyXkFqcGdeQXVyODE5NzE3OTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>




That looks good, the top results actually match the users three favorite results. Following this we see a lot of Marvel superhero films, which user `3` is probably going to enjoy judging from their current ratings.

Let's see another user, this time we choose `128`.


```python
user = 128
top_rated, scores = top_movies_user_rated(user)
display_posters(top_rated)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTAwMjU5OTgxNjZeQTJeQWpwZ15BbWU4MDUxNDYxODEx._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOTgwMzFiMWYtZDhlNS00ODNkLWJiODAtZDVhNzgyNzJhYjQ4L2ltYWdlXkEyXkFqcGdeQXVyNzEzOTYxNTQ@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjQ0MTgyNjAxMV5BMl5BanBnXkFtZTgwNjUzMDkyODE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTk0MDQ3MzAzOV5BMl5BanBnXkFtZTgwNzU1NzE3MjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYmI5ZGIxOGMtMjcwMS00Yzk3LWE0YWUtMzc5YTFhNGQ4OWZmXkEyXkFqcGdeQXVyNTIzOTk5ODM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOTc2OTA1MDM4M15BMl5BanBnXkFtZTgwNjczMDk5MjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTQ1MjQwMTE5OF5BMl5BanBnXkFtZTgwNjk3MTcyMDE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOTgxMDQwMDk0OF5BMl5BanBnXkFtZTgwNjU5OTg2NDE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOTAzODEzNDAzMl5BMl5BanBnXkFtZTgwMDU1MTgzNzE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTUxMjQ2NjI4OV5BMl5BanBnXkFtZTgwODc2NjUwNDE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOTMyMjEyNzIzMV5BMl5BanBnXkFtZTgwNzIyNjU0NzE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTM4OGJmNWMtOTM4Ni00NTE3LTg3MDItZmQxYjc4N2JhNmUxXkEyXkFqcGdeQXVyNTgzMDMzMTg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNzQ1MjQzMzM3OF5BMl5BanBnXkFtZTcwMzg3NzQ3OQ@@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTc2MTQ3MDA1Nl5BMl5BanBnXkFtZTgwODA3OTI4NjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>





```python
print(scores)
```

    [4.5, 4.5, 4.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]


Because this user seems to like everything, they also get recommended a mix of different things...


```python
urls = get_recommendations(user)
display_posters(urls)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYjFhOWY0OTgtNDkzMC00YWJkLTk1NGEtYWUxNjhmMmQ5ZjYyXkEyXkFqcGdeQXVyMjMxOTE0ODA@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNjIwYjg1ZTEtOWRjYy00MTY3LWIyYTktMTI1Zjk2YzZkNDhiL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMjExNjgyMTc@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNDEwZjU4ZmYtNjk0Ny00ZjVjLWE4OGUtNWE5NzFhNDI0MjgyXkEyXkFqcGdeQXVyNjU2NTIyOTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTUzMjAxMzg5M15BMl5BanBnXkFtZTgwNjIxNjk5NzE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTQ4NzI2OTg5NV5BMl5BanBnXkFtZTgwNjQ3MDgyMjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNDQ4YTAyNDktMDhhYi00MzgyLWI0ZTktMjNiMGQ4MGU0NDQyXkEyXkFqcGdeQXVyNDY5MTUyNjU@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjk4NGZiMzAtODU1NS00MmQ4LWJiNmQtNWU5ZWU4Y2VmNWI0XkEyXkFqcGdeQXVyODE5NzE3OTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYWE1NWFhZWEtOTYxZS00NTZmLWE5OWItMGQ2MTYzODNiNjQxXkEyXkFqcGdeQXVyNDM1ODc2NzE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BZWVlZmQ5N2EtZTQ2My00ZDUzLThkMmQtMDgyYTgwZWZlMjA0XkEyXkFqcGdeQXVyMjQ5NjMxNDA@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYjkzZWIyZTctN2U3Ny00MDZlLTkzZTYtMTI2MWI5YTFiZWZkXkEyXkFqcGdeQXVyNTM2NTg3Nzg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYmJhZmJlYTItZmZlNy00MGY0LTg0ZGMtNWFkYWU5NTA1YTNhXkEyXkFqcGdeQXVyODE5NzE3OTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMDNjZjkyNjQtNWMyMC00ODA5LTgyODctOGRiOWUwYTAzOWVjXkEyXkFqcGdeQXVyODE5NzE3OTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTQzMjE5NDQwMl5BMl5BanBnXkFtZTgwMjI2NzA2MDE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BY2ViYjZiYTktZmJmMS00MGU5LTkxYjgtZWNkYzIyMGFjNWU4XkEyXkFqcGdeQXVyNTMzOTU3NzA@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>





```python
user = 20000
top_rated, scores = top_movies_user_rated(user)
display_posters(top_rated)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTU2NjA1ODgzMF5BMl5BanBnXkFtZTgwMTM2MTI4MjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BN2U1YzdhYWMtZWUzMi00OWI1LWFkM2ItNWVjM2YxMGQ2MmNhXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMzM5NjUxOTEyMl5BMl5BanBnXkFtZTgwNjEyMDM0MDE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTc2MTQ3MDA1Nl5BMl5BanBnXkFtZTgwODA3OTI4NjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMDliOTIzNmUtOTllOC00NDU3LWFiNjYtMGM0NDc1YTMxNjYxXkEyXkFqcGdeQXVyNTM3NzExMDQ@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOTgwMzFiMWYtZDhlNS00ODNkLWJiODAtZDVhNzgyNzJhYjQ4L2ltYWdlXkEyXkFqcGdeQXVyNzEzOTYxNTQ@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOTMyMjEyNzIzMV5BMl5BanBnXkFtZTgwNzIyNjU0NzE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>





```python
print(scores)
```

    [5.0, 4.0, 3.5, 3.5, 3.5, 3.0, 1.0]


We can see more of a trend towards action films with this user, so we can expect the see similar action focused recommendations.


```python
urls = get_recommendations(user)
display_posters(urls)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjE2NDkxNTY2M15BMl5BanBnXkFtZTgwMDc2NzE0MTI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTU2NjA1ODgzMF5BMl5BanBnXkFtZTgwMTM2MTI4MjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTUyODU0ODU1Ml5BMl5BanBnXkFtZTgwNzM1MjIyMDE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjIzMzA5NDk0NF5BMl5BanBnXkFtZTgwMDY2OTE2OTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTg5MTE2NjA4OV5BMl5BanBnXkFtZTgwMTUyMjczMTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOGQyYTZjMDktMWQwNS00Y2NiLTg5MDctMjE3NjU5MjhmZDdiXkEyXkFqcGdeQXVyNTE0MDY4Mjk@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNDA0MTlkYWQtNzNiMS00ZWE3LTg2ODUtZTQwNmVkN2E3M2NhXkEyXkFqcGdeQXVyNjUzNjY0NTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTk1MjUzNzM0OF5BMl5BanBnXkFtZTgwMTg2MzIxMTI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYzc5MTU4N2EtYTkyMi00NjdhLTg3NWEtMTY4OTEyMzJhZTAzXkEyXkFqcGdeQXVyNjc1NTYyMjg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjMyNDkzMzI1OF5BMl5BanBnXkFtZTgwODcxODg5MjI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BODg2OTVhZGQtYTU3Yi00NDg3LTljNzQtMjZhNDBhZjNlOGEyXkEyXkFqcGdeQXVyNjU1OTg4OTM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOTQyODc5MTAwM15BMl5BanBnXkFtZTgwNjMwMjA1MjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYmY0NGNiNzQtYmQ2Yi00OWEyLThmMWMtZjUzM2UwNDg1YjUxXkEyXkFqcGdeQXVyNTg4MTExMTg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BM2Y3ZGM1OGItMTNjZS00MzI3LThkOGEtMDA2MmFlOTVlMTVhXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>




## Find Similar Movies

Now let's see how to find some similar movies.

Start by defining the `get_similar_movies` function. Given a specific `imdb_id`, we query directly using the pre-existing embedding for that ID stored in Pinecone.


```python
# search for similar movies in pinecone index
def get_similar_movies(imdb_id):
    # compute cosine similarity between movie and embedding vectors and return top k movies
    xc = index.query(id=imdb_id, top_k=14, include_metadata=True)
    result = []
    # iterate through results and extract movie posters
    for match in xc['matches']:
        poster = match['metadata']['poster']
        result.append(poster)
    return result
```


```python
# imdbid of Avengers Infinity War
imdb_id = "tt4154756"
# filter the imdbid from the unique_movies
movie = unique_movies[unique_movies["imdb_id"] == imdb_id]
movie
```





  <div id="df-ab0e361d-f5c4-46bb-a2b9-b79d18186f20">
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
      <th>imdb_id</th>
      <th>movie_id</th>
      <th>user_id</th>
      <th>rating</th>
      <th>title</th>
      <th>poster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>tt4154756</td>
      <td>1263</td>
      <td>153</td>
      <td>4.0</td>
      <td>Avengers: Infinity War - Part I (2018)</td>
      <td>https://m.media-amazon.com/images/M/MV5BMjMxNj...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ab0e361d-f5c4-46bb-a2b9-b79d18186f20')"
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
          document.querySelector('#df-ab0e361d-f5c4-46bb-a2b9-b79d18186f20 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ab0e361d-f5c4-46bb-a2b9-b79d18186f20');
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
# display the poster of the movie
display_posters(movie["poster"])
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjMxNjY2MDU1OV5BMl5BanBnXkFtZTgwNzY1MTUwNTM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>




Now we have *Avengers: Infinity War*. Let's find movies that are similar to this movie.


```python
similar_movies = get_similar_movies(imdb_id)
display_posters(similar_movies)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjMxNjY2MDU1OV5BMl5BanBnXkFtZTgwNzY1MTUwNTM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTc5MDE2ODcwNV5BMl5BanBnXkFtZTgwMzI2NzQ2NzM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjMyNDkzMzI1OF5BMl5BanBnXkFtZTgwODcxODg5MjI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjQ0MTgyNjAxMV5BMl5BanBnXkFtZTgwNjUzMDkyODE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTAwMjU5OTgxNjZeQTJeQWpwZ15BbWU4MDUxNDYxODEx._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNjM0NTc0NzItM2FlYS00YzEwLWE0YmUtNTA2ZWIzODc2OTgxXkEyXkFqcGdeQXVyNTgwNzIyNzg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTM4OGJmNWMtOTM4Ni00NTE3LTg3MDItZmQxYjc4N2JhNmUxXkEyXkFqcGdeQXVyNTgzMDMzMTg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNTk4ODQ1MzgzNl5BMl5BanBnXkFtZTgwMTMyMzM4MTI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYzc5MTU4N2EtYTkyMi00NjdhLTg3NWEtMTY4OTEyMzJhZTAzXkEyXkFqcGdeQXVyNjc1NTYyMjg@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYzFhMGM5ZTMtOGEyMC00ZTY5LWE1ZDUtNjQzM2NjZDdiMjg3XkEyXkFqcGdeQXVyNzIzMzE0NDY@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNmUyMzU3YjgtZTliNS00NWM2LWI5ODgtYWE3ZjAzODgyNjNhXkEyXkFqcGdeQXVyNjY1MTg4Mzc@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjMwNDkxMTgzOF5BMl5BanBnXkFtZTgwNTkwNTQ3NjM@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMDkzNmRhNTMtZDI4NC00Zjg1LTgxM2QtMjYxZDQ3OWJlMDRlXkEyXkFqcGdeQXVyNTU5MjkzMTU@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTc2MTQ3MDA1Nl5BMl5BanBnXkFtZTgwODA3OTI4NjE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>




The top results closely match *Avengers: Infinity War*, the top most similar movie being the movie itself. Following this we see a lot of other Marvel superhero films.


Let's see another movie. This time a cartoon.


```python
# imdbid of Moana
imdb_id = "tt3521164"
# filter the imdbid from the unique_movies
movie = unique_movies[unique_movies["imdb_id"] == imdb_id]
movie
```





  <div id="df-8d5ad3ee-589c-4ab9-8835-80dd6049362e">
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
      <th>imdb_id</th>
      <th>movie_id</th>
      <th>user_id</th>
      <th>rating</th>
      <th>title</th>
      <th>poster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>tt3521164</td>
      <td>5138</td>
      <td>24875</td>
      <td>5.0</td>
      <td>Moana (2016)</td>
      <td>https://m.media-amazon.com/images/M/MV5BMjI4Mz...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8d5ad3ee-589c-4ab9-8835-80dd6049362e')"
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
          document.querySelector('#df-8d5ad3ee-589c-4ab9-8835-80dd6049362e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8d5ad3ee-589c-4ab9-8835-80dd6049362e');
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
# display the poster of the movie
display_posters(movie["poster"])
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjI4MzU5NTExNF5BMl5BanBnXkFtZTgwNzY1MTEwMDI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>





```python
similar_movies = get_similar_movies(imdb_id)
display_posters(similar_movies)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjI4MzU5NTExNF5BMl5BanBnXkFtZTgwNzY1MTEwMDI@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMzJmNGFmYmMtMmZhOC00MGM2LTk5NWItYzMzZmM1MzgzMTgxXkEyXkFqcGdeQXVyNTM3MDMyMDQ@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMTM4ODg0MzM0MV5BMl5BanBnXkFtZTcwNDY2MTc3Nw@@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYjQ5NjM0Y2YtNjZkNC00ZDhkLWJjMWItN2QyNzFkMDE3ZjAxXkEyXkFqcGdeQXVyODIxMzk5NjA@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BZGUxZGMzYTYtNjJlMS00OGQ5LTg5YjItN2JjM2Y2NjQzMzdkL2ltYWdlXkEyXkFqcGdeQXVyNTAyODkwOQ@@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNDMyZDc3YzktNWI3Yy00ZDM1LWJjYTMtY2I2YzRmZGQ5MTU4XkEyXkFqcGdeQXVyMTY5Nzc4MDY@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjA2Mzg2NDMzNl5BMl5BanBnXkFtZTgwMjcwODUzOTE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMjAyODEwOTE4OV5BMl5BanBnXkFtZTgwNDIzMDc3ODE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNDk3MzYyMjU5NF5BMl5BanBnXkFtZTgwNzQ5MDkzMzE@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BZDUyNzhjMTAtNGI5OC00MjYzLWFlNDUtMTQzYTdhZjliMDk0XkEyXkFqcGdeQXVyNTc5OTMwOTQ@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BOGY5ZDA4MDEtNWIzNi00YjkxLWE3Y2EtNmJiNzBhOWEyMWVjXkEyXkFqcGdeQXVyNTE1NjY5Mg@@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BYzMzZWM0NjYtMmNjMi00MzUzLThlNTAtZWQ1NjQzM2QyNzIwXkEyXkFqcGdeQXVyNDQ5MDYzMTk@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BMzg2Mzg4YmUtNDdkNy00NWY1LWE3NmEtZWMwNGNlMzE5YzU3XkEyXkFqcGdeQXVyMjA5MTIzMjQ@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

    <figure style="margin: 5px !important;">
      <img src="https://m.media-amazon.com/images/M/MV5BNDc3YTEzZDItNjE2Yy00Nzg2LTgxMDAtNWMxOTJiMWQxZmNiXkEyXkFqcGdeQXVyMjExNjgyMTc@._V1_SX300.jpg" style="width: 120px; height: 150px" >
    </figure>

</div>




This result quality is good again. The top results returning plenty of cartoons.
