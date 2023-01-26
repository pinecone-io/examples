[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/integrations/haystack/haystack_lfqa.ipynb)

# Abstractive QA with LFQA

[*Find the full article version of this notebook here*](https://www.pinecone.io/learn/haystack-lfqa/)

We have seen incredible breakthroughs in Natural Language Processing (NLP) in the last several years. [Question Answering](https://www.pinecone.io/learn/question-answering/) (QA) systems that leverage popular language models such as BERT, ROBERTA, etc., can now easily answer questions from a given context with great precision. These QA systems accept a question, locate the most relevant document passages containing answers from a document store, and extract or generate the most likely answer.

Recent studies have focused on more advanced QA systems such as Long-Form Question Answering (LFQA) systems that can generate multi-sentenced abstractive (generated) answers to open-ended questions. It works by searching massive document stores for documents containing relevant information and then using this information to compose an accurate multi-sentence answer synthetically. The relevant documents give larger context for generating original, abstractive long-form answers.

At present, searching for information on a topic is painstaking. For instance, we might multiple queries on Google or any other search engine to pull snippets of information from several sources. LFQA simplifies this by pulling info from several sources and compressing them into a single, human-like answer.

## Using Haystack for LFQA

We will use [Haystack](https://www.pinecone.io/docs/integrations/haystack/) to build a LFQA system. Three main components are needed to build a LFQA pipeline in Haystack: *DocumentStore, Retriever, and Generator.*

### DocumentStore

As the name suggests, the document store is where all our documents are stored. Haystack has different document stores we can use for various use cases. For LFQA, we will use a dense/embedding-based retriever, but it is possible to use traditional methods such as TF–IDF and BM25. So, we need a vector-optimized document store to hold embedding vectors that represent our documents. We will use the *PineconeDocumentStore*, now available in Haystack starting from [version 1.3.0](https://github.com/deepset-ai/haystack/releases/tag/v1.3.0). We could have easily used any other vector-optimized document store such as *FAISSDocumentStore*.

### Retriever

The QA system needs information relevant to the query to generate an answer to the question. So, we need to retrieve the documents containing relevant information from the document store. The retriever’s job is to find the best candidates by computing the similarity between the question and the document embeddings. The final answer is generated based on the best candidates.

We will use Haystack’s *EmbeddingRetriever* in our LFQA pipeline. It works by first generating the query embedding using a language model and then computing the dot product or cosine similarity between the document embeddings in the document store. Then, the top-k most relevant documents are retrieved. We will use a SentenceTransformer model fine-tuned for the query/document matching task as the retriever.


### Generator

We will use ELI5 BART for the generator - a sequence-to-sequence model trained using the ‘Explain Like I’m 5’ (ELI5) dataset. Sequence-to-Sequence models can take a text sequence as input and produce a different text sequence as the output.

The input to the ELI5 BART model is a single string which is a concatenation of the query and the relevant documents providing the context for the answer. The documents are separated by a special token &lt;P>, so the input string will look as follows:

>question: What is a sonic boom? context: &lt;P> A sonic boom is a sound associated with shock waves created when an object travels through the air faster than the speed of sound. &lt;P> Sonic booms generate enormous amounts of sound energy, sounding similar to an explosion or a thunderclap to the human ear. &lt;P> Sonic booms due to large supersonic aircraft can be particularly loud and startling, tend to awaken people, and may cause minor damage to some structures. This led to prohibition of routine supersonic flight overland.

We will use Haystack’s *Seq2SeqGenerator* - a generic sequence-to-sequence generator based on HuggingFace's transformers library, to initialize the BART model. When using *Seq2SeqGenerator* the concatenation process above is automatically handled by the haystack and transformers library. The generator will compose a paragraph-long answer based on the relevant context documents.

More detail on how the ELI5 dataset was built is available [here](https://arxiv.org/abs/1907.09190) and how ELI5 BART model was trained is available [here](https://yjernite.github.io/lfqa.html).


Now let's build our LFQA system using Haystack.

## Preparing the Environment


```python
# Make sure you have a GPU running to speed up things.
!nvidia-smi
```

    Mon Oct 17 22:45:38 2022       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   55C    P8    10W /  70W |      0MiB / 15109MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


Install the required libraries and their dependencies


```python
!pip install -U pinecone-client
!pip install -U 'farm-haystack[pinecone]'>=1.8.0
!pip install datasets
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pinecone-client in /usr/local/lib/python3.7/dist-packages (2.0.13)
    Requirement already satisfied: loguru>=0.5.0 in /usr/local/lib/python3.7/dist-packages (from pinecone-client) (0.6.0)
    Requirement already satisfied: python-dateutil>=2.5.3 in /usr/local/lib/python3.7/dist-packages (from pinecone-client) (2.8.2)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.7/dist-packages (from pinecone-client) (2.23.0)
    Requirement already satisfied: dnspython>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from pinecone-client) (2.2.1)
    Requirement already satisfied: pyyaml>=5.4 in /usr/local/lib/python3.7/dist-packages (from pinecone-client) (6.0)
    Requirement already satisfied: urllib3>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from pinecone-client) (1.25.11)
    Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from pinecone-client) (4.1.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.5.3->pinecone-client) (1.15.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->pinecone-client) (2022.9.24)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->pinecone-client) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->pinecone-client) (2.10)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: datasets in /usr/local/lib/python3.7/dist-packages (2.6.1)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from datasets) (1.21.6)
    Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from datasets) (21.3)
    Requirement already satisfied: responses<0.19 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.18.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (6.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from datasets) (4.13.0)
    Requirement already satisfied: huggingface-hub<1.0.0,>=0.2.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.10.1)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.7/dist-packages (from datasets) (3.0.0)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from datasets) (1.3.5)
    Requirement already satisfied: dill<0.3.6 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.3.5.1)
    Requirement already satisfied: pyarrow>=6.0.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (6.0.1)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (2.23.0)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.7/dist-packages (from datasets) (0.70.13)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (4.64.1)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.7/dist-packages (from datasets) (3.8.3)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (2022.8.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (22.1.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.2.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.8.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (6.0.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (4.1.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (4.0.2)
    Requirement already satisfied: asynctest==0.13.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (0.13.0)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (2.1.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.2.0->datasets) (3.8.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->datasets) (3.0.9)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (1.25.11)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->datasets) (3.9.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2022.4)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->datasets) (1.15.0)


## Initializing the PineconeDocumentStore

We need an API key to use the PineconeDocumentStore in Haystack (you can sign up for free [here](https://app.pinecone.io/) and get an API key).


```python
from haystack.document_stores import PineconeDocumentStore

document_store = PineconeDocumentStore(
    api_key='<<YOUR_API_KEY>>',
    index='haystack-lfqa',
    similarity="cosine",
    embedding_dim=768
)
```


```python
document_store.metric_type
```




    'cosine'




```python
document_store.get_document_count()
```




    0




```python
document_store.get_embedding_count()
```




    0



The above code is all we need to initialize a Pinecone document store with Haystack. It will either create a Pinecone index named ```haystack-lfqa``` if it is not already there or connect to an existing index with the same name. The embedding dimension is set to 768 as the SentenceTransformer model we use to encode queries and documents outputs a vector with 768 dimensions. We also set the similarity metric to cosine as this particular model was trained to be used with cosine similarity.

## Preparing and Indexing Documents

We will use the Wiki Snippets dataset, containing over 17 million passages from Wikipedia, as our source documents. But for this demo, we will use only fifty thousand passages which contains 'History' in the 'section_title' column as indexing the whole dataset will take a lot of time. But feel free to use the entire dataset if you wish. Pinecone vector database can easily handle millions of documents for you. This dataset is available on HuggingFace, so we can use the HuggingFace dataset library to load the dataset quickly and filter the historical passages.


```python
from datasets import load_dataset

wiki_data = load_dataset(
    'vblagoje/wikipedia_snippets_streamed',
    split='train',
    streaming=True
)
wiki_data
```




    <datasets.iterable_dataset.IterableDataset at 0x7f4f8cca3110>



We are loading the dataset in the streaming mode so that we don't have to wait for the whole dataset to download (which is over 9GB). Instead, we access the data when we iterate through the dataset.


```python
# show the contents of a single document in the dataset
next(iter(wiki_data))
```




    {'wiki_id': 'Q7593707',
     'start_paragraph': 2,
     'start_character': 0,
     'end_paragraph': 6,
     'end_character': 511,
     'article_title': "St John the Baptist's Church, Atherton",
     'section_title': 'History',
     'passage_text': "St John the Baptist's Church, Atherton History There have been three chapels or churches on the site of St John the Baptist parish church. The first chapel at Chowbent was built in 1645 by John Atherton as a chapel of ease of Leigh Parish Church. It was sometimes referred to as the Old Bent Chapel. It was not consecrated and used by the Presbyterians as well as the Vicar of Leigh. In 1721 Lord of the manor Richard Atherton expelled the dissenters who subsequently built Chowbent Chapel. The first chapel was consecrated in 1723 by the Bishop of Sodor and"}




```python
# Filter only documents with History as section_title
history = wiki_data.filter(lambda d: d['section_title'].startswith('History'))
history
```




    <datasets.iterable_dataset.IterableDataset at 0x7f4e804b87d0>



Now the dataset is ready, we need to initialize the second component in our LFQA system - the Retriever.

## Initializing the Retriever

We will use Haystack's *EmbeddingRetriever* with a SentenceTransformer model trained based on Microsoft's MPNet. This model performs quite well for comparing the similarity between queries and documents. We can use the retriever to easily compute and update the embeddings for all the documents in the document store.


```python
import torch
# confirm GPU is available (if using CPU this step will be slower)
torch.cuda.is_available()
```




    True



It will take some time to compute all the embeddings and update the index. If you have access to a GPU, it will significantly speed up the process.


```python
from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers"
)
```

To index the documents, we first create Haystack Document objects containing the content and metadata for each document. We are iterating through the filtered dataset and adding the documents to the document store when `256` Document objects and embeddings are created.


```python
from haystack import Document
from tqdm.auto import tqdm  # progress bar

total_doc_count = 50000
batch_size = 256

counter = 0
docs = []
for d in tqdm(history, total=total_doc_count):
    # create haystack document object with text content and doc metadata
    doc = Document(
        content=d["passage_text"],
        meta={
            "article_title": d["article_title"],
            'section_title': d['section_title']
        }
    )
    docs.append(doc)
    counter += 1
    if counter % batch_size == 0:
        # writing docs everytime `batch_size` docs are reached
        embeds = retriever.embed_documents(docs)
        for i, doc in enumerate(docs):
            doc.embedding = embeds[i]
        document_store.write_documents(docs)
        docs.clear()
    if counter == total_doc_count:
        break
```


      0%|          | 0/50000 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



    Batches:   0%|          | 0/8 [00:00<?, ?it/s]



    Writing Documents:   0%|          | 0/256 [00:00<?, ?it/s]



```python
document_store.get_embedding_count()
```




    49915



The embeddings are updated to the document store, and our retriever is now ready. Let's test the retriever before we use it in the LFQA pipeline. We can test queries with the retriever by loading it into a *DocumentSearchPipeline*. Keep in mind we are only using fifty thousand passages from the Wiki Snippets dataset, so it is likely that documents containing relevant information for our exact queries are not in the document store. You can always test whether relevant documents are returned by running some queries on *DocumentSearchPipeline*.


```python
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

search_pipe = DocumentSearchPipeline(retriever)
result = search_pipe.run(
    query="When was the first electric power system built?",
    params={"Retriever": {"top_k": 2}}
)

print_documents(result)
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: When was the first electric power system built?
    
    {   'content': 'Electric power system History In 1881, two electricians built '
                   "the world's first power system at Godalming in England. It was "
                   'powered by two waterwheels and produced an alternating current '
                   'that in turn supplied seven Siemens arc lamps at 250 volts and '
                   '34 incandescent lamps at 40 volts. However, supply to the '
                   'lamps was intermittent and in 1882 Thomas Edison and his '
                   'company, The Edison Electric Light Company, developed the '
                   'first steam-powered electric power station on Pearl Street in '
                   'New York City. The Pearl Street Station initially powered '
                   'around 3,000 lamps for 59 customers. The power station '
                   'generated direct current and',
        'name': None}
    
    {   'content': 'by a coal burning steam engine, and it started generating '
                   'electricity on September 4, 1882, serving an initial load of '
                   '400 incandescent lamps used by 85 customers located within '
                   'about 2 miles (3.2\xa0km) of the station. \n'
                   'However, with the advent of AC, there came the use of '
                   'transformers to convert the generated power to a much higher '
                   'voltage for transmission allowed the power plants and users to '
                   'be separated by hundreds of miles if needed. The high voltage '
                   'could then use transformers to obtain lower voltages for final '
                   'use. Single point failures were minimized in the plant design. '
                   'The AC',
        'name': None}
    


As you can see, the retriever can find relevant documents from the document store. Now let's initialize the third compontent in our LFQA system - the Generator.

## Initializing the Generator

For the generator we will load Haystack’s generic *Seq2SeqGenerator* with a model trained specifically for LFQA. We could use bart_lfqa by [vblagoje](https://huggingface.co/vblagoje) or bart_eli5 by [yjernite](https://huggingface.co/yjernite), both models performs quite well. bart_lfqa was trained with a newer ELI5 dataset, so we will go with that. For more details about the new ELI5 dataset, please refer to this [article](https://towardsdatascience.com/long-form-qa-beyond-eli5-an-updated-dataset-and-approach-319cb841aabb).


```python
from haystack.nodes import Seq2SeqGenerator

generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
```

## Initializing a Generative QA Pipeline

Finally, we need to add the retriever and generator to Haystack's *GenerativeQAPipeline*, a ready-made pipeline for generative QA task. The *GenerativeQAPipeline*, as you might expect, combines the retriever with the generator to produce answers to our questions, and it is the primary interface for communicating with our LFQA system.


```python
from haystack.pipelines import GenerativeQAPipeline

pipe = GenerativeQAPipeline(generator, retriever)
```

## Asking Questions

Now let's run some queries in our LFQA system. When queying we can specificy the number of documents we want the retriever to retrieve and the number of answers the generator to produce. The final answers will be generated based on the documents retrieved from the document store.

The code below is what you need to run queries in the LFQA system.


```python
result = pipe.run(
        query="what was the war of currents?",
        params={
            "Retriever": {"top_k": 3},
            "Generator": {"top_k": 1}
        })

result
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]





    {'query': 'what was the war of currents?',
     'answers': [<Answer {'answer': "The War of Currents was the rivalry between Thomas Edison and George Westinghouse's companies over which form of transmission (direct or alternating current) was superior.", 'type': 'generative', 'score': None, 'context': None, 'offsets_in_document': None, 'offsets_in_context': None, 'document_id': None, 'meta': {'doc_ids': ['3a43249b33b1435e94ef9b22f01989b6', '54f12cf9010626d589b22712a1547983', 'ef560b97430a71ba9ff193062d6a9d0b'], 'doc_scores': [0.711440012, 0.6970715000000001, 0.6848947255], 'content': ['consultant at the Westinghouse Electric & Manufacturing Company\'s Pittsburgh labs.\nBy 1888, the electric power industry was flourishing, and power companies had built thousands of power systems (both direct and alternating current) in the United States and Europe. These networks were effectively dedicated to providing electric lighting. During this time the rivalry between Thomas Edison and George Westinghouse\'s companies had grown into a propaganda campaign over which form of transmission (direct or alternating current) was superior, a series of events known as the "War of Currents". In 1891, Westinghouse installed the first major power system that was designed to drive a', 'of the British administration a favorite route for the smuggling of slaves.', 'of migration began, this state of affairs sometimes led to international incidents, with countries of origin refusing to recognize the new nationalities of natives who had migrated, and when possible, conscripting natives who had naturalized as citizens of another country into military service. The most notable example was the War of 1812, triggered by British impressment of American seamen who were alleged to be British subjects into naval service.\nIn the aftermath of the 1867 Fenian Rising, Irish-Americans who had gone to Ireland to participate in the uprising and were caught were charged with treason, as the British authorities considered them'], 'titles': ['', '', '']}}>],
     'documents': [<Document: {'content': 'consultant at the Westinghouse Electric & Manufacturing Company\'s Pittsburgh labs.\nBy 1888, the electric power industry was flourishing, and power companies had built thousands of power systems (both direct and alternating current) in the United States and Europe. These networks were effectively dedicated to providing electric lighting. During this time the rivalry between Thomas Edison and George Westinghouse\'s companies had grown into a propaganda campaign over which form of transmission (direct or alternating current) was superior, a series of events known as the "War of Currents". In 1891, Westinghouse installed the first major power system that was designed to drive a', 'content_type': 'text', 'score': 0.711440012, 'meta': {'article_title': 'Electric power system', 'section_title': 'History'}, 'embedding': None, 'id': '3a43249b33b1435e94ef9b22f01989b6'}>,
      <Document: {'content': 'of the British administration a favorite route for the smuggling of slaves.', 'content_type': 'text', 'score': 0.6970715000000001, 'meta': {'article_title': 'Muri, Nigeria', 'section_title': 'History'}, 'embedding': None, 'id': '54f12cf9010626d589b22712a1547983'}>,
      <Document: {'content': 'of migration began, this state of affairs sometimes led to international incidents, with countries of origin refusing to recognize the new nationalities of natives who had migrated, and when possible, conscripting natives who had naturalized as citizens of another country into military service. The most notable example was the War of 1812, triggered by British impressment of American seamen who were alleged to be British subjects into naval service.\nIn the aftermath of the 1867 Fenian Rising, Irish-Americans who had gone to Ireland to participate in the uprising and were caught were charged with treason, as the British authorities considered them', 'content_type': 'text', 'score': 0.6848947255, 'meta': {'article_title': 'Multiple citizenship', 'section_title': 'History'}, 'embedding': None, 'id': 'ef560b97430a71ba9ff193062d6a9d0b'}>],
     'root_node': 'Query',
     'params': {'Retriever': {'top_k': 3}, 'Generator': {'top_k': 1}},
     'node_id': 'Generator'}



We can clean up the output using Haystack's `print_answers` util.


```python
from haystack.utils import print_answers

result = pipe.run(
        query="what was the war of currents?",
        params={
            "Retriever": {"top_k": 3},
            "Generator": {"top_k": 1}
        })

print_answers(result, details="minimum")
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: what was the war of currents?
    Answers:
    [   {   'answer': 'The War of Currents was the rivalry between Thomas Edison '
                      "and George Westinghouse's companies over which form of "
                      'transmission (direct or alternating current) was superior.'}]


The answer here is good although there is not too much detail. When we find an answer is either not good or lacking detail there can be two combining factors for this:

* The generator model has not been trained on data that includes information about the *"war on currents"* and so it has not *memorized* this information within it's model weights.

* We have not returned any contexts that contain the answer, so the generator has no reliable external sources of information.

If neither of these conditions are satisfied, the generator cannot produce a factually correct answer. However, in our case we are returning some good external context. We can try and return more detail by increasing the number of contexts retrieved.


```python
result = pipe.run(
        query="what was the war of currents?",
        params={
            "Retriever": {"top_k": 10},
            "Generator": {"top_k": 1}
        })

print_answers(result, details="minimum")
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: what was the war of currents?
    Answers:
    [   {   'answer': 'The War of Currents was the rivalry between Thomas Edison '
                      "and George Westinghouse's companies over which form of "
                      'transmission (direct or alternating current) was superior.'}]


Now we're seeing much more info. Some of it rambles but for the most part it is relevant. We can also compare these results to generator created answer *without* any context by querying the generator directly.


```python
result = generator.predict(
    query="what was the war of currents?",
    documents=[Document(content="")],
    top_k=1
)

print_answers(result, details="minimum")
```

    
    Query: what was the war of currents?
    Answers:
    [{'answer': 'I\'m not sure what you mean by "war".'}]


Clearly, the retrieved contexts are important. Although this isn't always the case, for example if we ask a more well-known question...


```python
result = generator.predict(
    query="who was the first person on the moon?",
    documents=[Document(content="")],
    top_k=1
)

print_answers(result, details="minimum")
```

    
    Query: who was the first person on the moon?
    Answers:
    [{'answer': 'The first man to walk on the moon was Neil Armstrong.'}]


For this type of general knowledge, the generator model is able to pull the answer directly from it's own *"memory"*, eg the model weights optimized during training, where it will have been given training data containing this information. Larger models have a larger memory, but when asking more specific questions (like our question about the war on currents) we rarely return good answers without an external data source.

Let's try some more questions.


```python
result = pipe.run(
        query="when was the first electric power system built?",
        params={
            "Retriever": {"top_k": 3},
            "Generator": {"top_k": 1}
        })

print_answers(result, details="minimum")
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: when was the first electric power system built?
    Answers:
    [   {   'answer': 'The first electric power system was built in 1881 at '
                      'Godalming in England. It was powered by two waterwheels and '
                      'produced an alternating current that in turn supplied seven '
                      'Siemens arc lamps at 250 volts and 34 incandescent lamps at '
                      '40 volts.'}]


We can confirm the correctness of this answer by checking the contexts that this answer has been built from:


```python
for doc in result['documents']:
    print(doc.content, end='\n---\n')
```

    Electric power system History In 1881, two electricians built the world's first power system at Godalming in England. It was powered by two waterwheels and produced an alternating current that in turn supplied seven Siemens arc lamps at 250 volts and 34 incandescent lamps at 40 volts. However, supply to the lamps was intermittent and in 1882 Thomas Edison and his company, The Edison Electric Light Company, developed the first steam-powered electric power station on Pearl Street in New York City. The Pearl Street Station initially powered around 3,000 lamps for 59 customers. The power station generated direct current and
    ---
    by a coal burning steam engine, and it started generating electricity on September 4, 1882, serving an initial load of 400 incandescent lamps used by 85 customers located within about 2 miles (3.2 km) of the station. 
    However, with the advent of AC, there came the use of transformers to convert the generated power to a much higher voltage for transmission allowed the power plants and users to be separated by hundreds of miles if needed. The high voltage could then use transformers to obtain lower voltages for final use. Single point failures were minimized in the plant design. The AC
    ---
    consultant at the Westinghouse Electric & Manufacturing Company's Pittsburgh labs.
    By 1888, the electric power industry was flourishing, and power companies had built thousands of power systems (both direct and alternating current) in the United States and Europe. These networks were effectively dedicated to providing electric lighting. During this time the rivalry between Thomas Edison and George Westinghouse's companies had grown into a propaganda campaign over which form of transmission (direct or alternating current) was superior, a series of events known as the "War of Currents". In 1891, Westinghouse installed the first major power system that was designed to drive a
    ---


In some cases the generator will generate a false answer if it is asked about a topic and does not recieve any relevant contexts.


```python
result = pipe.run(
        query="where did COVID-19 originate?",
        params={
            "Retriever": {"top_k": 3},
            "Generator": {"top_k": 1}
        })

print_answers(result, details="minimum")
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: where did COVID-19 originate?
    Answers:
    [   {   'answer': 'COVID-19 is a zoonotic disease, which means that it is a '
                      'virus that is transmitted from one animal to another. This '
                      'means that there is no way to know for sure where it came '
                      'from.'}]


This is one drawback of the LFQA pipeline, although this can be mitigated to an extent by implementing thresholds on answer confidence scores, and including the sources behind any generated answers. Let's finish with a few final questions.


```python
result = pipe.run(
    query="what was NASAs most expensive project?",
    params={
        "Retriever": {"top_k": 3},
        "Generator": {"top_k": 1}
    }
)

print_answers(result, details="minimum")
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: what was NASAs most expensive project?
    Answers:
    [   {   'answer': 'The Space Shuttle was the most expensive project in the '
                      'history of NASA. It cost over $100 billion to build.'}]



```python
result = pipe.run(
        query="tell me something interesting about the history of Earth?",
        params={
            "Retriever": {"top_k": 3},
            "Generator": {"top_k": 1}
        })

print_answers(result, details="minimum")
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: tell me something interesting about the history of Earth?
    Answers:
    [   {   'answer': "I'm not sure if this is what you're looking for, but I've "
                      "always been fascinated by the fact that the Earth's "
                      'magnetic field is so weak compared to the rest of the solar '
                      'system. The magnetic field of the Earth is about 1/10th the '
                      'strength of that of the strongest magnetic field in the '
                      'Solar System.'}]



```python
result = pipe.run(
        query="who created the Nobel prize and why?",
        params={
            "Retriever": {"top_k": 10},
            "Generator": {"top_k": 1}
        })

print_answers(result, details="minimum")
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: who created the Nobel prize and why?
    Answers:
    [   {   'answer': 'The Nobel Prize was created by Alfred Nobel in his will in '
                      '1896. The idea was that he would use his fortune to create '
                      'a series of prizes for those who confer the "greatest '
                      'benefit on mankind" in physics, chemistry, physiology or '
                      'medicine, literature, and peace.'}]



```python
result = pipe.run(
        query="how is the nobel prize funded?",
        params={
            "Retriever": {"top_k": 10},
            "Generator": {"top_k": 1}
        })

print_answers(result, details="minimum")
```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Query: how is the nobel prize funded?
    Answers:
    [   {   'answer': 'The Nobel Prizes are awarded by the Swedish Academy of '
                      'Sciences and the Norwegian Nobel Committee. The Swedish '
                      'Academy is made up of members of the Royal Swedish Academy, '
                      'the Norwegian Academy, and the American Academy of Arts and '
                      'Sciences. The Nobel Foundation is a non-profit organization '
                      "that is funded by Alfred Nobel's personal fortune."}]


---
