# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "marimo>=0.23.6",
#     "numpy",
#     "pinecone==9.0.1",
#     "tqdm",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Semantic Search

    This notebook demonstrates semantic search using Pinecone and a multilingual translation dataset.

    We'll work with a corpus of English sentences and retrieve results by meaning — not by keyword match.

    Semantic search finds documents similar in meaning to a query, regardless of the exact words used.
    It works well for use cases where intent matters more than vocabulary, such as question answering
    over a document corpus or multilingual search.
    """)
    return


@app.cell
def _():
    import os

    from datasets import load_dataset
    from pinecone import Pinecone

    return Pinecone, load_dataset, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up

    ### Pinecone API Key

    Set your `PINECONE_API_KEY` environment variable before running this notebook.
    You can get a free key at [app.pinecone.io](https://app.pinecone.io).
    """)
    return


@app.cell
def _(Pinecone, os):
    # Initialize client
    api_key = os.environ.get("PINECONE_API_KEY")

    pc = Pinecone(
        api_key=api_key,
        source_tag="pinecone_examples:docs:semantic_search",
    )
    return (pc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating an Index

    Semantic search typically requires three components: records to search over, an embedding model
    to encode meaning, and a vector database to store and query embeddings.

    With Pinecone's Integrated Inference, the index is paired with a hosted embedding model.
    Pinecone handles embedding automatically when you upsert and query records — no separate
    embedding step required.

    We'll use the [llama-text-embed-v2](https://docs.pinecone.io/models/llama-text-embed-v2) model
    and map it to the `chunk_text` field in our records. To embed multilingual content instead,
    swap in the [multilingual-e5-large](https://docs.pinecone.io/models/multilingual-e5-large) model.
    """)
    return


@app.cell
def _(mo, pc):
    index_name = "semantic-search"

    if not pc.indexes.exists(name=index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"},
            },
        )

    index_desc = pc.describe_index(name=index_name)

    index = pc.index(host=index_desc.host)

    mo.inspect(index.describe_index_stats())
    return index, index_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating the Dataset

    We're using a subset of [Tatoeba](https://tatoeba.org/), a multilingual sentence translation
    dataset with hundreds of thousands of pairs. Here are a few records:
    """)
    return


@app.cell
def _(load_dataset):
    tatoeba = load_dataset(
        "Helsinki-NLP/tatoeba",
        lang1="en",
        lang2="es",
        split="train",
    )
    return (tatoeba,)


@app.cell
def _(tatoeba):
    tatoeba[0:3]

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before embedding the full dataset, we can use a keyword filter to pre-select a focused
    subset of sentences. This keeps the demo small and fast.
    """)
    return


@app.cell
def _():
    def simple_keyword_filter(sentence, keywords):
        for keyword in keywords:
            if keyword in sentence:
                return True
        return False


    def prepare_sentences(dataset, keywords=None):
        if keywords:
            dataset = dataset.filter(
                lambda x: simple_keyword_filter(
                    sentence=x["translation"]["en"], keywords=keywords
                )
            )

        dataset = dataset.flatten()
        dataset = dataset.remove_columns("translation.es")
        dataset = dataset.rename_column("translation.en", "sentence")

        # The dataset has some english sentences multiple times
        # with different spanish translations, but we're not interested
        # in that for this demo so we remove the dupes.
        seen = set()
        def is_unique(example):
            if example["sentence"] in seen:
                return False
            seen.add(example["sentence"])
            return True

        dataset = dataset.filter(is_unique)

        return dataset.add_column("lang", ["en"] * len(dataset))

    return (prepare_sentences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The filtered dataset contains mostly what we expect, but also some sentences where "park"
    appears as a substring but not at a word boundary — for example, "A glass of sparkling water, please."

    This isn't a problem. The embedding model encodes meaning, so these sentences will land in
    a different region of the vector space and won't surface as relevant results for queries
    about parks or parking.
    """)
    return


@app.cell
def _(mo, prepare_sentences, tatoeba):
    keywords = ["park"]
    sentences = prepare_sentences(tatoeba, keywords=keywords)

    mo.ui.table(sentences, page_size=10)
    return (sentences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `to_records` converts the prepared sentences into the format Pinecone's `upsert_records`
    expects: a list of dicts, each with a unique `id`, a `chunk_text` field, and any additional
    metadata — here, `lang`.

    The `chunk_text` field name comes from the `field_map` we set when creating the index.
    Pinecone uses that mapping to know which field to embed automatically on upsert.
    """)
    return


@app.function
def to_records(sentences, column):
    return [
        {
            "id": sentence["id"],
            "chunk_text": sentence[column],
            "lang": sentence["lang"],
        }
    for sentence in sentences ]


@app.cell
def _(mo, sentences):
    records = to_records(sentences, column="sentence")

    mo.ui.table(records)
    return (records,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Upserting Records

    Each record flows through the embedding model specified at index creation time, producing a vector.
    We store these vectors in Pinecone alongside any metadata fields.

    We use a namespace (`english-sentences`) to group records.
    [Namespaces](https://docs.pinecone.io/guides/index-data/indexing-overview#namespaces) partition
    an index and scope queries to a subset of records.

    Records also include a `lang` metadata field.
    [Metadata filtering](https://docs.pinecone.io/guides/index-data/indexing-overview#metadata)
    lets you narrow results by field value — useful if you later add sentences in other languages
    to the same index.
    """)
    return


@app.cell
def _(index, mo, records):
    batch_size = 96
    namespace = "english-sentences"

    # Batching avoids hitting the embedding model's rate limit
    for start in mo.status.progress_bar(
        range(0, len(records), batch_size),
        title="Upserting records",
        show_rate=True,
        show_eta=True,
    ):
        index.upsert_records(
            records=records[start : start + batch_size], namespace=namespace
        )
    return (namespace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Querying

    With Integrated Inference, querying works the same way as upserting: pass text directly and
    Pinecone embeds it using the same model. The query vector is compared against all stored vectors
    and the closest matches are returned.

    We'll run two queries using different meanings of "park" and observe how the results differ.
    """)
    return


@app.cell(hide_code=True)
def _(index, namespace):
    def print_results(query, results):
        print(f"Query: '{query}'")
        for hit in results.result.hits:
            print(f"  {hit.fields['chunk_text']} (score: {hit.score:.4f})")
        print()


    def search(query, top_k=10):
        results = index.search(
            namespace=namespace,
            top_k=top_k,
            inputs={"text": query},
        )
        print_results(query, results)

    return (search,)


@app.cell
def _(search):
    search("I want to go to the park and relax")
    return


@app.cell(hide_code=True)
def _(search):
    search("I need a place to park")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How It Works

    When you call `index.search` with a text string, Pinecone first embeds it using the same model
    configured at index creation — in this case, `llama-text-embed-v2`. This produces a query vector
    in the same embedding space as the stored sentence vectors.

    Pinecone then compares the query vector against all stored vectors using cosine similarity: the
    cosine of the angle between two vectors. A score of 1.0 means the vectors point in the same
    direction (identical meaning); a score near 0 means they are unrelated. The `top_k` results
    with the highest scores are returned.

    Because the query string and the stored sentences are encoded by the same model, the proximity
    in vector space reflects proximity in meaning — which is what makes semantic search work.

    At scale, comparing a query vector against every stored vector would be slow. Pinecone uses
    approximate nearest neighbor (ANN) algorithms to find the closest matches in sub-linear time,
    maintaining low latency even across billions of vectors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleanup

    Delete the index when you're done to free up resources.
    """)
    return


@app.cell
def _(mo):
    delete_button = mo.ui.run_button(label="Delete index")
    delete_button
    return (delete_button,)


@app.cell(hide_code=True)
def _(delete_button, index_name, mo, pc):
    mo.stop(not delete_button.value)
    pc.indexes.delete(index_name)
    return


if __name__ == "__main__":
    app.run()
