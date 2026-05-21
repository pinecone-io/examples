# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets==3.5.1",
#     "marimo>=0.23.6",
#     "pinecone==9.0.1",
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

    We'll use the [multilingual-e5-large](https://docs.pinecone.io/models/multilingual-e5-large)
    model, which encodes text from many languages into the same vector space. This means a query
    in English can return results in Spanish (and vice versa) without any translation step.
    """)
    return


@app.cell
def _(pc):
    index_name = "semantic-search"

    if pc.indexes.exists(name=index_name):
        pc.indexes.delete(name=index_name)

    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "multilingual-e5-large",
            "field_map": {"text": "chunk_text"},
        },
    )

    index = pc.index(name=index_name)

    index.describe_index_stats()
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
    Before embedding, we filter the dataset to a focused subset. We filter on the English side
    of each translation pair, then extract sentences for both languages separately — so the
    Spanish sentences are the actual translations of the matched English sentences.
    """)
    return


@app.cell
def _():
    def simple_keyword_filter(sentence, keywords):
        for keyword in keywords:
            if keyword in sentence:
                return True
        return False

    def filter_pairs(dataset, keywords):
        """Filter translation pairs where the English sentence contains any keyword."""
        return dataset.filter(
            lambda x: simple_keyword_filter(x["translation"]["en"], keywords)
        ).flatten()

    def extract_sentences(pairs, lang):
        """Extract and deduplicate sentences for one language from filtered pairs."""
        other = "es" if lang == "en" else "en"
        dataset = pairs.remove_columns(f"translation.{other}")
        dataset = dataset.rename_column(f"translation.{lang}", "sentence")

        seen = set()

        def is_unique(example):
            if example["sentence"] in seen:
                return False
            seen.add(example["sentence"])
            return True

        dataset = dataset.filter(is_unique)
        return dataset.add_column("lang", [lang] * len(dataset))

    return extract_sentences, filter_pairs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The English sentences contain mostly what we expect, but also some where "park" appears as a
    substring — for example, "A glass of sparkling water, please." This isn't a problem: the
    embedding model encodes meaning, so these sentences will land in a different region of the
    vector space and won't surface as relevant results.
    """)
    return


@app.cell
def _(extract_sentences, filter_pairs, mo, tatoeba):
    keywords = ["park"]
    filtered_pairs = filter_pairs(tatoeba, keywords=keywords)
    english = extract_sentences(filtered_pairs, lang="en")
    spanish = extract_sentences(filtered_pairs, lang="es")

    mo.vstack(
        [
            mo.md(f"**English** — {len(english)} sentences"),
            mo.ui.table(english, page_size=5),
            mo.md(f"**Spanish** — {len(spanish)} sentences"),
            mo.ui.table(spanish, page_size=5),
        ]
    )
    return english, spanish


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `to_records` converts the prepared sentences into the format Pinecone's `upsert_records`
    expects: a list of dicts, each with a unique `id`, a `chunk_text` field, and any additional
    metadata — here, `lang`.

    The `chunk_text` field name comes from the `field_map` we set when creating the index.
    Pinecone uses that mapping to know which field to embed automatically on upsert.

    We prefix IDs with the language code (`en-`, `es-`) to avoid collisions when combining
    records from multiple languages.
    """)
    return


@app.function
def to_records(sentences, column, id_prefix=""):
    return [
        {
            "id": f"{id_prefix}{idx}",
            "chunk_text": sentence[column],
            "lang": sentence["lang"],
        }
        for idx, sentence in enumerate(sentences)
    ]


@app.cell
def _(english, mo, spanish):
    records = to_records(english, column="sentence", id_prefix="en-") + to_records(
        spanish, column="sentence", id_prefix="es-"
    )

    mo.ui.table(records, page_size=10)
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
    namespace = "sentences"

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

    Because both English and Spanish sentences share the same vector space, an English query can
    surface Spanish results — and vice versa. The `lang` column in the results shows where each
    match came from.

    We'll run the same query in English and Spanish to show they retrieve semantically similar results.
    """)
    return


@app.cell
def _(index, mo, namespace):
    def print_results(query, results):
        data = [
            {
                "lang": hit.fields.get("lang", ""),
                "sentence": hit.fields["chunk_text"],
                "score": round(hit.score, 4),
            }
            for hit in results.result.hits
        ]
        return mo.vstack(
            [
                mo.md(f"**Query:** {query}"),
                mo.ui.table(data, show_column_summaries=False),
            ]
        )

    def search(query, top_k=10, lang=None):
        results = index.search(
            namespace=namespace,
            top_k=top_k,
            inputs={"text": query},
            filter={"lang": {"$eq": lang}} if lang else None,
        )
        return print_results(query, results)

    return (search,)


@app.cell
def _(search):
    search("I want to go to the park and relax")
    return


@app.cell
def _(search):
    search("Quiero ir al parque a relajarme")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Meaning Over Keywords

    The next query contains no form of the word "park" — yet it will still retrieve sentences
    about parking a car. This is the key distinction between semantic search and keyword search:
    results are **ranked by meaning**, not by word overlap.
    """)
    return


@app.cell
def _(search):
    search("where can I leave my car downtown")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How It Works

    When you call `index.search` with a text string, Pinecone first embeds it using the same model
    configured at index creation — in this case, `multilingual-e5-large`. This produces a query vector
    in the same embedding space as the stored sentence vectors.

    Pinecone then compares the query vector against all stored vectors using cosine similarity: the
    cosine of the angle between two vectors. A score of 1.0 means the vectors point in the same
    direction (identical meaning); a score near 0 means they are unrelated. The `top_k` results
    with the highest scores are returned.

    Because `multilingual-e5-large` encodes text from many languages into the same vector space,
    a query in English can retrieve Spanish results — and vice versa — without any translation step.
    Proximity in vector space reflects proximity in meaning, regardless of which language the text
    is in.

    **Model selection determines what the vector space looks like.** A model trained only on English
    text would not place Spanish and English sentences near each other. A model trained on code would
    cluster programs by functionality rather than natural language meaning. Choosing the right model
    for your data and use case is the most consequential decision in a semantic search system —
    Pinecone's [model catalog](https://docs.pinecone.io/models/overview) lists available options
    with guidance on when to use each.

    At scale, comparing a query vector against every stored vector would be slow. Pinecone uses
    approximate nearest neighbor (ANN) algorithms to find the closest matches in sub-linear time,
    maintaining low latency even across billions of vectors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filtering by Language

    Querying in one language to find semantically similar results in another is a basic form of
    translation — without any explicit translation step. But sometimes you want to search within
    a single language instead.

    The `lang` metadata field on each record lets you scope results using Pinecone's
    [metadata filtering](https://docs.pinecone.io/guides/search/filter-by-metadata).
    The embedding model still encodes the query the same way — the filter simply restricts which
    records are eligible to be returned.
    """)
    return


@app.cell
def _(search):
    search("I am meeting a friend at the park", lang="en")
    return


@app.cell
def _(search):
    search("Quiero reunirme con un amigo en el parque", lang="es")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try It Yourself

    Enter a query and select a language filter. Results update when you finish typing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    query_input = mo.ui.text(
        placeholder="Enter a search query...",
        value="The park is crowded today",
        full_width=True,
    )

    lang_select = mo.ui.radio(
        options={"All languages": None, "English only": "en", "Spanish only": "es"},
        value="All languages",
    )

    mo.vstack([query_input, lang_select])
    return lang_select, query_input


@app.cell(hide_code=True)
def _(lang_select, query_input, search):
    search(query_input.value, lang=lang_select.value)
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


@app.cell
def _(delete_button, index_name, mo, pc):
    mo.stop(not delete_button.value)
    pc.indexes.delete(name=index_name)
    return


if __name__ == "__main__":
    app.run()
