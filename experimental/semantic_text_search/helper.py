import os
import itertools
import re
import getpass

from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import seaborn as sns


ENVIRONMENTAL_VARIABLE_NAME = 'PINECONE_EXAMPLE_API_KEY'
PINECONE_PALETTE = [
    "#1C17FF",
    "#000000",
    "#030080",
    "#25239D",
    "#738FAB",
    "#DFECF9",
    "#F1F5F8",
    "#FFFFFF",
    "#FAFF00",
    "#8CF1FF"
]
pinecone_api_key = None


def set_pinecone_api_key():
    global pinecone_api_key
    api_key_prompt = (
        f'{ENVIRONMENTAL_VARIABLE_NAME} not found in environmental variables list.\n'
        'Get yours at https://app.pinecone.io and enter it here: '
    )
    printmd(f'Extracting API Key from environmental variable `{ENVIRONMENTAL_VARIABLE_NAME}`...')
    pinecone_api_key = os.getenv(ENVIRONMENTAL_VARIABLE_NAME)
    if not pinecone_api_key:
        printmd(api_key_prompt)
        pinecone_api_key = getpass.getpass('')
    printmd('Pinecone API Key available at `h.pinecone_api_key`')


printmd = lambda x: display(Markdown(x))


def chunks(lst, n):
    """A generator function that iterates through lst in batches.
    
    Each batch is of size n except possibly the last batch, which may be of 
    size less than n.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_top_sources(dataframe, n=20):
    """Return an iterable with the top n most frequent domains."""
    sources = dataframe.domain.value_counts().head(n).index.tolist()
    return sources


def get_processed_domain(df_row, sources):
    """Return metadata for top sources."""
    domain = df_row['domain']
    return domain if domain in sources else 'other'


def get_text_prefix(text, num_fragments_to_keep=5):
    """Return an abridged version of text."""
    fragmented_text = re.split(r'(?<=[.:;])\s', text)
    abridged_text = " ".join(fragmented_text[:num_fragments_to_keep])
    return abridged_text


def get_processed_df(df):
    """Return processed dataframe ready for usage."""
    # keep first few sentences
    df['text_to_encode'] = df.title + ' ' + df.text.apply(get_text_prefix)
    # parse date
    df.date = pd.to_datetime(df.date)
    df['year'] = df.date.dt.strftime('%Y').fillna(-1).astype(int)
    df['month'] = df.date.dt.strftime('%m').fillna(-1).astype(int)
    # Process domain (keeping top 20 and labeling the rest as 'other')
    sources = get_top_sources(df)
    df['processed_domain'] = df.apply(
        get_processed_domain, 
        axis=1, 
        args=(sources,)
    )
    # prepare index as string
    df.index = df.index.map(str)
    df.index.name = 'vector_id'
    return df


def get_tqdm_kwargs(dataframe, chunksize):
    return dict(
        smoothing=0, 
        unit='chunk of vectors', 
        total=int(np.ceil(len(dataframe)/chunksize))
    )


def get_ids_scores(response):
    """Return ids and scores from Pinecone query response."""
    matches = response['matches']
    ids, scores = zip(*[(match['id'], match['score']) for match in matches])
    return list(ids), list(scores)


def make_clickable(val):
    # target _blank to open new window
    return f'<a target="_blank" href="{val}">link</a>'


def run_on_module_import():
    sns.set_palette(PINECONE_PALETTE)
    set_pinecone_api_key()
    pd.set_option('display.max_colwidth', 2000)


run_on_module_import()

