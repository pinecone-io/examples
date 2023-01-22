import os
import getpass
import itertools 

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
JEOPARDY_STANDARD_AMOUNTS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
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


def is_index_nonempty(index):
    """Return a boolean that corresponds with whether the index has 
    been used.
    """
    return bool(index.describe_index_stats()['namespaces'])


def get_processed_df(df):
    """Return processed dataframe ready for usage."""
    # rename columns to conventional lowercase naming with no space
    df = df.rename(columns={'value': 'amount'})
    # remove the rows with no amount or nonstandard amount 
    df = df.drop(df.index[~df.amount.isin(JEOPARDY_STANDARD_AMOUNTS)])
    # parse air date
    df.air_date = df.air_date.apply(pd.to_datetime)
    df['year'] = df.air_date.dt.strftime('%Y')
    df['month'] = df.air_date.dt.strftime('%m')
    # prepare text to encode
    df['text_to_encode'] = df.question + ' ' + df.answer
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


def get_ids(response):
    """Return ids from results."""
    matches = response['matches']
    return [match['id'] for match in matches]


def get_query_results_from_api_response(dataframe, response, query):
    """Return pandas.DataFrame containing query_results with original text."""
    response_df = dataframe.loc[get_ids(response), ['question', 'answer', 'amount']]
    response_df['query'] = query
    return response_df


def get_query_results_from_query(query, pinecone_index, dataframe, model, top_k=1, filter_criteria=None):
    """Return a dataframe of results from a Pinecone query."""
    embedding = model.encode(query).tolist()
    response = pinecone_index.query(
        embedding,
        top_k=top_k,
        filter=filter_criteria,
        include_metadata=True,
    )
    return get_query_results_from_api_response(dataframe, response, query)


def get_jeopardy_questions(queries, pinecone_index, dataframe, model):
    """Return the questions to be used by making API requests to Pinecone."""
    df_list = []
    get_single_result = lambda query, amount: get_query_results_from_query(
        query,
        pinecone_index,
        dataframe,
        model,
        filter_criteria={'amount': {'$eq': amount}}
    )
    for qry, amt in itertools.product(queries, JEOPARDY_STANDARD_AMOUNTS):
        results = get_single_result(qry, amt)
        df_list.append(results)
    return pd.concat(df_list)


def get_jeopardy_boards(jeopardy_questions, queries):
    """Return Jeopardy! and Double Jeopardy! boards according to various topics.
    
    Each category will have representation from each historical difficulty level. 
    
    This capability is only made possible by Pinecone's metadata filtering feature.
    """
    
    # get one result per amount in [$200, $400, .. , $2000]
    
    # wrangle data and display 3-category jeopardy board
    
    jeopardy_board = pd.DataFrame(
        columns=queries, 
        index=JEOPARDY_STANDARD_AMOUNTS)
    grouper = jeopardy_questions.groupby(['query', 'amount'])['question']
    for (query, amount), question_series in grouper:
        jeopardy_board.loc[amount, query] = question_series[0]

    jeopardy_board.index.name = 'amount'
    jeopardy_round_1_filter = jeopardy_board.index <= 1000
    jeopardy_board_first_round = jeopardy_board[jeopardy_round_1_filter]
    jeopardy_board_second_round = jeopardy_board[~jeopardy_round_1_filter]
    
    return jeopardy_board_first_round, jeopardy_board_second_round


def show_answer_widget(jeopardy_questions, queries):
    """Display answer widget in Jupyter Notebook."""
    from ipywidgets import Dropdown, widgets
    from IPython.display import clear_output

    def handle_change(c):
        pass

    def on_button_clicked(b):
        query, amount = dropdown_query.value, dropdown_amount.value
        row_idx_cond = (jeopardy_questions['query'] == query) & (jeopardy_questions['amount'] == amount)
        with output:
            clear_output()
            print(jeopardy_questions.loc[row_idx_cond].iloc[0], flush=True)

    dropdown_query = Dropdown(
        description="query:", 
        options=queries)
    dropdown_amount = Dropdown(
        description="amount:", 
        options=JEOPARDY_STANDARD_AMOUNTS)
    dropdown_query.observe(handle_change, names="value")
    dropdown_amount.observe(handle_change, names="value")
    button = widgets.Button(description='Submit')
    button.on_click(on_button_clicked)
    output = widgets.Output()
    return display(dropdown_query, dropdown_amount, button, output)


def run_on_module_import():
    sns.set_palette(PINECONE_PALETTE)
    set_pinecone_api_key()
    pd.set_option('display.max_colwidth', 2000)


run_on_module_import()

