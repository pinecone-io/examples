import streamlit as st
from st_card_component import card_component
from haystack.document_stores import PineconeDocumentStore
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

PINECONE_API_KEY = st.secrets["PINECONE_KEY"]
RETRIEVER = 'mpnet'
RETRIEVER_URL = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
READER = 'roberta-distilled'
READER_URL = 'deepset/roberta-base-squad2-distilled'
INDEX = 'haystack-nhs-jul'
DIMS = 768

@st.experimental_singleton(show_spinner=False)
def init_pipeline():
    # initialize the pinecone doc store
    document_store = PineconeDocumentStore(
        api_key=PINECONE_API_KEY,
        index=INDEX,
        embedding_dim=DIMS
    )
    # initialize the retriever
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=RETRIEVER_URL,
        model_format="sentence_transformers"
    )
    # initialize the reader
    reader = FARMReader(
        model_name_or_path=READER_URL,
        context_window_size=500
    )
    pipe = ExtractiveQAPipeline(reader, retriever)
    return pipe

st.markdown("""
<link
  rel="stylesheet"
  href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap"
/>
""", unsafe_allow_html=True)

#model_name = 'mpnet-discourse'

libraries = [
    "Streamlit",
    "HuggingFace",
    "Sentence-Transformers",
    "PyTorch",
    "TensorFlow"
]

with st.spinner("Initializing the QA pipeline..."):
    pipe = init_pipeline()

def main():
    st.write("# NHS Q&A")
    query = st.text_input('Ask questions about health!', "")
    with st.expander("Advanced Options"):
        # top_k slider
        top_k = st.slider(
            "top_k",
            min_value=1,
            max_value=20,
            value=2
        )

    st.sidebar.write(f"""
    ### Pinecone Index Detail

    **Index name**: {INDEX}

    **Vector dimensionality**: {DIMS}

    ---

    ### How it Works

    The NHS search tool allows us to ask questions based on documents scraped from the NHS website.

    Ask questions like **"Who does pre-eclampsia affect?"** and return relevant results!
    
    The interface you see is powered and hosted by 
    [Streamlit](https://streamlit.io), and the Q&A function works using *magic*, eg
    [Haystack](https://github.com/deepset-ai/haystack) and the
    [PineconeDocumentStore](https://www.pinecone.io/docs/integrations/haystack/).
    
    ---

    ### Usage
    
    If you'd like to restrict your search to a specific number of items (`top_k`)
    you can with the *Advanced Options* dropdown.

    See a relevant chunk of text that seems to just miss what you need? No problem, just
    click on the boxed arrow icon on the left of each result card to find the original
    source.
    """)

    if query != "":
        with st.spinner("Querying, please wait..."):
            # make the query
            prediction = pipe.run(
                query=query,
                params={
                    "Retriever": {"top_k": top_k},
                    "Reader": {"top_k": top_k}
                }
            )
        # display each context
        for i, doc in enumerate(prediction['answers']):
            context = doc.context
            span = doc.offsets_in_context[0]
            start, end = span.start, span.end
            answer = doc.context[start:end]
            score = doc.score
            url = doc.meta['url']
            card_component(
                title=answer,
                context=context,
                highlight_start=start,
                highlight_end=end,
                score=round(score, 2),
                url=url,
                key=i
            )