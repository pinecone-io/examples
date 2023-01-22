import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

@st.experimental_singleton
def init_pinecone():
    # find API key at app.pinecone.io
    pinecone.init(api_key="<<YOUR_API_KEY>>", environment="us-west1-gcp")
    return pinecone.Index('gif-search')
    
@st.experimental_singleton
def init_retriever():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

index = init_pinecone()
retriever = init_retriever()


def card(urls):
    figures = [f"""
        <figure style="margin-top: 5px; margin-bottom: 5px; !important;">
            <img src="{url}" style="width: 130px; height: 100px; padding-left: 5px; padding-right: 5px" >
        </figure>
    """ for url in urls]
    return st.markdown(f"""
        <div style="display: flex; flex-flow: row wrap; text-align: center; justify-content: center;">
        {''.join(figures)}
        </div>
    """, unsafe_allow_html=True)

 
st.write("""
## ⚡️ AI-Powered GIF Search ⚡️
""")

query = st.text_input("What are you looking for?", "")

if query != "":
    with st.spinner(text="Similarity Searching..."):
        xq = retriever.encode([query]).tolist()
        xc = index.query(xq, top_k=30, include_metadata=True)
        
        urls = []
        for context in xc['results'][0]['matches']:
            urls.append(context['metadata']['url'])

    with st.spinner(text="Fetching GIFs 🚀🚀🚀"):
        card(urls)
