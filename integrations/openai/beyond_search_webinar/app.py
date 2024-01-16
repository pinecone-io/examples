import streamlit as st
from pinecone import Pinecone
import openai
from openai.embeddings_utils import get_embedding
import json

OPENAI_KEY = st.secrets["OPENAI_KEY"]
PINECONE_KEY = st.secrets["PINECONE_KEY"]
INDEX = 'apr-demo'
instructions = {
    "conservative q&a": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    "paragraph about a question":"Write a paragraph, addressing the question, and use the text below to obtain relevant information\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nParagraph long Answer:",
    "bullet points": "Write a bullet point list of possible answers, addressing the question, and use the text below to obtain relevant information\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nBullet point Answer:",
    "summarize problems given a topic": "Write a summary of the problems addressed by the questions below\"\n\n{0}\n\n---\n\n",
    "extract key libraries and tools": "Write a list of libraries and tools present in the context below\"\n\nContext:\n{0}\n\n---\n\n",
    "simple instructions": "{1} given the common questions and answers below \n\n{0}\n\n---\n\n",
    "summarize": "Write an elaborate, paragraph long summary about \"{1}\" given the questions and answers from a public forum on this topic\n\n{0}\n\n---\n\nSummary:",
}

@st.experimental_singleton(show_spinner=False)
def init_openai():
    # initialize connection to OpenAI
    openai.api_key = OPENAI_KEY

@st.experimental_singleton(show_spinner=False)
def init_key_value():
    with open('./beyond_search_openai/src/beyond_search/mapping.json', 'r') as fp:
        mappings = json.load(fp)
    return mappings

@st.experimental_singleton(show_spinner=False)
def init_pinecone(index_name):
    # initialize connection to Pinecone vector DB (app.pinecone.io for API key)
    pinecone.init(
        api_key=PINECONE_KEY,
        environment="YOUR_ENV"  # find next to API key in console
    )
    index = pinecone.Index(index_name)
    stats = index.describe_index_stats()
    dims = stats['dimension']
    count = stats['namespaces']['']['vector_count']
    return index, dims, count

def create_context(question, index, mappings, lib_meta, max_len=3750, size="curie", top_k=5):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(question, engine=f'text-search-{size}-query-001')
    res = index.query(
        vector=q_embed, top_k=top_k,
        include_metadata=True, filter={
            'docs': {'$in': lib_meta}
        })
    

    cur_len = 0
    contexts = []
    sources = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
            sources.append(row['metadata'])
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    return "\n\n###\n\n".join(contexts), sources

def answer_question(
    index,
    mappings,
    fine_tuned_qa_model="text-davinci-002",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    instruction="Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    max_len=3550,
    size="curie",
    top_k=5,
    debug=False,
    max_tokens=400,
    stop_sequence=None,
    domains=["huggingface", "tensorflow", "streamlit", "pytorch"],
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context, sources = create_context(
        question,
        index,
        mappings,
        lib_meta=domains,
        max_len=max_len,
        size=size,
        top_k=top_k
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # fine-tuned models requires model parameter, whereas other models require engine parameter
        model_param = (
            {"model": fine_tuned_qa_model}
            if ":" in fine_tuned_qa_model
            and fine_tuned_qa_model.split(":")[1].startswith("ft")
            else {"engine": fine_tuned_qa_model}
        )
        #print(instruction.format(context, question))
        response = openai.Completion.create(
            prompt=instruction.format(context, question),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            **model_param,
        )
        return response["choices"][0]["text"].strip(), sources
    except Exception as e:
        print(e)
        return ""

def search(index, text_map, query, style, top_k, lib_filters):
    if query != "":
        with st.spinner("Retrieving, please wait..."):
            answer, sources = answer_question(
                index, text_map,
                question=query,
                instruction=instructions[style],
                top_k=top_k
            )
            # lowercase relevant lib filters
            lib_meta = [lib.lower() for lib in lib_filters.keys() if lib_filters[lib]]
            lower_libs = [lib.lower() for lib in libraries]
        # display the answer
        st.write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.write(f"""
                {source['docs']} > {source['category']} > [{source['thread']}]({source['href']})
                """)

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
    "PyTorch",
    "TensorFlow"
]

with st.spinner("Connecting to OpenAI..."):
    retriever = init_openai()

with st.spinner("Connecting to Pinecone..."):
    index, dims, count = init_pinecone(INDEX)
    text_map = init_key_value()

def main():
    st.write("# ML Q&A")
    search = st.container()
    query = search.text_input('Ask a framework-specific question!', "")
    
    with search.expander("Search Options"):
        style = st.radio(label='Style', options=[
            'Paragraph about a question', 'Conservative Q&A',
            'Bullet points', 'Summarize problems given a topic',
            'Extract key libraries and tools', 'Simple instructions',
            'Summarize'
        ])
        # add section for filters
        st.write("""
        #### Metadata Filters

        **Libraries**
        """)
        # create two cols
        cols = st.columns(2)
        # add filtering based on library
        lib_filters = {}
        for lib in libraries:
            i = len(lib_filters.keys()) % 2
            with cols[i]:
                lib_filters[lib] = st.checkbox(lib, value=True)
        st.write("---")
        top_k = st.slider(
            "top_k",
            min_value=1,
            max_value=20,
            value=5
        )

    st.sidebar.write(f"""
    ### Info

    **Pinecone index name**: {INDEX}

    **Pinecone index size**: {count}

    **OpenAI embedding model**: *text-embedding-ada-002*

    **Vector dimensionality**: {dims}

    **OpenAI generation model**: *gpt-3.5-turbo-instruct*

    ---

    ### How it Works

    The Q&A tool takes discussions and docs from some of the best Python ML
    libraries and collates their content into a natural language search and Q&A tool.

    Ask questions like **"How do I use the gradient tape in tensorflow?"** or **"What is the difference
    between Tensorflow and PyTorch?"**, choose a answer style, and return relevant results!
    
    The app is powered using OpenAI's embedding service with Pinecone's vector database. The whole process consists
    of *three* steps:
    
    **1**. Questions are fed into OpenAI's embeddings service to generate a {dims}-dimensional query vector.
    
    **2**. We use Pinecone to identify similar context vectors (previously encoded from Q&A pages).

    **3**. Relevant pages are passed in a new question to OpenAI's generative model, returning our answer.

    **How do I make something like this?**

    It's easy! Learn how to [integrate OpenAI and Pinecone here](https://www.pinecone.io/docs/integrations/openai/)!

    ---

    ### Usage
    
    If you'd like to restrict your search to a specific library (such as PyTorch or
    Streamlit) you can with the *Advanced Options* dropdown. The source of information
    can be switched between official docs and forum discussions too!

    If you'd like OpenAI to consider more or less pages, try changing the `top_k` slider.

    Want to see the original sources that GPT-3 is using to generate the answer? No problem, just click on the **Sources** box.
    """)

    #if style.lower() == 'conservative q&a':
    #    search.info("*Access search options above.*")

    if search.button("Go!") or query != "":
        with st.spinner("Retrieving, please wait..."):
            # lowercase relevant lib filters
            lib_meta = [lib.lower() for lib in lib_filters.keys() if lib_filters[lib]]
            # ask the question
            answer, sources = answer_question(
                index, text_map,
                question=query,
                instruction=instructions[style.lower()],
                top_k=top_k,
                domains=lib_meta
            )
        # display the answer
        st.write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.write(f"""
                {source['docs']} > {source['category']} > [{source['thread']}]({source['href']})
                """)
