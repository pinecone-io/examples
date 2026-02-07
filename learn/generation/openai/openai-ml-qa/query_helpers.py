"""Helper functions for querying Pinecone and generating answers with OpenAI.

This module provides reusable utilities for:
- Creating embeddings from queries
- Querying Pinecone for relevant contexts
- Building prompts with retrieved contexts
- Generating answers using OpenAI's completion API
"""

from typing import Any


def create_query_embedding(query: str, openai_client: Any, embed_model: str) -> list[float]:
    """Create an embedding vector for a query string.

    Args:
        query: The query text to embed
        openai_client: OpenAI client instance
        embed_model: Name of the embedding model to use (e.g., 'text-embedding-ada-002')

    Returns:
        List of floats representing the query embedding
    """
    res = openai_client.Embedding.create(
        input=[query],
        engine=embed_model
    )
    return res['data'][0]['embedding']


def query_pinecone(index: Any, query_embedding: list[float], top_k: int = 3) -> dict:
    """Query Pinecone index with an embedding vector.

    Args:
        index: Pinecone index instance
        query_embedding: Query vector to search for
        top_k: Number of results to return (default: 3)

    Returns:
        Dictionary containing query results with matches and metadata
    """
    return index.query(vector=query_embedding, top_k=top_k, include_metadata=True)


def extract_contexts(query_results: dict) -> list[str]:
    """Extract context strings from Pinecone query results.

    Args:
        query_results: Results dictionary from Pinecone query

    Returns:
        List of context strings from the matches
    """
    return [match['metadata']['context'] for match in query_results['matches']]


def build_prompt(
    query: str,
    contexts: list[str],
    prompt_type: str = "basic",
    limit: int = 3750
) -> str:
    """Build a prompt for OpenAI completion with retrieved contexts.

    Args:
        query: The user's question
        contexts: List of context strings from vector search
        prompt_type: Type of prompt - "basic" or "exhaustive" (default: "basic")
        limit: Maximum character limit for contexts (default: 3750)

    Returns:
        Formatted prompt string ready for OpenAI completion
    """
    if prompt_type == "basic":
        prompt_start = (
            "Answer the question based on the context below.\n\n"
            "Context:\n"
        )
    else:  # exhaustive
        prompt_start = (
            "Give an exhaustive summary and answer based on the question "
            "using the contexts below.\n\n"
            "Context:\n"
        )

    prompt_end = f"\n\nQuestion: {query}\nAnswer:"

    # Handle empty contexts
    if not contexts:
        return prompt_start + prompt_end

    # Append contexts until hitting limit
    prompt = None
    for i in range(1, len(contexts) + 1):
        joined_contexts = "\n\n---\n\n".join(contexts[:i])
        if len(joined_contexts) >= limit:
            # Use contexts up to i-1 if we exceeded the limit
            if i > 1:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i-1]) +
                    prompt_end
                )
            else:
                # Even first context exceeds limit, use it anyway
                prompt = prompt_start + contexts[0] + prompt_end
            break
        elif i == len(contexts):
            # We've included all contexts without exceeding limit
            prompt = prompt_start + joined_contexts + prompt_end
            break

    return prompt


def generate_answer(
    prompt: str,
    openai_client: Any,
    engine: str = 'text-davinci-003',
    temperature: float = 0,
    max_tokens: int = 400
) -> str:
    """Generate an answer using OpenAI's completion API.

    Args:
        prompt: The prompt to send to OpenAI
        openai_client: OpenAI client instance
        engine: OpenAI model to use (default: 'text-davinci-003')
        temperature: Sampling temperature (default: 0)
        max_tokens: Maximum tokens to generate (default: 400)

    Returns:
        Generated answer string
    """
    res = openai_client.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()


def query_and_answer(
    query: str,
    index: Any,
    openai_client: Any,
    embed_model: str = "text-embedding-ada-002",
    top_k: int = 3,
    prompt_type: str = "basic",
    engine: str = 'text-davinci-003'
) -> tuple[str, list[str]]:
    """Complete pipeline: query Pinecone and generate an answer.

    This is a convenience function that chains together:
    1. Creating query embedding
    2. Querying Pinecone
    3. Building prompt with contexts
    4. Generating answer with OpenAI

    Args:
        query: User's question
        index: Pinecone index instance
        openai_client: OpenAI client instance
        embed_model: Embedding model name (default: 'text-embedding-ada-002')
        top_k: Number of results from Pinecone (default: 3)
        prompt_type: "basic" or "exhaustive" (default: "basic")
        engine: OpenAI completion engine (default: 'text-davinci-003')

    Returns:
        Tuple of (answer, contexts) where answer is the generated text
        and contexts is the list of retrieved context strings
    """
    # Create embedding for query
    query_embedding = create_query_embedding(query, openai_client, embed_model)

    # Query Pinecone
    query_results = query_pinecone(index, query_embedding, top_k=top_k)

    # Extract contexts
    contexts = extract_contexts(query_results)

    # Handle case where Pinecone returns no results
    if not contexts:
        return "I'm sorry, I don't have enough information to answer that query.", []

    # Build prompt
    prompt = build_prompt(query, contexts, prompt_type=prompt_type)

    # Generate answer
    answer = generate_answer(prompt, openai_client, engine=engine)

    return answer, contexts
