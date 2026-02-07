"""Unit tests for query_helpers module.

Tests cover:
- Embedding creation
- Pinecone querying
- Context extraction
- Prompt building
- Answer generation
- End-to-end pipeline
"""

from unittest.mock import MagicMock, Mock

import pytest

from query_helpers import (
    build_prompt,
    create_query_embedding,
    extract_contexts,
    generate_answer,
    query_and_answer,
    query_pinecone,
)


class TestCreateQueryEmbedding:
    """Tests for create_query_embedding function."""

    def test_creates_embedding_from_query(self):
        """Test that embedding is created from query string."""
        # Arrange
        mock_openai = MagicMock()
        mock_openai.Embedding.create.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}]
        }
        query = "What is machine learning?"
        embed_model = "text-embedding-ada-002"

        # Act
        result = create_query_embedding(query, mock_openai, embed_model)

        # Assert
        assert result == [0.1, 0.2, 0.3]
        mock_openai.Embedding.create.assert_called_once_with(
            input=[query],
            engine=embed_model
        )

    def test_handles_different_embedding_models(self):
        """Test that different embedding models can be used."""
        # Arrange
        mock_openai = MagicMock()
        mock_openai.Embedding.create.return_value = {
            'data': [{'embedding': [0.5, 0.6]}]
        }
        query = "Test query"
        custom_model = "custom-embedding-model"

        # Act
        result = create_query_embedding(query, mock_openai, custom_model)

        # Assert
        assert result == [0.5, 0.6]
        mock_openai.Embedding.create.assert_called_once_with(
            input=[query],
            engine=custom_model
        )


class TestQueryPinecone:
    """Tests for query_pinecone function."""

    def test_queries_index_with_embedding(self):
        """Test that Pinecone index is queried with embedding vector."""
        # Arrange
        mock_index = MagicMock()
        mock_index.query.return_value = {
            'matches': [
                {'id': '1', 'score': 0.9, 'metadata': {'context': 'Context 1'}}
            ]
        }
        query_embedding = [0.1, 0.2, 0.3]
        top_k = 5

        # Act
        result = query_pinecone(mock_index, query_embedding, top_k=top_k)

        # Assert
        assert 'matches' in result
        assert len(result['matches']) == 1
        mock_index.query.assert_called_once_with(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

    def test_default_top_k_is_3(self):
        """Test that default top_k value is 3."""
        # Arrange
        mock_index = MagicMock()
        mock_index.query.return_value = {'matches': []}
        query_embedding = [0.1, 0.2, 0.3]

        # Act
        query_pinecone(mock_index, query_embedding)

        # Assert
        mock_index.query.assert_called_once_with(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )


class TestExtractContexts:
    """Tests for extract_contexts function."""

    def test_extracts_contexts_from_matches(self):
        """Test that contexts are extracted from query results."""
        # Arrange
        query_results = {
            'matches': [
                {'id': '1', 'metadata': {'context': 'Context 1'}},
                {'id': '2', 'metadata': {'context': 'Context 2'}},
                {'id': '3', 'metadata': {'context': 'Context 3'}},
            ]
        }

        # Act
        result = extract_contexts(query_results)

        # Assert
        assert result == ['Context 1', 'Context 2', 'Context 3']

    def test_handles_empty_matches(self):
        """Test that empty matches return empty list."""
        # Arrange
        query_results = {'matches': []}

        # Act
        result = extract_contexts(query_results)

        # Assert
        assert result == []


class TestBuildPrompt:
    """Tests for build_prompt function."""

    def test_builds_basic_prompt(self):
        """Test basic prompt construction."""
        # Arrange
        query = "What is Python?"
        contexts = ["Python is a programming language."]

        # Act
        result = build_prompt(query, contexts, prompt_type="basic")

        # Assert
        assert "Answer the question based on the context below." in result
        assert "Python is a programming language." in result
        assert "Question: What is Python?" in result
        assert "Answer:" in result

    def test_builds_exhaustive_prompt(self):
        """Test exhaustive prompt construction."""
        # Arrange
        query = "Explain TensorFlow"
        contexts = ["TensorFlow is a ML framework."]

        # Act
        result = build_prompt(query, contexts, prompt_type="exhaustive")

        # Assert
        assert "Give an exhaustive summary" in result
        assert "TensorFlow is a ML framework." in result
        assert "Question: Explain TensorFlow" in result

    def test_respects_character_limit(self):
        """Test that prompt respects character limit."""
        # Arrange
        query = "Test?"
        # Create contexts that will exceed the limit
        long_context = "x" * 4000
        contexts = [long_context, "Short context"]
        limit = 3750

        # Act
        result = build_prompt(query, contexts, limit=limit)

        # Assert
        # Should only include first context even though it exceeds limit
        # (as we can't split a single context)
        assert long_context in result

    def test_includes_multiple_contexts_with_separator(self):
        """Test that multiple contexts are separated correctly."""
        # Arrange
        query = "Test?"
        contexts = ["Context 1", "Context 2", "Context 3"]

        # Act
        result = build_prompt(query, contexts)

        # Assert
        assert "Context 1" in result
        assert "Context 2" in result
        assert "Context 3" in result
        assert "\n\n---\n\n" in result

    def test_handles_empty_contexts(self):
        """Test handling of empty contexts list."""
        # Arrange
        query = "Test?"
        contexts = []

        # Act
        result = build_prompt(query, contexts)

        # Assert
        assert "Question: Test?" in result
        assert "Answer:" in result


class TestGenerateAnswer:
    """Tests for generate_answer function."""

    def test_generates_answer_from_prompt(self):
        """Test that answer is generated from prompt."""
        # Arrange
        mock_openai = MagicMock()
        mock_openai.Completion.create.return_value = {
            'choices': [{'text': '  The answer is 42.  '}]
        }
        prompt = "Question: What is the answer?"
        engine = "text-davinci-003"

        # Act
        result = generate_answer(prompt, mock_openai, engine=engine)

        # Assert
        assert result == "The answer is 42."
        mock_openai.Completion.create.assert_called_once()
        call_args = mock_openai.Completion.create.call_args[1]
        assert call_args['engine'] == engine
        assert call_args['prompt'] == prompt
        assert call_args['temperature'] == 0
        assert call_args['max_tokens'] == 400

    def test_uses_default_parameters(self):
        """Test that default parameters are used correctly."""
        # Arrange
        mock_openai = MagicMock()
        mock_openai.Completion.create.return_value = {
            'choices': [{'text': 'Answer'}]
        }
        prompt = "Test prompt"

        # Act
        generate_answer(prompt, mock_openai)

        # Assert
        call_args = mock_openai.Completion.create.call_args[1]
        assert call_args['engine'] == 'text-davinci-003'
        assert call_args['temperature'] == 0
        assert call_args['max_tokens'] == 400

    def test_accepts_custom_parameters(self):
        """Test that custom parameters are applied."""
        # Arrange
        mock_openai = MagicMock()
        mock_openai.Completion.create.return_value = {
            'choices': [{'text': 'Custom answer'}]
        }
        prompt = "Test"

        # Act
        generate_answer(
            prompt,
            mock_openai,
            engine='gpt-3.5-turbo',
            temperature=0.7,
            max_tokens=500
        )

        # Assert
        call_args = mock_openai.Completion.create.call_args[1]
        assert call_args['engine'] == 'gpt-3.5-turbo'
        assert call_args['temperature'] == 0.7
        assert call_args['max_tokens'] == 500


class TestQueryAndAnswer:
    """Tests for query_and_answer end-to-end function."""

    def test_complete_pipeline(self):
        """Test the complete query and answer pipeline."""
        # Arrange
        mock_index = MagicMock()
        mock_openai = MagicMock()

        # Mock embedding creation
        mock_openai.Embedding.create.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}]
        }

        # Mock Pinecone query
        mock_index.query.return_value = {
            'matches': [
                {'id': '1', 'metadata': {'context': 'PyTorch is a framework.'}}
            ]
        }

        # Mock completion
        mock_openai.Completion.create.return_value = {
            'choices': [{'text': 'PyTorch is used for deep learning.'}]
        }

        query = "What is PyTorch?"

        # Act
        answer, contexts = query_and_answer(query, mock_index, mock_openai)

        # Assert
        assert answer == "PyTorch is used for deep learning."
        assert contexts == ['PyTorch is a framework.']
        mock_openai.Embedding.create.assert_called_once()
        mock_index.query.assert_called_once()
        mock_openai.Completion.create.assert_called_once()

    def test_uses_custom_parameters(self):
        """Test that custom parameters are passed through correctly."""
        # Arrange
        mock_index = MagicMock()
        mock_openai = MagicMock()

        mock_openai.Embedding.create.return_value = {
            'data': [{'embedding': [0.5]}]
        }
        mock_index.query.return_value = {
            'matches': [{'id': '1', 'metadata': {'context': 'Test'}}]
        }
        mock_openai.Completion.create.return_value = {
            'choices': [{'text': 'Answer'}]
        }

        # Act
        query_and_answer(
            "Test?",
            mock_index,
            mock_openai,
            embed_model="custom-model",
            top_k=10,
            prompt_type="exhaustive",
            engine="gpt-4"
        )

        # Assert
        # Verify embedding model
        embed_call = mock_openai.Embedding.create.call_args[1]
        assert embed_call['engine'] == "custom-model"

        # Verify top_k
        query_call = mock_index.query.call_args[1]
        assert query_call['top_k'] == 10

        # Verify completion engine
        completion_call = mock_openai.Completion.create.call_args[1]
        assert completion_call['engine'] == "gpt-4"

    def test_returns_tuple_with_answer_and_contexts(self):
        """Test that function returns a tuple of (answer, contexts)."""
        # Arrange
        mock_index = MagicMock()
        mock_openai = MagicMock()

        mock_openai.Embedding.create.return_value = {
            'data': [{'embedding': [0.1]}]
        }
        mock_index.query.return_value = {
            'matches': [
                {'id': '1', 'metadata': {'context': 'C1'}},
                {'id': '2', 'metadata': {'context': 'C2'}},
            ]
        }
        mock_openai.Completion.create.return_value = {
            'choices': [{'text': 'Final answer'}]
        }

        # Act
        result = query_and_answer("Q?", mock_index, mock_openai)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        answer, contexts = result
        assert answer == "Final answer"
        assert contexts == ['C1', 'C2']

    def test_handles_empty_pinecone_results(self):
        """Test that function handles empty Pinecone results gracefully."""
        # Arrange
        mock_index = MagicMock()
        mock_openai = MagicMock()

        mock_openai.Embedding.create.return_value = {
            'data': [{'embedding': [0.1]}]
        }
        # Pinecone returns no matches
        mock_index.query.return_value = {'matches': []}

        # Act
        answer, contexts = query_and_answer("Q?", mock_index, mock_openai)

        # Assert
        assert answer == "I'm sorry, I don't have enough information to answer that query."
        assert contexts == []
        # Should not call OpenAI completion when there are no contexts
        mock_openai.Completion.create.assert_not_called()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_build_prompt_with_very_long_contexts(self):
        """Test handling of contexts that exceed character limit."""
        # Arrange
        query = "Test query?"
        # Create a context that's longer than the limit
        very_long_context = "x" * 5000
        contexts = [very_long_context, "Short context 2", "Short context 3"]
        limit = 3750

        # Act
        result = build_prompt(query, contexts, limit=limit)

        # Assert
        # Should include the first context even though it exceeds the limit
        assert very_long_context in result
        # Should not include the second context
        assert "Short context 2" not in result
        assert "Question: Test query?" in result

    def test_build_prompt_with_multiple_contexts_exceeding_limit(self):
        """Test that contexts are truncated when combined length exceeds limit."""
        # Arrange
        query = "Test?"
        # Create contexts that together exceed the limit
        contexts = ["a" * 2000, "b" * 2000, "c" * 2000]
        limit = 3750

        # Act
        result = build_prompt(query, contexts, limit=limit)

        # Assert
        # Should include first context
        assert "a" * 2000 in result
        # Should not include third context (would exceed limit)
        assert "c" * 2000 not in result

    def test_extract_contexts_handles_missing_metadata(self):
        """Test extraction handles matches without proper metadata structure."""
        # Arrange - matches without context in metadata
        query_results = {
            'matches': [
                {'id': '1', 'metadata': {}},  # Missing 'context' key
            ]
        }

        # Act & Assert
        # This should raise a KeyError, which is expected behavior
        # In production, we'd want proper error handling
        with pytest.raises(KeyError):
            extract_contexts(query_results)

    def test_query_and_answer_with_single_context_exceeding_limit(self):
        """Test query_and_answer when single context exceeds character limit."""
        # Arrange
        mock_index = MagicMock()
        mock_openai = MagicMock()

        mock_openai.Embedding.create.return_value = {
            'data': [{'embedding': [0.1]}]
        }
        # Return a very long context
        long_context = "x" * 5000
        mock_index.query.return_value = {
            'matches': [
                {'id': '1', 'metadata': {'context': long_context}}
            ]
        }
        mock_openai.Completion.create.return_value = {
            'choices': [{'text': 'Answer based on long context'}]
        }

        # Act
        answer, contexts = query_and_answer("Q?", mock_index, mock_openai)

        # Assert
        assert answer == "Answer based on long context"
        assert contexts == [long_context]
        # Should still call OpenAI even with very long context
        mock_openai.Completion.create.assert_called_once()

    def test_build_prompt_with_special_characters_in_query(self):
        """Test prompt building with special characters in query."""
        # Arrange
        query = 'What is "Machine Learning" & AI?'
        contexts = ["ML is a field of AI."]

        # Act
        result = build_prompt(query, contexts)

        # Assert
        assert 'What is "Machine Learning" & AI?' in result
        assert "ML is a field of AI." in result

    def test_generate_answer_strips_whitespace(self):
        """Test that answer generation strips leading/trailing whitespace."""
        # Arrange
        mock_openai = MagicMock()
        mock_openai.Completion.create.return_value = {
            'choices': [{'text': '\n\n  Answer with lots of whitespace  \n\n'}]
        }
        prompt = "Test prompt"

        # Act
        result = generate_answer(prompt, mock_openai)

        # Assert
        assert result == "Answer with lots of whitespace"
        assert not result.startswith('\n')
        assert not result.endswith('\n')
