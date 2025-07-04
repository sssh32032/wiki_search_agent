import pytest
from unittest.mock import patch, MagicMock
from app.core.llm_chain import WikipediaRAGChain, GraphState

class TestWikipediaRAGChainUnit:
    @patch('app.core.llm_chain.VectorStoreRetrieverMemory')
    def test_check_memory_node(self, mock_memory):
        rag = WikipediaRAGChain()
        mock_memory.return_value.load_memory_variables.return_value = {"history": "Some answer"}
        state = GraphState({
            "input": "test question",
            "retrieved": "",
            "output": "",
            "history": [],
            "initial_node_choice": "",
            "search_and_retrieve_count": 0,
            "searched_queries": []
        })
        new_state = rag.check_memory_node(state)
        assert new_state["retrieved"] == "Some answer"
        assert new_state["history"][-1] == "check_memory"

    @patch('app.core.llm_chain.VectorStoreRetriever')
    @patch('app.core.llm_chain.cohere.Client')
    def test_retrieve_node(self, mock_cohere_client, mock_retriever):
        rag = WikipediaRAGChain()
        # Mock retriever returns docs
        mock_doc = MagicMock()
        mock_doc.page_content = "Relevant content"
        mock_retriever.return_value.get_relevant_documents.return_value = [mock_doc]
        # Mock cohere rerank
        mock_cohere_client.return_value.rerank.return_value.results = [MagicMock(index=0)]
        state = GraphState({
            "input": "test",
            "retrieved": "",
            "output": "",
            "history": [],
            "initial_node_choice": "",
            "search_and_retrieve_count": 0,
            "searched_queries": []
        })
        new_state = rag.retrieve_node(state)
        assert "Relevant content" in new_state["retrieved"]
        assert new_state["history"][-1] == "retrieve"

    @patch('app.core.llm_chain.cohere.Client')
    @patch('app.core.llm_chain.WikipediaFetcher')
    @patch('app.core.llm_chain.VectorStoreRetriever')
    def test_search_and_retrieve_node(self, mock_retriever, mock_fetcher, mock_cohere_client):
        rag = WikipediaRAGChain()
        # Mock LLM topic generation
        mock_cohere_client.return_value.chat.return_value.text = 'Test Topic'
        # Mock fetcher returns fake page
        mock_fetcher.return_value.search_and_fetch_pages.return_value = {
            'pages': [{'content': 'Wiki content'}], 'saved_file': None
        }
        # Mock text splitter
        rag.text_splitter.split_text = MagicMock(return_value=["chunk1", "chunk2"])
        # Mock retriever returns docs
        mock_doc = MagicMock()
        mock_doc.page_content = "Relevant wiki chunk"
        mock_retriever.return_value.get_relevant_documents.return_value = [mock_doc]
        state = GraphState({
            "input": "test",
            "retrieved": "",
            "output": "",
            "history": [],
            "initial_node_choice": "",
            "search_and_retrieve_count": 0,
            "searched_queries": []
        })
        new_state = rag.search_and_retrieve_node(state)
        assert "Relevant wiki chunk" in new_state["retrieved"]
        assert new_state["history"][-1] == "search_and_retrieve"

    @patch('app.core.llm_chain.cohere.Client')
    def test_llm1_decision_node(self, mock_cohere_client):
        rag = WikipediaRAGChain()
        mock_cohere_client.return_value.chat.return_value.text = '{"translated_input": "test", "initial_node_choice": "check_memory"}'
        state = GraphState({
            "input": "原始問題",
            "retrieved": "",
            "output": "",
            "history": [],
            "initial_node_choice": "",
            "search_and_retrieve_count": 0,
            "searched_queries": []
        })
        new_state = rag.llm1_decision_node(state)
        assert new_state["input"] == "test"
        assert new_state["initial_node_choice"] == "check_memory"
        assert new_state["history"][-1] == "llm1_decision"

    @patch('app.core.llm_chain.cohere.Client')
    def test_llm2_relevance_check_node(self, mock_cohere_client):
        rag = WikipediaRAGChain()
        mock_cohere_client.return_value.chat.return_value.text = 'relevant'
        state = GraphState({
            "input": "test",
            "retrieved": "Some context",
            "output": "",
            "history": [],
            "initial_node_choice": "",
            "search_and_retrieve_count": 0,
            "searched_queries": []
        })
        new_state = rag.llm2_relevance_check_node(state)
        assert new_state["output"] == "relevant"
        assert new_state["history"][-1] == "llm2_relevance_check"

    @patch('app.core.llm_chain.cohere.Client')
    @patch('app.core.llm_chain.VectorStoreRetrieverMemory')
    def test_llm3_answer_node(self, mock_memory, mock_cohere_client):
        rag = WikipediaRAGChain()
        mock_cohere_client.return_value.chat.return_value.text = 'Final answer.'
        state = GraphState({
            "input": "test",
            "retrieved": "Some context",
            "output": "relevant",
            "history": [],
            "initial_node_choice": "",
            "search_and_retrieve_count": 0,
            "searched_queries": []
        })
        new_state = rag.llm3_answer_node(state)
        assert new_state["output"] == "Final answer."
        assert new_state["history"][-1] == "llm3_answer" 