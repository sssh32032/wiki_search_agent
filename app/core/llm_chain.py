"""
LLM Chain module using LangGraph for modular RAG workflow
"""

import os
# Disable OpenTelemetry to prevent connection errors
os.environ["OTEL_SDK_DISABLED"] = "true"

import json
import logging
from typing import TypedDict, List, Dict, Any
from pathlib import Path
from logging.handlers import RotatingFileHandler
import re

# Guardrails imports
from guardrails import Guard
from guardrails.hub import DetectJailbreak, ToxicLanguage

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangChain imports
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.memory import VectorStoreRetrieverMemory
from langchain.tools import tool

# External imports
import cohere

# Local imports
from app.config import settings

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(console_handler)

# File handler (rotating)
file_handler = RotatingFileHandler('logs/rag_pipeline.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(file_handler)

MAX_RETRY_COUNT = 5

class GraphState(TypedDict):
    """
    State for the RAG workflow:
    - input: user input (possibly translated)
    - retrieved: current context retrieved from memory/vector DB/external
    - output: final answer or intermediate output
    - history: list of node names traversed
    - initial_node_choice: str, one of 'check_memory', 'retrieve', 'search_and_retrieve' (set by LLM1_decision)
    - search_and_retrieve_count: int, how many times search_and_retrieve has been called
    - input_validated: bool, whether the input has passed validation
    - output_validated: bool, whether the output has passed validation
    - llm3_retry: int, how many times the LLM3 answer has been retried
    - final_relevance_result: str, the result of the relevance check
    """
    input: str
    retrieved: str
    output: str
    history: list[str]
    initial_node_choice: str
    search_and_retrieve_count: int
    searched_queries: list[str]
    input_validated: bool
    output_validated: bool
    llm3_retry: int
    final_relevance_result: str

def log_state_summary(state: GraphState, node_name: str):
    """Log a summary of state with only key information"""
    summary = {
        "input": state.get("input", ""),
        "output": state.get("output", ""),
        "search_and_retrieve_count": state.get("search_and_retrieve_count", 0),
        "searched_queries": state.get("searched_queries", []),
        "input_validated": state.get("input_validated", False),
        "output_validated": state.get("output_validated", False),
        "llm3_retry": state.get("llm3_retry", 0)
    }
    logger.info(f"[Node] {node_name} state summary: {summary}")

class WikipediaRAGChain:
    """Wikipedia RAG Chain using LangGraph"""
    def __init__(self):
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Text splitter with config - using smaller chunk size for token limit
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=min(settings.chunk_size, 256),  # Respect model token limit
            chunk_overlap=min(settings.chunk_overlap, 50),   # Respect model token limit
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Cohere client (legacy, for direct API calls)
        self.cohere_client = cohere.Client(settings.cohere_api_key)
        
        # Setup Vector DB (wiki_db)
        self.vector_db, self.vector_db_path = self._setup_vector_db()
        
        # Setup Memory DB
        self.memory_db, self.memory_db_path = self._setup_memory_db()
        
        # Setup Graph
        self.graph = self._setup_graph()
        
        # Initialize components
        self.retriever = VectorStoreRetriever(
            vectorstore=self.vector_db,
            search_kwargs={"k": settings.top_k}  # Use settings.top_k
        )
        self.memory = VectorStoreRetrieverMemory(
            retriever=VectorStoreRetriever(vectorstore=self.memory_db)
        )
        self.app = self.graph.compile()

    def _setup_vector_db(self):
        """Setup Vector DB (wiki_db) for storing Wikipedia content"""
        vector_db_path = Path("faiss_index/wiki_db")
        vector_db_path.mkdir(parents=True, exist_ok=True)
        if (vector_db_path / "index.faiss").exists():
            try:
                vector_db = FAISS.load_local(str(vector_db_path), self.embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded existing vector database (wiki_db)")
            except Exception as e:
                logger.warning(f"Failed to load existing vector database: {e}")
                # Create new database if loading fails
                vector_db = FAISS.from_texts(["Initial placeholder text for database creation"], self.embeddings)
                vector_db.save_local(str(vector_db_path))
                logger.info("Created new vector database (wiki_db)")
        else:
            # Create empty vector database with proper initialization
            vector_db = FAISS.from_texts(["Initial placeholder text for database creation"], self.embeddings)
            vector_db.save_local(str(vector_db_path))
            logger.info("Created new vector database (wiki_db)")
        
        return vector_db, vector_db_path

    def _setup_memory_db(self):
        """Setup Memory DB for Q&A cache (persistent)"""
        memory_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        memory_db_path = Path("faiss_index/memory_db")
        memory_db_path.mkdir(parents=True, exist_ok=True)
        if (memory_db_path / "index.faiss").exists():
            try:
                memory_db = FAISS.load_local(str(memory_db_path), memory_embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded existing memory_db from disk")
            except Exception as e:
                logger.warning(f"Failed to load memory_db: {e}")
                memory_db = FAISS.from_texts(["Initial memory placeholder"], memory_embeddings)
                memory_db.save_local(str(memory_db_path))
                logger.info("Created new memory_db on disk")
        else:
            memory_db = FAISS.from_texts(["Initial memory placeholder"], memory_embeddings)
            memory_db.save_local(str(memory_db_path))
            logger.info("Created new memory_db on disk")
        
        return memory_db, memory_db_path

    def _setup_graph(self):
        logger.info('Setting up RAG graph with Guardrails')
        graph = StateGraph(GraphState)
        graph.add_node("input_validation", self.input_validation_node)
        graph.add_node("llm1_decision", self.llm1_decision_node)
        graph.add_node("check_memory", self.check_memory_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("search_and_retrieve", self.search_and_retrieve_node)
        graph.add_node("llm2_relevance_check", self.llm2_relevance_check_node)
        graph.add_node("llm3_answer", self.llm3_answer_node)
        graph.add_node("output_validation_node", self.output_validation_node)
        graph.set_entry_point("input_validation")

        def input_validation_edge(state):
            if state.get("input_validated", False):
                return "llm1_decision"
            else:
                return END

        def llm1_decision_edge(state):
            return state["initial_node_choice"]

        def llm2_relevance_check_edge(state):
            if state["output"] == "sufficient":
                return "llm3_answer"
            last = state["history"][-2] if len(state["history"]) >= 2 else ""
            if last == "check_memory":
                return "retrieve"
            if last == "retrieve":
                return "search_and_retrieve"
            if last == "search_and_retrieve":
                if state["search_and_retrieve_count"] < 3:
                    return "search_and_retrieve"
                else:
                    return "llm3_answer"
            return "llm3_answer"

        def output_validation_edge(state):
            if state.get("output_validated", False):
                return END
            else:
                if state.get("llm3_retry", 0) < 3:
                    return "llm3_answer"
                else:
                    state["output"] = "Sorry, the system could not generate a safe response. Please try again later."
                    return END

        graph.add_conditional_edges("input_validation", input_validation_edge)
        graph.add_conditional_edges("llm1_decision", llm1_decision_edge)
        graph.add_edge("check_memory", "llm2_relevance_check")
        graph.add_edge("retrieve", "llm2_relevance_check")
        graph.add_edge("search_and_retrieve", "llm2_relevance_check")
        graph.add_conditional_edges("llm2_relevance_check", llm2_relevance_check_edge)
        graph.add_edge("llm3_answer", "output_validation_node")
        graph.add_conditional_edges("output_validation_node", output_validation_edge)
        return graph

    # Node: Check memory for existing answer
    def check_memory_node(self, state: GraphState) -> GraphState:
        """Node: Check memory for an existing answer to the user's question."""
        log_state_summary(state, "check_memory")
        result = self.memory.load_memory_variables({"query": state["input"]})
        state["retrieved"] = result.get("history", "")
        state["history"].append("check_memory")
        return state

    # Node: Retrieve relevant information from the vector database
    def retrieve_node(self, state: GraphState) -> GraphState:
        """Node: Retrieve relevant information from the vector database with Cohere Rerank."""
        log_state_summary(state, "retrieve")
        
        # Get initial documents
        docs = self.retriever.invoke(state["input"])
        if not docs:
            state["retrieved"] = ""
            state["history"].append("retrieve")
            return state
        
        # Extract passages for reranking
        passages = [doc.page_content for doc in docs]
        
        try:
            # Use Cohere Rerank to improve relevance
            logger.info(f"[Node] retrieve: Applying Cohere Rerank to {len(passages)} documents")
            rerank_results = self.cohere_client.rerank(
                query=state["input"], 
                documents=passages, 
                top_n=settings.rerank_top_n
            )
            
            # Get reranked documents
            reranked_docs = [docs[r.index] for r in rerank_results.results]
            reranked_passages = [doc.page_content for doc in reranked_docs]
            
            logger.info(f"[Node] retrieve: Rerank completed, selected top {len(reranked_passages)} documents")
            state["retrieved"] = "\n".join(reranked_passages)
            
        except Exception as e:
            logger.warning(f"[Node] retrieve: Rerank failed, using original results: {e}")
            # Fallback to original results
            passages = [doc.page_content for doc in docs[:settings.rerank_top_n]]
            state["retrieved"] = "\n".join(passages)
        
        state["history"].append("retrieve")
        return state

    # Node: Search Wikipedia, update DB, then retrieve
    def search_and_retrieve_node(self, state: GraphState) -> GraphState:
        """Node: Search Wikipedia, update the database, then retrieve relevant information."""
        log_state_summary(state, "search_and_retrieve")
        # Ensure searched_queries exists in state
        if "searched_queries" not in state:
            state["searched_queries"] = []
        # Use LLM to generate a new topic for wiki search
        topic_prompt = f"""
You are an assistant to provide Wikipedia search queries to retrieve sufficient information for the user's question.
The previously searched queries cannot retrieve sufficient information for the user's question.
So, given the following user question and previously searched queries, provide an informative and non-duplicate query to retrieve the missing information for a comprehensive answer.
Only return the query string, do not include any explanation or formatting.
User question: {state['input']}
Previously searched queries: {state['searched_queries']}
"""
        response = self.cohere_client.chat(message=topic_prompt)
        query = response.text.strip().strip('"')
        logger.info(f"[Node] search_and_retrieve_node LLM suggested query: {query}")
        # Avoid duplicate search
        if query in state["searched_queries"]:
            logger.info(f"[Node] search_and_retrieve_node: query '{query}' already searched, skipping.")
            state["retrieved"] = ""
            state["output"] = "No new query to search."
            state["history"].append("search_and_retrieve")
            state["search_and_retrieve_count"] += 1
            return state
        # Add query to searched_queries
        state["searched_queries"].append(query)
        # --- WikipediaFetcher fetch and update logic ---
        from scripts.fetch_wiki import WikipediaFetcher
        import os
        from pathlib import Path
        
        fetcher = WikipediaFetcher()
        search_results = fetcher.search_and_fetch_pages(query, limit=settings.wiki_max_pages)
        new_texts = []
        for page in search_results.get('pages', []):
            chunks = self.text_splitter.split_text(page['content'])
            new_texts.extend(chunks)
        if new_texts:
            self.vector_db.add_texts(new_texts)
            self.vector_db.save_local(str(self.vector_db_path))
            logger.info(f"Added {len(new_texts)} new chunks to vector database")
        
        # Auto-cleanup: Delete the specific file that was just created
        try:
            saved_file = search_results.get('saved_file')
            if saved_file and os.path.exists(saved_file):
                os.remove(saved_file)
                logger.info(f"Auto-cleanup: Deleted newly created file {saved_file}")
            else:
                logger.info("Auto-cleanup: No file to clean up (no saved_file in results)")
        except Exception as e:
            logger.warning(f"Auto-cleanup failed: {e}")
        # Retrieve again after update
        docs = self.retriever.invoke(state["input"])
        passages = [doc.page_content for doc in docs]
        if passages:
            state["retrieved"] = "\n".join(passages)
        else:
            state["retrieved"] = ""
        state["output"] = ""
        state["search_and_retrieve_count"] += 1
        state["history"].append("search_and_retrieve")
        return state
    
    # Node: LLM1 Decision
    def llm1_decision_node(self, state: GraphState) -> GraphState:
        """LLM1 node: uses Cohere LLM to translate and decide next action."""
        log_state_summary(state, "llm1_decision")
        prompt = f"""
You are an assistant to handle user input for a Wikipedia RAG system.
If the user input is not in English, translate it to English and use that as the new input.
Then, based on the new input, decide which action to take next.
You can choose one of: 'check_memory', 'retrieve', or 'search_and_retrieve'.
If the user just asks a question, choose 'check_memory'. This is the default action.
If the user wants to retrieve in the database, choose 'retrieve'.
If the user specifies to perform a new search, choose 'search_and_retrieve'.
Return only a valid JSON object with two keys: 'translated_input' (the translated English input) and 'initial_node_choice' ('check_memory', 'retrieve', or 'search_and_retrieve').
Do NOT include any explanation or markdown code block.
User input: {state['input']}
"""
        try:
            response = self.cohere_client.chat(message=prompt)
            # Try to parse JSON output from LLM
            import json
            result = json.loads(response.text)
            state["input"] = result.get("translated_input", state["input"])
            state["initial_node_choice"] = result.get("initial_node_choice", "check_memory")
            logger.info(f"[Node] llm1_decision: Parsed result - {result}")
        except Exception as e:
            logger.warning(f"[Node] llm1_decision: JSON parsing failed - {e}")
            state["input"] = state["input"]
            state["initial_node_choice"] = "check_memory"
        
        state["history"].append("llm1_decision")
        return state
    
    # Node: LLM2 Relevance Check
    def llm2_relevance_check_node(self, state: GraphState) -> GraphState:
        """LLM2 node: checks if the retrieved content is sufficient to answer the question."""
        log_state_summary(state, "llm2_relevance_check")
        if not state["retrieved"].strip():
            state["output"] = "insufficient"
            state["history"].append("llm2_relevance_check")
            return state
        prompt = f"""
You are an assistant that checks if the provided context sufficiently answers all aspects of the question without missing information.
Respond with only one of the following: 'sufficient' or 'insufficient'.
Do NOT include any explanation or markdown code block.
Context: {state['retrieved']}
Question: {state['input']}
"""
        try:
            response = self.cohere_client.chat(message=prompt)
            response_text = response.text.strip().lower()
            # explicitly check for both keywords
            if "insufficient" in response_text:
                state["output"] = "insufficient"
            elif "sufficient" in response_text:
                state["output"] = "sufficient"
            else:
                # Default to insufficient if neither keyword is found
                state["output"] = "insufficient"
            logger.info(f"[Node] llm2_relevance_check: Response '{response_text}'")
        except Exception as e:
            logger.warning(f"[Node] llm2_relevance_check: API call failed - {e}")
            state["output"] = "insufficient"
        
        state["history"].append("llm2_relevance_check")
        return state

    # Node: LLM3 Answer
    def llm3_answer_node(self, state: GraphState) -> GraphState:
        """Node: Generate the final answer using Cohere LLM. Only set final_relevance_result if it is empty."""
        log_state_summary(state, "llm3_answer")
        if not state["retrieved"].strip():
            state["output"] = "No relevant information found."
            if not state.get("final_relevance_result"):
                state["final_relevance_result"] = "insufficient"
            state["history"].append("llm3_answer")
            return state
        # Only set final_relevance_result if it is empty
        if not state.get("final_relevance_result"):
            state["final_relevance_result"] = state["output"]
        prompt = f"""
You are an assistant. You must answer the question strictly based on the provided context.
If the context does not contain the answer, reply with 'No relevant information found.'
Do NOT make up or supplement any information that is not present in the context. Do NOT hallucinate or fabricate answers.
Context: {state['retrieved']}
Question: {state['input']}
"""
        response = self.cohere_client.chat(message=prompt)
        logger.info(f"[Node] llm3_answer Cohere response: {response.text}")
        state["output"] = response.text.strip()
        state["history"].append("llm3_answer")
        return state

    def input_validation_node(self, state: GraphState) -> GraphState:
        """Node: Validate user input for jailbreak attempts using Guardrails."""
        log_state_summary(state, "input_validation")
        guard = Guard().use(DetectJailbreak, threshold=0.7, on_fail="exception")
        try:
            guard.validate(state["input"])
            state["input_validated"] = True
            state["history"].append("input_validation")
            logger.info("[Node] input_validation: Input passed jailbreak detection")
        except Exception as e:
            state["input_validated"] = False
            state["output"] = "Your input was flagged as unsafe. Please rephrase your question."
            state["history"].append("input_validation_failed")
            logger.warning(f"[Node] input_validation: Jailbreak detected - {e}")
            logger.info(f"[Node] input_validation output: {state['output']}")
        return state

    def output_validation_node(self, state: GraphState) -> GraphState:
        """Node: Validate output content for toxic language using Guardrails. Save to memory if valid and relevant."""
        log_state_summary(state, "output_validation")
        guard = Guard().use(ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception")
        try:
            guard.validate(state["output"])
            state["output_validated"] = True
            state["history"].append("output_validation")
            logger.info("[Node] output_validation: Output passed toxic language detection")
            # Save to memory only if relevance is sufficient
            if state.get("final_relevance_result", "") == "sufficient":
                self.memory.save_context({"query": state["input"]}, {"answer": state["output"]})
                self.memory_db.save_local(str(self.memory_db_path))
                logger.info(f"[Node] output_validation: Saved completed Q&A to memory_db")
            else:
                logger.info(f"[Node] output_validation: Skipped saving to memory_db (content was insufficient)")
        except Exception as e:
            state["output_validated"] = False
            state["llm3_retry"] = state.get("llm3_retry", 0) + 1
            state["output"] = f"Output validation failed: {e}"
            state["history"].append("output_validation_failed")
            logger.warning(f"[Node] output_validation: Toxic language detected - {e}")
        return state

    def generate(self, query: str) -> str:
        logger.info(f"[RAG] generate called with query: {query}")
        state = GraphState({
            "input": query,
            "retrieved": "",
            "output": "",
            "history": [],
            "initial_node_choice": "",
            "search_and_retrieve_count": 0,
            "searched_queries": [],
            "input_validated": False,
            "output_validated": False,
            "llm3_retry": 0,
            "final_relevance_result": ""
        })
        result = self.app.invoke(state)
        return result["output"]

# For backward compatibility
class LLMChainGraph:
    def __init__(self, retriever=None, **kwargs):
        self.rag_chain = WikipediaRAGChain()
    def generate(self, query: str, top_k: int = 5) -> str:
        return self.rag_chain.generate(query) 