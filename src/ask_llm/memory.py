import os
import importlib.util
import logging
import time
import pathlib

# Need Config for type hint
from ask_llm.utils.config import Config

# Check for presence without importing the full libraries yet
_chromadb_present = importlib.util.find_spec("chromadb") is not None
_sentence_transformers_present = importlib.util.find_spec("sentence_transformers") is not None

if _sentence_transformers_present:
    # Dummy SentenceTransformer for type hinting if needed
    pass # No specific type needed globally for now
else:
    pass


logger = logging.getLogger(__name__)

# Default embedding model - small and fast
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_COLLECTION_NAME = "chat_memory"

class MemoryManager:
    """Manages long-term memory using a vector database."""

    # Accept config object
    def __init__(self, config: Config):
        """
        Initializes the MemoryManager using settings from the Config object.

        Args:
            config: The loaded application Config object.

        Raises:
            ImportError: If required libraries (chromadb, sentence-transformers) are not installed.
            RuntimeError: If initialization of components fails.
        """
        # --- Dependency checks (using pre-checked flags) ---
        if not _chromadb_present:
            logger.debug("ChromaDB is required for memory features. Please install it: pip install chromadb")
            raise ImportError("ChromaDB not found. Memory features unavailable.")
        if not _sentence_transformers_present:
            logger.debug("SentenceTransformers is required for memory features. Please install it: pip install sentence-transformers")
            raise ImportError("SentenceTransformers not found. Memory features unavailable.")

        # --- Store config and settings --- 
        self.config = config
        self.embedding_model_name = self.config.EMBEDDING_MODEL_NAME
        self.collection_name = self.config.CHROMA_COLLECTION_NAME
        self.db_path = self.config.CHROMA_DB_CACHE_DIR # Use the unified config path

        # --- Actual imports happen HERE, only when MemoryManager is instantiated ---
        try:
            logger.debug("Importing chromadb...")
            import chromadb
            from chromadb.api.types import EmbeddingFunction
            from chromadb.config import Settings

            logger.debug("Importing sentence_transformers...")
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.exception("Failed to import required libraries even after check passed.")
            raise RuntimeError(f"MemoryManager initialization failed during import: {e}") from e

        # --- Initialization logic using the now-imported libraries --- 
        try:
            chroma_persistent_path = pathlib.Path(self.db_path)
            chroma_persistent_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initializing ChromaDB client (Persistent in: {chroma_persistent_path})...")
            # Ensure PersistentClient is used with the configured path
            self.client = chromadb.PersistentClient(path=str(chroma_persistent_path), settings=Settings(anonymized_telemetry=False))

            logger.debug(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Define the embedding function wrapper using the imported types
            class MyEmbeddingFunction(EmbeddingFunction):
                 def __init__(self, model):
                      self.model = model
                 def __call__(self, texts):
                      return self.model.encode(texts).tolist()
            self.embedding_function = MyEmbeddingFunction(self.embedding_model)

            logger.debug(f"Getting or creating ChromaDB collection: '{self.collection_name}'")
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                # embedding_function=self.embedding_function # Chroma uses its own wrapper if embedding_function not provided
            )
            logger.debug("MemoryManager initialized successfully.")

        except Exception as e:
            logger.exception(f"Failed to initialize MemoryManager components: {e}")
            raise RuntimeError(f"MemoryManager initialization failed: {e}") from e

    def add_memory(self, message_id: str, role: str, content: str, timestamp: float):
        """
        Adds a message to the vector memory.

        Args:
            message_id: A unique identifier for the message.
            role: The role of the message sender ('user' or 'assistant').
            content: The text content of the message.
            timestamp: The time the message was created (Unix timestamp).
        """
        # Check if initialization succeeded
        if not hasattr(self, 'collection') or not hasattr(self, 'embedding_model') or not self.collection or not self.embedding_model:
             logger.error("MemoryManager is not properly initialized. Cannot add memory.")
             return

        try:
            logger.debug(f"Generating embedding for memory ID: {message_id}")
            # Ensure content is not empty or whitespace
            if not content or content.isspace():
                logger.warning(f"Skipping empty content for memory ID: {message_id}")
                return

            embedding = self.embedding_model.encode([content])[0].tolist() # Get first (only) embedding and convert to list

            logger.debug(f"Adding memory to collection '{self.collection_name}': ID={message_id}")
            self.collection.add(
                ids=[message_id],
                embeddings=[embedding],
                documents=[content], # Storing the original content is helpful
                metadatas=[{"role": role, "timestamp": timestamp}]
            )
        except Exception as e:
            logger.exception(f"Failed to add message ID {message_id} to memory: {e}")

    def search_relevant_memories(self, query_text: str, n_results: int = 5) -> list[dict] | None:
        """
        Searches for memories relevant to the query text.

        Args:
            query_text: The text to search for relevant memories.
            n_results: The maximum number of results to return.

        Returns:
            A list of dictionaries containing the retrieved memories (including documents, metadatas, distances),
            or None if the search fails or the manager is not initialized.
        """
        if not hasattr(self, 'collection') or not hasattr(self, 'embedding_model') or not self.collection or not self.embedding_model:
            logger.error("MemoryManager is not properly initialized. Cannot search memory.")
            return None
        
        if not query_text or query_text.isspace():
             logger.warning("Attempted to search memory with empty query text. Skipping.")
             return [] # Return empty list for empty query

        try:
            logger.debug(f"Generating embedding for memory search query: '{query_text[:50]}...'")
            query_embedding = self.embedding_model.encode([query_text])[0].tolist()

            logger.debug(f"Searching collection '{self.collection_name}' for {n_results} results.")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances'] # Request specific data
            )
            logger.debug(f"Memory search returned {len(results.get('ids', [[]])[0])} results.")
            # Chroma returns results nested in lists, one per query embedding. We only have one query.
            # Reformat results into a more usable list of dicts
            formatted_results = []
            ids = results.get('ids', [[]])[0]
            distances = results.get('distances', [[]])
            distances = distances[0] if distances else []
            metadatas = results.get('metadatas', [[]]) or [[]]
            metadatas = metadatas[0] if metadatas else []
            documents = results.get('documents', [[]])
            documents = documents[0] if documents else []

            for i, doc_id in enumerate(ids):
                formatted_results.append({
                    'id': doc_id,
                    'distance': distances[i],
                    'metadata': metadatas[i],
                    'document': documents[i]
                })

            return formatted_results

        except Exception as e:
            logger.exception(f"Failed to search memory for query '{query_text[:50]}...': {e}")
            return None

    def peek_collection(self, n: int = 5):
        """Helper method for debugging: Shows the first N items in the collection."""
        if not self.collection:
             logger.error("MemoryManager is not properly initialized. Cannot peek collection.")
             return None
        try:
            return self.collection.peek(limit=n)
        except Exception as e:
            logger.exception(f"Failed to peek collection '{self.collection_name}': {e}")
            return None

    # search_relevant_memories will be added in Stage 2 