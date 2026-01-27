"""Vector store using LlamaIndex with state-of-the-art chunking strategies."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    Document,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
    NodeParser,
)
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from submittal_agent.config import (
    CHROMA_PERSIST_DIR,
    KNOWLEDGE_BASE_DIR,
    EmbeddingConfig,
    RetrievalConfig,
    OPENAI_API_KEY,
)


class ParsingStrategy(str, Enum):
    """
    Available parsing/chunking strategies.

    Each strategy has different strengths:
    - SENTENCE: Fast, simple sentence-based splitting (default)
    - SEMANTIC: Groups semantically similar content together
    - HIERARCHICAL: Creates parent/child relationships for multi-level retrieval
    - SENTENCE_WINDOW: Replaces chunks with surrounding context at query time
    """
    SENTENCE = "sentence"           # Fast, basic splitting
    SEMANTIC = "semantic"           # Semantic similarity-based splitting
    HIERARCHICAL = "hierarchical"   # Parent/child node relationships
    SENTENCE_WINDOW = "window"      # Surrounding context retrieval


def get_node_parser(
    strategy: ParsingStrategy,
    embed_model: OpenAIEmbedding,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> NodeParser:
    """
    Get the appropriate node parser based on strategy.

    Args:
        strategy: The parsing strategy to use
        embed_model: Embedding model (needed for semantic splitting)
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks

    Returns:
        Configured NodeParser instance
    """
    if strategy == ParsingStrategy.SENTENCE:
        # Fast, simple sentence-based splitting
        # Best for: Quick indexing, general purpose
        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif strategy == ParsingStrategy.SEMANTIC:
        # Splits by semantic similarity using embeddings
        # Best for: Documents where topic boundaries matter
        return SemanticSplitterNodeParser(
            buffer_size=1,  # Sentences to group
            breakpoint_percentile_threshold=95,  # Similarity threshold
            embed_model=embed_model,
        )

    elif strategy == ParsingStrategy.HIERARCHICAL:
        # Creates parent/child node relationships
        # Best for: Multi-level retrieval, finding both detail and context
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128],  # Large -> Medium -> Small
        )

    elif strategy == ParsingStrategy.SENTENCE_WINDOW:
        # Stores sentences with surrounding window context
        # Best for: Precise retrieval with context expansion
        return SentenceWindowNodeParser.from_defaults(
            window_size=3,  # Sentences on each side
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

    else:
        raise ValueError(f"Unknown parsing strategy: {strategy}")


class ConstructionVectorStore:
    """
    Vector store optimized for construction specifications.

    Uses LlamaIndex with configurable parsing strategies:
    - SENTENCE: Fast, simple splitting (default)
    - SEMANTIC: Semantic similarity-based grouping
    - HIERARCHICAL: Parent/child relationships for multi-level retrieval
    - SENTENCE_WINDOW: Surrounding context at query time

    Example usage:
        # Default (sentence-based)
        store = ConstructionVectorStore()

        # Semantic splitting (groups similar content)
        store = ConstructionVectorStore(parsing_strategy=ParsingStrategy.SEMANTIC)

        # Hierarchical (multi-level retrieval)
        store = ConstructionVectorStore(parsing_strategy=ParsingStrategy.HIERARCHICAL)

        # Sentence window (precise + context)
        store = ConstructionVectorStore(parsing_strategy=ParsingStrategy.SENTENCE_WINDOW)
    """

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = RetrievalConfig.COLLECTION_NAME,
        parsing_strategy: ParsingStrategy = ParsingStrategy.SENTENCE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.parsing_strategy = parsing_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._index: Optional[VectorStoreIndex] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None

        # Configure embedding model
        self.embed_model = OpenAIEmbedding(
            model=EmbeddingConfig.MODEL,
            api_key=OPENAI_API_KEY,
        )
        Settings.embed_model = self.embed_model

        # Configure node parser based on strategy
        self.node_parser = get_node_parser(
            strategy=parsing_strategy,
            embed_model=self.embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        Settings.node_parser = self.node_parser

    @property
    def chroma_client(self) -> chromadb.PersistentClient:
        if self._chroma_client is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
        return self._chroma_client

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            self._index = self._load_or_create_index()
        return self._index

    def _load_or_create_index(self) -> VectorStoreIndex:
        """Load existing index or create new one."""
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Check if we have existing data
        if chroma_collection.count() > 0:
            return VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
            )
        
        # Create empty index
        return VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
        )

    def ingest_directory(self, dir_path: Path | str) -> dict:
        """
        Ingest all documents from a directory using the configured parsing strategy.

        Returns dict with ingestion stats:
            - documents: Number of documents processed
            - nodes: Number of chunks/nodes created
            - strategy: Parsing strategy used
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Load documents
        reader = SimpleDirectoryReader(
            input_dir=str(dir_path),
            recursive=True,
            filename_as_id=True,
        )
        documents = reader.load_data()

        if not documents:
            return {"documents": 0, "nodes": 0, "strategy": self.parsing_strategy.value}

        # Parse documents into nodes using the configured strategy
        nodes = self.node_parser.get_nodes_from_documents(documents, show_progress=True)

        # Create index with nodes
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        return {
            "documents": len(documents),
            "nodes": len(nodes),
            "strategy": self.parsing_strategy.value,
        }

    def query(
        self,
        query_text: str,
        top_k: int = RetrievalConfig.TOP_K,
    ) -> list[dict]:
        """
        Query the vector store.
        
        Returns list of relevant chunks with metadata.
        """
        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
        )
        
        nodes = retriever.retrieve(query_text)
        
        results = []
        for i, node in enumerate(nodes, 1):
            results.append({
                "source_num": i,
                "content": node.text,
                "score": node.score,
                "file": node.metadata.get("file_name", "unknown"),
                "page": node.metadata.get("page_label", ""),
            })
        
        return results

    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed collection."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "parsing_strategy": self.parsing_strategy.value,
            }
        except Exception:
            return {
                "name": self.collection_name,
                "count": 0,
                "parsing_strategy": self.parsing_strategy.value,
            }

    def clear_collection(self) -> bool:
        """Clear all data from the collection (useful for re-indexing with different strategy)."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self._index = None
            return True
        except Exception:
            return False


def init_knowledge_base(
    strategy: ParsingStrategy = ParsingStrategy.SENTENCE,
    force_reindex: bool = True,
) -> dict:
    """
    Initialize the knowledge base from the data directory.

    Args:
        strategy: Parsing strategy to use (SENTENCE, SEMANTIC, HIERARCHICAL, SENTENCE_WINDOW)
        force_reindex: If True, clear existing index and reindex

    Returns dict with indexing stats.

    Example:
        # Default indexing
        stats = init_knowledge_base()

        # Try semantic splitting
        stats = init_knowledge_base(ParsingStrategy.SEMANTIC, force_reindex=True)

        # Try hierarchical for multi-level retrieval
        stats = init_knowledge_base(ParsingStrategy.HIERARCHICAL, force_reindex=True)
    """
    store = ConstructionVectorStore(parsing_strategy=strategy)

    if force_reindex:
        store.clear_collection()

    if not KNOWLEDGE_BASE_DIR.exists():
        KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
        return {"documents": 0, "nodes": 0, "strategy": strategy.value}

    # Check if already indexed
    stats = store.get_collection_stats()
    if stats["count"] > 0 and not force_reindex:
        return {
            "documents": "cached",
            "nodes": stats["count"],
            "strategy": strategy.value,
            "message": "Using existing index. Set force_reindex=True to rebuild.",
        }

    # Check if any documents exist
    docs = list(KNOWLEDGE_BASE_DIR.glob("**/*.pdf"))
    if not docs:
        return {"documents": 0, "nodes": 0, "strategy": strategy.value}

    return store.ingest_directory(KNOWLEDGE_BASE_DIR)


# Singleton instance with configurable strategy
_vector_store: Optional[ConstructionVectorStore] = None
_current_strategy: Optional[ParsingStrategy] = None


def get_vector_store(
    strategy: ParsingStrategy = ParsingStrategy.SENTENCE
) -> ConstructionVectorStore:
    """
    Get or create the vector store singleton.

    Args:
        strategy: Parsing strategy to use

    Note: Changing strategy requires re-initialization.
    """
    global _vector_store, _current_strategy

    if _vector_store is None or _current_strategy != strategy:
        _vector_store = ConstructionVectorStore(parsing_strategy=strategy)
        _current_strategy = strategy

    return _vector_store


# Convenience function to compare strategies
def compare_strategies(query: str, top_k: int = 5) -> dict:
    """
    Compare retrieval results across different parsing strategies.

    Useful for tuning and experimentation.

    Args:
        query: Search query
        top_k: Number of results per strategy

    Returns:
        Dict with results from each strategy
    """
    results = {}

    for strategy in ParsingStrategy:
        try:
            store = ConstructionVectorStore(
                parsing_strategy=strategy,
                collection_name=f"{RetrievalConfig.COLLECTION_NAME}_{strategy.value}",
            )
            stats = store.get_collection_stats()
            if stats["count"] > 0:
                results[strategy.value] = {
                    "results": store.query(query, top_k=top_k),
                    "node_count": stats["count"],
                }
            else:
                results[strategy.value] = {
                    "results": [],
                    "node_count": 0,
                    "message": "Not indexed. Call init_knowledge_base with this strategy first.",
                }
        except Exception as e:
            results[strategy.value] = {"error": str(e)}

    return results
