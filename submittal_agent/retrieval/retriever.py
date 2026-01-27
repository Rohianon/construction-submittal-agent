"""Retrieval module for fetching relevant specifications."""

import logging
from typing import Optional, Protocol

from submittal_agent.config import RetrievalConfig, VECTOR_STORE_BACKEND

logger = logging.getLogger(__name__)


class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations."""

    def query(self, query_text: str, top_k: int) -> list[dict]:
        """Query the vector store."""
        ...


def get_vector_store() -> VectorStoreProtocol:
    """
    Get the appropriate vector store based on configuration.

    Uses VECTOR_STORE_BACKEND env var:
      - "pinecone" -> PineconeVectorStore (for production/Render)
      - "chroma" -> ConstructionVectorStore (for local development)
    """
    if VECTOR_STORE_BACKEND == "pinecone":
        from submittal_agent.indexing.pinecone_store import get_pinecone_store
        logger.info("Using Pinecone vector store (production)")
        return get_pinecone_store()
    else:
        from submittal_agent.indexing.vector_store import (
            get_vector_store as get_chroma_store,
        )
        logger.info("Using ChromaDB vector store (local)")
        return get_chroma_store()


class SpecificationRetriever:
    """
    Retrieves relevant specification chunks for submittal review.

    Uses the configured vector store (Pinecone or ChromaDB).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreProtocol] = None,
        top_k: int = RetrievalConfig.TOP_K,
        relevance_threshold: float = RetrievalConfig.RELEVANCE_THRESHOLD,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        submittal_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve relevant specification chunks.

        Args:
            query: Search query (typically submittal content summary)
            submittal_type: Type of submittal for filtering

        Returns:
            List of relevant chunks with metadata
        """
        # Build enhanced query
        enhanced_query = query
        if submittal_type:
            enhanced_query = f"{submittal_type}: {query}"

        # Query vector store
        results = self.vector_store.query(
            query_text=enhanced_query,
            top_k=self.top_k * 2,  # Over-fetch for filtering
        )

        # Filter by relevance threshold
        filtered = [
            r for r in results
            if r.get("score", 0) >= self.relevance_threshold
        ]

        # Return top_k after filtering
        return filtered[:self.top_k]

    def build_context(self, chunks: list[dict]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return "No relevant specifications found."

        parts = []
        for chunk in chunks:
            source_num = chunk.get("source_num", 0)
            file_name = chunk.get("file", "unknown")
            content = chunk.get("content", "")
            page = chunk.get("page", "")

            header = f"[Source {source_num}] {file_name}"
            if page:
                header += f" (Page {page})"

            parts.append(f"{header}\n{content}\n---")

        return "\n\n".join(parts)


# Singleton
_retriever: Optional[SpecificationRetriever] = None


def get_retriever() -> SpecificationRetriever:
    """Get or create the retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = SpecificationRetriever()
    return _retriever
