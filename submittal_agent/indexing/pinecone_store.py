"""Pinecone vector store for production deployment (query-only)."""

import logging
from typing import Optional

from openai import OpenAI
from pinecone import Pinecone

from submittal_agent.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PineconeConfig,
    RetrievalConfig,
)

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """
    Read-only Pinecone vector store for production queries.

    This class is designed for the deployed web service.
    It only queries the pre-indexed vectors in Pinecone.

    For indexing, use scripts/embedding_pipeline.py locally.
    """

    def __init__(
        self,
        index_name: str = PineconeConfig.INDEX_NAME,
        namespace: str = PineconeConfig.NAMESPACE,
    ):
        self.index_name = index_name
        self.namespace = namespace
        self._index = None
        self._openai_client = None

    @property
    def openai_client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return self._openai_client

    @property
    def index(self):
        """Lazy-load Pinecone index."""
        if self._index is None:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            self._index = pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        return self._index

    def _embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def query(
        self,
        query_text: str,
        top_k: int = RetrievalConfig.TOP_K,
    ) -> list[dict]:
        """
        Query the vector store.

        Args:
            query_text: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self._embed_query(query_text)

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        # Format results
        formatted = []
        for i, match in enumerate(results.matches, 1):
            formatted.append({
                "source_num": i,
                "content": match.metadata.get("text", ""),
                "score": match.score,
                "file": match.metadata.get("file_name", "unknown"),
                "page": match.metadata.get("page_label", ""),
            })

        return formatted

    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed collection."""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, {})
            return {
                "name": self.index_name,
                "namespace": self.namespace,
                "count": namespace_stats.get("vector_count", 0),
                "total_vectors": stats.total_vector_count,
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {
                "name": self.index_name,
                "namespace": self.namespace,
                "count": 0,
                "error": str(e),
            }


# Singleton
_pinecone_store: Optional[PineconeVectorStore] = None


def get_pinecone_store() -> PineconeVectorStore:
    """Get or create the Pinecone store singleton."""
    global _pinecone_store
    if _pinecone_store is None:
        _pinecone_store = PineconeVectorStore()
    return _pinecone_store
