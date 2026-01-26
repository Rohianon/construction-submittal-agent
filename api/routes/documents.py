"""Knowledge base document management endpoints."""

from fastapi import APIRouter

from submittal_agent.indexing.vector_store import (
    get_vector_store,
    init_knowledge_base,
    ParsingStrategy,
)
from submittal_agent.config import KNOWLEDGE_BASE_DIR

router = APIRouter()


@router.get("/documents")
async def list_documents():
    """List indexed knowledge base documents and statistics."""
    store = get_vector_store()
    stats = store.get_collection_stats()

    # List available PDF files
    pdf_files = []
    if KNOWLEDGE_BASE_DIR.exists():
        pdf_files = [f.name for f in KNOWLEDGE_BASE_DIR.glob("**/*.pdf")]

    return {
        "indexed": stats,
        "available_files": pdf_files,
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR),
    }


@router.post("/documents/reindex")
async def reindex_documents(
    strategy: str = "sentence",
    force: bool = False,
):
    """
    Reindex the knowledge base with a specific parsing strategy.

    Strategies:
    - sentence: Fast, simple splitting (default)
    - semantic: Semantic similarity-based splitting
    - hierarchical: Parent/child relationships
    - window: Sentence window with context
    """
    try:
        parsing_strategy = ParsingStrategy(strategy)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid strategy. Choose from: {[s.value for s in ParsingStrategy]}",
        }

    result = init_knowledge_base(strategy=parsing_strategy, force_reindex=force)

    return {
        "success": True,
        "result": result,
    }
