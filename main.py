"""FastAPI application for Construction Submittal Review Agent."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import health, review, documents
from submittal_agent.config import KNOWLEDGE_BASE_DIR, IS_PRODUCTION, VECTOR_STORE_BACKEND

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize knowledge base on startup."""
    logger.info("Starting Construction Submittal Review Agent...")
    logger.info(f"Vector store backend: {VECTOR_STORE_BACKEND}")

    # Only init local ChromaDB if not using Pinecone
    if VECTOR_STORE_BACKEND != "pinecone":
        from submittal_agent.indexing.vector_store import init_knowledge_base, ParsingStrategy
        if KNOWLEDGE_BASE_DIR.exists() and list(KNOWLEDGE_BASE_DIR.glob("**/*.pdf")):
            logger.info("Initializing local knowledge base...")
            result = init_knowledge_base(strategy=ParsingStrategy.SENTENCE)
            logger.info(f"Knowledge base initialized: {result}")
        else:
            logger.warning(f"No documents found in {KNOWLEDGE_BASE_DIR}")
    else:
        logger.info("Using Pinecone - skipping local knowledge base init")

    yield

    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Construction Submittal Review Agent",
    description="AI-powered review of construction submittals with multimodal analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(review.router, prefix="/api", tags=["Review"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the frontend or redirect to docs."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Construction Submittal Review Agent API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=not IS_PRODUCTION,
    )
