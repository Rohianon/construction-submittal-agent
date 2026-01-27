#!/usr/bin/env python3
"""
Embedding Pipeline for Construction Specifications.

This standalone script processes PDFs from the knowledge base,
generates embeddings using OpenAI, and uploads them to Pinecone.

Run this locally before deploying the web service:
    python scripts/embedding_pipeline.py

Requirements:
    - OPENAI_API_KEY in .env
    - PINECONE_API_KEY in .env
    - PDFs in data/knowledge_base/

The web service only needs to query Pinecone (read-only),
this pipeline handles the one-time indexing.
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Generator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# Load environment
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "construction-specs")
PINECONE_NAMESPACE = "qcs-2024"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
BATCH_SIZE = 100

# Paths
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"


def validate_environment():
    """Validate required environment variables."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Please add them to your .env file.")
        sys.exit(1)


def get_pinecone_index():
    """Initialize Pinecone and get or create the index."""
    print(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(1)

    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Connected to index '{PINECONE_INDEX_NAME}'")

    # Show current stats
    stats = index.describe_index_stats()
    print(f"Current vectors: {stats.total_vector_count}")

    return index


def load_documents() -> list:
    """Load PDF documents from knowledge base."""
    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"Error: Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
        sys.exit(1)

    pdf_files = list(KNOWLEDGE_BASE_DIR.glob("**/*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {KNOWLEDGE_BASE_DIR}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files")

    reader = SimpleDirectoryReader(
        input_dir=str(KNOWLEDGE_BASE_DIR),
        recursive=True,
        filename_as_id=True,
    )

    print("Loading documents...")
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document pages")

    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into chunks."""
    print(f"Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    print(f"Created {len(nodes)} chunks")

    return nodes


def generate_chunk_id(content: str, metadata: dict) -> str:
    """Generate a deterministic ID for a chunk."""
    unique_string = f"{metadata.get('file_name', '')}:{metadata.get('page_label', '')}:{content[:100]}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def batch_embed(
    client: OpenAI, texts: list[str], batch_size: int = BATCH_SIZE
) -> Generator[list[list[float]], None, None]:
    """Generate embeddings in batches."""
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        yield [item.embedding for item in response.data]


def upload_to_pinecone(index, nodes: list):
    """Generate embeddings and upload to Pinecone."""
    print(f"\nGenerating embeddings and uploading to Pinecone...")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Prepare all texts and metadata
    texts = [node.text for node in nodes]
    metadatas = []
    ids = []

    for node in nodes:
        metadata = {
            "text": node.text[:1000],  # Store truncated text in metadata
            "file_name": node.metadata.get("file_name", "unknown"),
            "page_label": str(node.metadata.get("page_label", "")),
        }
        metadatas.append(metadata)
        ids.append(generate_chunk_id(node.text, node.metadata))

    # Process in batches
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    vectors_uploaded = 0

    for batch_idx, embeddings in enumerate(batch_embed(client, texts)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + len(embeddings), len(nodes))

        # Prepare vectors for this batch
        vectors = []
        for i, embedding in enumerate(embeddings):
            idx = start_idx + i
            vectors.append(
                {
                    "id": ids[idx],
                    "values": embedding,
                    "metadata": metadatas[idx],
                }
            )

        # Upsert to Pinecone
        index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
        vectors_uploaded += len(vectors)

        # Progress
        progress = (batch_idx + 1) / total_batches * 100
        print(f"  Progress: {progress:.1f}% ({vectors_uploaded}/{len(nodes)} vectors)")

    return vectors_uploaded


def main():
    """Run the embedding pipeline."""
    print("=" * 60)
    print("Construction Specifications Embedding Pipeline")
    print("=" * 60)
    print()

    # Validate environment
    validate_environment()

    # Initialize Pinecone
    index = get_pinecone_index()

    # Load documents
    documents = load_documents()

    # Chunk documents
    nodes = chunk_documents(documents)

    # Upload to Pinecone
    start_time = time.time()
    vectors_uploaded = upload_to_pinecone(index, nodes)
    elapsed = time.time() - start_time

    # Final stats
    print()
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"  Documents processed: {len(documents)}")
    print(f"  Chunks created: {len(nodes)}")
    print(f"  Vectors uploaded: {vectors_uploaded}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Index: {PINECONE_INDEX_NAME}")
    print(f"  Namespace: {PINECONE_NAMESPACE}")
    print()

    # Show final index stats
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
