# Construction Submittal Review Agent

AI-powered agent that reviews construction submittals (shop drawings, product data, material samples) and produces structured approval/rejection responses following AIA/EJCDC standards.

## Features

- **Multimodal Analysis**: Processes both PDF documents and embedded images/drawings using Claude Vision
- **RAG-powered Context**: Uses LlamaIndex with configurable parsing strategies for accurate retrieval
- **Dual LLM Support**: Primary Claude (claude-sonnet-4-20250514) with GPT-4 fallback
- **Structured Output**: Returns decisions with confidence scores, citations, and detailed comments
- **AIA/EJCDC Compliant**: Decisions follow industry-standard categories (Approved, Approved as Noted, Revise and Resubmit, Rejected)
- **Modern Frontend**: Tailwind CSS + Alpine.js interface with drag-and-drop upload

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Alpine.js)                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐                     │
│  │ /health  │ │ /review  │ │ /documents   │                     │
│  └──────────┘ └────┬─────┘ └──────────────┘                     │
└─────────────────────┼───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Review Agent                                  │
│  ┌────────────────┐ ┌─────────────────┐ ┌───────────────────┐   │
│  │ Vision Analyzer│ │ Context Retriever│ │ LLM Client       │   │
│  │ (Claude Vision)│ │ (LlamaIndex RAG) │ │ (Claude + GPT-4) │   │
│  └────────────────┘ └─────────────────┘ └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              Knowledge Base (ChromaDB)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Configurable Parsing Strategies:                        │    │
│  │ • SentenceSplitter (fast, basic)                        │    │
│  │ • SemanticSplitterNodeParser (similarity-based)         │    │
│  │ • HierarchicalNodeParser (parent/child relationships)   │    │
│  │ • SentenceWindowNodeParser (surrounding context)        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.13+
- Anthropic API key
- OpenAI API key (for embeddings and fallback)

### Installation

```bash
# Clone the repository
git clone https://github.com/Rohianon/construction-submittal-agent.git
cd construction-submittal-agent

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Add Knowledge Base

Place your project specification PDFs in the knowledge base directory:

```bash
mkdir -p data/knowledge_base
# Copy your PDF specifications to data/knowledge_base/
```

### Run Locally

```bash
python main.py
# or
uvicorn main:app --reload
```

Open http://localhost:8000 in your browser.

## API Reference

### POST /api/review

Review a construction submittal.

**Request (multipart/form-data):**
- `file`: PDF file to review (required)
- `project_name`: Project name (optional)
- `spec_section`: Specification section reference (optional)
- `parsing_strategy`: One of `sentence`, `semantic`, `hierarchical`, `window` (optional, default: `sentence`)

**Response:**
```json
{
  "decision": "APPROVED_AS_NOTED",
  "confidence_score": 0.85,
  "reasoning": "The submitted material meets specifications with minor clarifications needed.",
  "detailed_comments": [
    {
      "category": "technical",
      "severity": "medium",
      "comment": "Finish color specified as 'warm gray' - confirm this matches spec section 09 90 00 color schedule.",
      "location": "Page 2, Material Schedule"
    }
  ],
  "compliance_checks": {
    "material_compliance": {
      "status": "pass",
      "details": "Material meets ASTM A36 requirements"
    }
  },
  "required_actions": [
    "Confirm color selection with architect"
  ],
  "citations": [
    {
      "source": "Project Specifications",
      "section": "05 12 00",
      "page": 3,
      "quote": "All structural steel shall conform to ASTM A36"
    }
  ]
}
```

### GET /api/health

Health check endpoint.

### GET /api/documents

List documents in the knowledge base.

### POST /api/documents/ingest

Re-ingest knowledge base with a different parsing strategy.

## Parsing Strategies

The system supports multiple LlamaIndex parsing strategies for experimentation:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `sentence` | Basic sentence-level splitting | Fast processing, general use |
| `semantic` | Similarity-based chunking | Documents with varying section lengths |
| `hierarchical` | Parent/child node relationships | Complex structured documents |
| `window` | Preserves surrounding context | Detailed cross-reference needs |

Switch strategies via the `/api/documents/ingest` endpoint or the `parsing_strategy` parameter on reviews.

## Deployment

### Docker

```bash
docker build -t submittal-agent .
docker run -p 8000:8000 --env-file .env submittal-agent
```

### Railway/Render

1. Connect your GitHub repository
2. Set environment variables:
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `ENVIRONMENT=production`
3. Deploy

## Project Structure

```
construction-submittal-agent/
├── main.py                     # FastAPI application entry point
├── api/
│   └── routes/
│       ├── health.py           # Health check endpoint
│       ├── review.py           # Main review endpoint
│       └── documents.py        # Document management
├── submittal_agent/
│   ├── agent/
│   │   ├── llm_client.py       # Unified LLM client (Claude + GPT-4)
│   │   ├── prompts.py          # System prompts
│   │   ├── review_agent.py     # Main orchestrator
│   │   └── vision_analyzer.py  # Image/drawing analysis
│   ├── indexing/
│   │   └── vector_store.py     # ChromaDB + LlamaIndex
│   ├── ingestion/
│   │   ├── pdf_parser.py       # PDF text extraction
│   │   └── image_extractor.py  # Drawing extraction
│   ├── retrieval/
│   │   └── retriever.py        # RAG retrieval
│   └── schemas/
│       └── review_response.py  # Pydantic models
├── static/
│   └── index.html              # Frontend (Tailwind + Alpine.js)
├── data/
│   └── knowledge_base/         # Project specifications
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Decision Types

Following AIA/EJCDC standards:

- **APPROVED**: Submittal conforms to contract documents
- **APPROVED_AS_NOTED**: Approved with minor modifications noted
- **REVISE_AND_RESUBMIT**: Requires corrections before approval
- **REJECTED**: Does not meet requirements, resubmission required

## License

MIT
