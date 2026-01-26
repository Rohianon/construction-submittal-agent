"""Main review endpoint for submittal analysis."""

import time
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from submittal_agent.agent.review_agent import get_review_agent
from submittal_agent.schemas.review_response import (
    ReviewDecision,
    ReviewResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/review", response_model=ReviewResponse)
async def review_submittal(
    file: UploadFile = File(..., description="PDF submittal to review"),
    submittal_type: str = Form(default="product_data", description="Type of submittal"),
    enable_vision: bool = Form(default=True, description="Enable vision analysis"),
    additional_context: Optional[str] = Form(default=None, description="Additional context"),
):
    """
    Review a construction submittal and return approval decision.

    Accepts a PDF file and returns a structured review with:
    - Decision (approved, approved_as_noted, revise_and_resubmit, rejected)
    - Confidence score
    - Compliance checks
    - Citations to specifications
    - Required actions (if any)
    """
    start_time = time.time()

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:
        # Read file content
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Get review agent and process
        agent = get_review_agent()
        decision = agent.review(
            pdf_content=content,
            filename=file.filename,
            submittal_type=submittal_type,
            enable_vision=enable_vision,
            additional_context=additional_context or "",
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return ReviewResponse(
            success=True,
            data=decision,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Review failed: {e}")
        elapsed_ms = (time.time() - start_time) * 1000
        return ReviewResponse(
            success=False,
            error=str(e),
            processing_time_ms=elapsed_ms,
        )
