"""Pydantic models for structured review responses."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Standard construction submittal review decisions per AIA/EJCDC standards."""
    APPROVED = "approved"
    APPROVED_AS_NOTED = "approved_as_noted"
    REVISE_AND_RESUBMIT = "revise_and_resubmit"
    REJECTED = "rejected"


class ComplianceStatus(str, Enum):
    """Status for individual compliance checks."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class ComplianceCheck(BaseModel):
    """Result of a single compliance check."""
    status: ComplianceStatus
    notes: str = Field(description="Explanation of check result")


class Citation(BaseModel):
    """A citation to source specification."""
    source_num: int = Field(description="Source number [1], [2], etc.")
    file: str = Field(description="Source document filename")
    section: Optional[str] = Field(default=None, description="Spec section number")
    page: Optional[str] = Field(default=None, description="Page number or label")
    quote: str = Field(description="Relevant quote from source")


class DetailedComment(BaseModel):
    """A specific comment about the submittal."""
    category: str = Field(description="Category: specification, code, completeness, quality, coordination")
    comment: str = Field(description="The comment text")
    citation: Optional[str] = Field(default=None, description="Source citation [Source N]")


class DrawingAnalysis(BaseModel):
    """Analysis results from vision processing of drawings."""
    images_analyzed: int = Field(default=0, description="Number of images analyzed")
    dimensions_found: list[str] = Field(default_factory=list, description="Extracted dimensions")
    materials_identified: list[str] = Field(default_factory=list, description="Identified materials")
    potential_issues: list[str] = Field(default_factory=list, description="Potential issues found")
    notes: str = Field(default="", description="General notes about the drawings")


class ReasoningStep(BaseModel):
    """A single step in the agent's reasoning process."""
    step: int = Field(description="Step number (1-5)")
    title: str = Field(description="Short title for the step")
    status: str = Field(description="Status: pending, in_progress, completed, skipped, failed")
    description: str = Field(description="What's happening in this step")
    duration_ms: Optional[float] = Field(default=None, description="Time taken for this step")
    details: Optional[dict] = Field(default=None, description="Step-specific details")


class ReviewDecision(BaseModel):
    """Complete submittal review decision - the main output schema."""

    # Core decision
    decision: DecisionType = Field(description="The review decision")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in decision (0-1)")
    reasoning: str = Field(description="Brief explanation of decision")

    # Detailed analysis
    detailed_comments: list[DetailedComment] = Field(default_factory=list)
    compliance_checks: dict[str, ComplianceCheck] = Field(
        description="Results of compliance checks by category"
    )

    # Vision analysis (if enabled)
    drawing_analysis: Optional[DrawingAnalysis] = Field(
        default=None,
        description="Analysis of drawings/images if vision was enabled"
    )

    # Actions required (for revise/reject)
    required_actions: list[str] = Field(
        default_factory=list,
        description="Required actions before resubmission"
    )

    # Citations for grounding
    citations: list[Citation] = Field(default_factory=list)

    # Metadata
    reviewed_at: datetime = Field(default_factory=datetime.now)
    llm_used: str = Field(default="claude-sonnet-4-20250514")
    submittal_type: str = Field(default="product_data")

    class Config:
        json_schema_extra = {
            "example": {
                "decision": "approved_as_noted",
                "confidence_score": 0.87,
                "reasoning": "Material meets specification requirements. Minor dimension clarification needed.",
                "detailed_comments": [
                    {
                        "category": "specification",
                        "comment": "Concrete mix design meets 4000 PSI requirement",
                        "citation": "[Source 1]"
                    }
                ],
                "compliance_checks": {
                    "specification": {"status": "pass", "notes": "Meets spec requirements"},
                    "code": {"status": "pass", "notes": "Fire rating compliant"},
                    "completeness": {"status": "partial", "notes": "Missing curing schedule"},
                    "quality": {"status": "pass", "notes": "Drawings are clear and legible"},
                    "coordination": {"status": "not_applicable", "notes": "No coordination issues identified"}
                },
                "required_actions": ["Provide curing schedule per Section 03 30 00"],
                "citations": [
                    {
                        "source_num": 1,
                        "file": "spec_div03.pdf",
                        "section": "03 30 00",
                        "quote": "Concrete shall achieve minimum 4000 PSI compressive strength"
                    }
                ]
            }
        }


class ReviewRequest(BaseModel):
    """Request model for submittal review API."""
    submittal_type: str = Field(default="product_data", description="Type of submittal")
    enable_vision: bool = Field(default=True, description="Enable vision analysis of drawings")
    additional_context: Optional[str] = Field(default=None, description="Additional context for review")


class ReviewResponse(BaseModel):
    """Response wrapper for the review API."""
    success: bool
    data: Optional[ReviewDecision] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Steps the agent took during reasoning"
    )
