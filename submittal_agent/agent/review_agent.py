"""Main review agent orchestrating the submittal review process."""

import json
import logging
import time
from typing import Optional
from pathlib import Path

from submittal_agent.agent.llm_client import get_llm_client, LLMClient
from submittal_agent.agent.vision_analyzer import get_vision_analyzer, VisionAnalyzer
from submittal_agent.agent.prompts import (
    REVIEW_SYSTEM_PROMPT,
    build_review_prompt,
    build_spec_context,
)
from submittal_agent.retrieval.retriever import get_retriever, SpecificationRetriever
from submittal_agent.ingestion.pdf_parser import PDFParser, ParsedDocument
from submittal_agent.ingestion.image_extractor import ImageExtractor, ImageExtractionResult
from submittal_agent.schemas.review_response import (
    ReviewDecision,
    DecisionType,
    ComplianceCheck,
    ComplianceStatus,
    Citation,
    DetailedComment,
    DrawingAnalysis,
)

logger = logging.getLogger(__name__)


class SubmittalReviewAgent:
    """
    Main agent for reviewing construction submittals.

    Orchestrates:
    1. PDF parsing and text extraction
    2. Image extraction for vision analysis
    3. Specification retrieval from knowledge base
    4. LLM-based review with citations
    5. Structured output generation
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        vision_analyzer: Optional[VisionAnalyzer] = None,
        retriever: Optional[SpecificationRetriever] = None,
    ):
        self.llm = llm_client or get_llm_client()
        self.vision = vision_analyzer or get_vision_analyzer()
        self.retriever = retriever or get_retriever()
        self.pdf_parser = PDFParser()
        self.image_extractor = ImageExtractor()

    def review(
        self,
        pdf_content: bytes,
        filename: str = "submittal.pdf",
        submittal_type: str = "product_data",
        enable_vision: bool = True,
        additional_context: str = "",
        model: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_k: int = 8,
    ) -> ReviewDecision:
        """
        Perform a complete submittal review.

        Args:
            pdf_content: PDF file content as bytes
            filename: Name of the uploaded file
            submittal_type: Type of submittal (product_data, shop_drawing, etc.)
            enable_vision: Whether to analyze images/drawings
            additional_context: Additional context from user
            model: LLM model to use (auto, claude, gpt-4)
            temperature: Temperature for LLM generation
            max_tokens: Maximum response tokens
            top_k: Number of specification chunks to retrieve

        Returns:
            ReviewDecision with complete review results
        """
        start_time = time.time()

        # Step 1: Parse PDF for text
        logger.info(f"Parsing PDF: {filename}")
        parsed_doc = self.pdf_parser.parse_bytes(pdf_content, filename)
        submittal_text = parsed_doc.get_full_text()

        # Step 2: Extract and analyze images (if enabled)
        drawing_analysis = None
        if enable_vision:
            logger.info("Extracting images for vision analysis")
            images = self.image_extractor.extract_from_bytes(pdf_content, filename)
            if images.images:
                logger.info(f"Analyzing {len(images.images)} images")
                drawing_analysis = self.vision.analyze(images)

        # Step 3: Retrieve relevant specifications
        logger.info(f"Retrieving relevant specifications (top_k={top_k})")
        query = self._build_retrieval_query(submittal_text, submittal_type)
        retrieved_chunks = self.retriever.retrieve(query, submittal_type, top_k=top_k)
        spec_context = build_spec_context(retrieved_chunks)

        # Step 4: Generate review decision
        logger.info(f"Generating review decision (model={model}, temp={temperature})")
        review_prompt = build_review_prompt(
            submittal_content=submittal_text,
            submittal_type=submittal_type,
            spec_context=spec_context,
            additional_context=additional_context,
        )

        response, provider = self.llm.complete(
            messages=[{"role": "user", "content": review_prompt}],
            system=REVIEW_SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Step 5: Parse and validate response
        decision = self._parse_review_response(response, provider, submittal_type)

        # Add drawing analysis if available
        if drawing_analysis:
            decision.drawing_analysis = drawing_analysis

        # Validate and enhance citations
        decision.citations = self._validate_citations(decision.citations, retrieved_chunks)

        elapsed = time.time() - start_time
        logger.info(f"Review completed in {elapsed:.2f}s using {provider}")

        return decision

    def _build_retrieval_query(self, text: str, submittal_type: str) -> str:
        """Build a focused query for specification retrieval."""
        # Use first 1000 chars as query (contains key info)
        summary = text[:1000].replace("\n", " ").strip()
        return f"{submittal_type} specifications requirements: {summary}"

    def _parse_review_response(
        self,
        response: str,
        provider: str,
        submittal_type: str,
    ) -> ReviewDecision:
        """Parse the LLM response into a ReviewDecision."""
        try:
            # Clean response and parse JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response)

            # Build compliance checks
            compliance_checks = {}
            for category, check_data in data.get("compliance_checks", {}).items():
                compliance_checks[category] = ComplianceCheck(
                    status=ComplianceStatus(check_data.get("status", "not_applicable")),
                    notes=check_data.get("notes", ""),
                )

            # Build detailed comments
            detailed_comments = [
                DetailedComment(
                    category=c.get("category", ""),
                    comment=c.get("comment", ""),
                    citation=c.get("citation"),
                )
                for c in data.get("detailed_comments", [])
            ]

            # Build citations
            citations = [
                Citation(
                    source_num=c.get("source_num", 0),
                    file=c.get("file", ""),
                    section=c.get("section"),
                    page=c.get("page"),
                    quote=c.get("quote", ""),
                )
                for c in data.get("citations", [])
            ]

            return ReviewDecision(
                decision=DecisionType(data.get("decision", "revise_and_resubmit")),
                confidence_score=float(data.get("confidence_score", 0.5)),
                reasoning=data.get("reasoning", "Review completed."),
                detailed_comments=detailed_comments,
                compliance_checks=compliance_checks,
                required_actions=data.get("required_actions", []),
                citations=citations,
                llm_used=provider,
                submittal_type=submittal_type,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse review response: {e}")
            # Return a safe fallback
            return self._create_fallback_decision(response, provider, submittal_type)

    def _create_fallback_decision(
        self,
        raw_response: str,
        provider: str,
        submittal_type: str,
    ) -> ReviewDecision:
        """Create a fallback decision when parsing fails."""
        return ReviewDecision(
            decision=DecisionType.REVISE_AND_RESUBMIT,
            confidence_score=0.3,
            reasoning=f"Review analysis completed but response parsing failed. Manual review recommended. Raw analysis: {raw_response[:500]}...",
            compliance_checks={
                "specification": ComplianceCheck(status=ComplianceStatus.PARTIAL, notes="Automated review incomplete"),
                "code": ComplianceCheck(status=ComplianceStatus.NOT_APPLICABLE, notes="Requires manual review"),
                "completeness": ComplianceCheck(status=ComplianceStatus.PARTIAL, notes="Requires manual review"),
                "quality": ComplianceCheck(status=ComplianceStatus.PARTIAL, notes="Requires manual review"),
                "coordination": ComplianceCheck(status=ComplianceStatus.NOT_APPLICABLE, notes="Requires manual review"),
            },
            required_actions=["Manual review recommended due to processing error"],
            llm_used=provider,
            submittal_type=submittal_type,
        )

    def _validate_citations(
        self,
        citations: list[Citation],
        retrieved_chunks: list[dict],
    ) -> list[Citation]:
        """Validate citations against actually retrieved chunks."""
        # Build lookup of retrieved chunks
        chunk_lookup = {
            chunk.get("source_num", i): chunk
            for i, chunk in enumerate(retrieved_chunks, 1)
        }

        validated = []
        for citation in citations:
            if citation.source_num in chunk_lookup:
                chunk = chunk_lookup[citation.source_num]
                # Enhance citation with actual chunk info
                validated.append(Citation(
                    source_num=citation.source_num,
                    file=chunk.get("file", citation.file),
                    section=citation.section,
                    page=chunk.get("page", citation.page),
                    quote=citation.quote or chunk.get("content", "")[:200],
                ))
            else:
                # Keep citation but mark as unverified
                validated.append(citation)

        return validated


# Singleton
_review_agent: Optional[SubmittalReviewAgent] = None


def get_review_agent() -> SubmittalReviewAgent:
    """Get or create the review agent singleton."""
    global _review_agent
    if _review_agent is None:
        _review_agent = SubmittalReviewAgent()
    return _review_agent
