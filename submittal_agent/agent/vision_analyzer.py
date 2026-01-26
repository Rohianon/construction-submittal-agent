"""Vision analysis for construction shop drawings using Claude Vision."""

import json
import logging
from typing import Optional

from submittal_agent.agent.llm_client import get_llm_client
from submittal_agent.agent.prompts import VISION_SYSTEM_PROMPT, VISION_USER_PROMPT
from submittal_agent.schemas.review_response import DrawingAnalysis
from submittal_agent.ingestion.image_extractor import ImageExtractionResult

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """
    Analyzes construction drawings and images using Claude Vision.

    Extracts:
    - Dimensions and measurements
    - Material specifications
    - Assembly details
    - Potential issues
    """

    def __init__(self):
        self.llm = get_llm_client()

    def analyze(
        self,
        images: ImageExtractionResult,
        max_images: int = 5,
    ) -> DrawingAnalysis:
        """
        Analyze extracted images from a submittal.

        Args:
            images: ImageExtractionResult from the image extractor
            max_images: Maximum images to analyze (cost control)

        Returns:
            DrawingAnalysis with extracted information
        """
        if not images.images:
            return DrawingAnalysis(
                images_analyzed=0,
                notes="No images found in the submittal document."
            )

        # Prepare images for vision API
        images_to_analyze = images.images[:max_images]
        image_data = [
            {
                "data": img.base64_data,
                "media_type": img.media_type,
            }
            for img in images_to_analyze
        ]

        try:
            response, provider = self.llm.complete_with_vision(
                text_prompt=VISION_USER_PROMPT,
                images_base64=image_data,
                system=VISION_SYSTEM_PROMPT,
            )

            # Parse JSON response
            analysis = self._parse_vision_response(response)
            analysis.images_analyzed = len(images_to_analyze)

            logger.info(f"Vision analysis completed using {provider}")
            return analysis

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return DrawingAnalysis(
                images_analyzed=len(images_to_analyze),
                notes=f"Vision analysis encountered an error: {str(e)}",
                potential_issues=["Vision analysis failed - manual review recommended"]
            )

    def _parse_vision_response(self, response: str) -> DrawingAnalysis:
        """Parse the JSON response from vision analysis."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response)

            return DrawingAnalysis(
                dimensions_found=data.get("dimensions_found", []),
                materials_identified=data.get("materials_identified", []),
                potential_issues=data.get("potential_issues", []),
                notes=data.get("notes", ""),
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse vision response as JSON: {e}")
            # Return with raw response as notes
            return DrawingAnalysis(
                notes=f"Analysis (unstructured): {response[:500]}"
            )


# Singleton
_vision_analyzer: Optional[VisionAnalyzer] = None


def get_vision_analyzer() -> VisionAnalyzer:
    """Get or create the vision analyzer singleton."""
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer
