"""Construction-specific prompts for submittal review."""

REVIEW_SYSTEM_PROMPT = """You are an expert construction submittal reviewer with 20+ years of experience in architectural and engineering review.

Your role is to review submitted documents (shop drawings, product data, material submittals) against project specifications and building codes to produce professional approval decisions.

CRITICAL RULES:
1. ONLY base your decision on the provided specifications and context
2. ALWAYS cite specific sources using [Source N] notation
3. If information is missing or unclear, note it as a deficiency
4. Be specific about what needs correction for "Revise and Resubmit" decisions
5. Follow AIA/EJCDC industry standards for review language

DECISION DEFINITIONS (per AIA A201-2017):
- APPROVED: Fully complies with contract documents, no changes needed
- APPROVED AS NOTED: Minor deviations acceptable, contractor must comply with notes
- REVISE AND RESUBMIT: Significant issues requiring revision before approval
- REJECTED: Fundamental non-compliance, cannot be approved

COMPLIANCE CATEGORIES:
1. SPECIFICATION: Materials, dimensions, performance vs. spec requirements
2. CODE: Fire rating, structural, accessibility, building code compliance
3. COMPLETENESS: Required information, calculations, certifications present
4. QUALITY: Legibility, detail adequacy, accuracy of information
5. COORDINATION: Conflicts with other trades, systems, or documents"""


REVIEW_USER_PROMPT = """Review the following construction submittal against the provided specifications.

SUBMITTAL INFORMATION:
Type: {submittal_type}
Content:
{submittal_content}

RELEVANT SPECIFICATIONS:
{spec_context}

{additional_context}

Provide your review decision in the following JSON format:
{{
    "decision": "approved" | "approved_as_noted" | "revise_and_resubmit" | "rejected",
    "confidence_score": 0.0-1.0,
    "reasoning": "Brief 2-3 sentence explanation of the decision",
    "detailed_comments": [
        {{"category": "specification|code|completeness|quality|coordination", "comment": "specific comment", "citation": "[Source N]"}}
    ],
    "compliance_checks": {{
        "specification": {{"status": "pass|fail|partial", "notes": "explanation"}},
        "code": {{"status": "pass|fail|partial|not_applicable", "notes": "explanation"}},
        "completeness": {{"status": "pass|fail|partial", "notes": "explanation"}},
        "quality": {{"status": "pass|fail|partial", "notes": "explanation"}},
        "coordination": {{"status": "pass|fail|partial|not_applicable", "notes": "explanation"}}
    }},
    "required_actions": ["Action 1 if revise/reject", "Action 2"],
    "citations": [
        {{"source_num": 1, "file": "filename.pdf", "section": "section number", "quote": "relevant quote"}}
    ]
}}

IMPORTANT: Return ONLY valid JSON, no additional text."""


VISION_SYSTEM_PROMPT = """You are an expert at analyzing construction shop drawings and technical documents.

Your task is to extract key information from drawings and images, including:
- Dimensions and measurements
- Material specifications
- Assembly details and sequences
- Potential issues or conflicts

Be precise and technical in your analysis."""


VISION_USER_PROMPT = """Analyze the following construction drawing/image and extract:

1. DIMENSIONS: List all visible dimensions and measurements
2. MATERIALS: Identify any materials, grades, or specifications shown
3. DETAILS: Note assembly details, connection types, or sequences
4. ISSUES: Flag any potential issues, unclear details, or conflicts

Provide your analysis in JSON format:
{{
    "dimensions_found": ["dimension 1", "dimension 2"],
    "materials_identified": ["material 1", "material 2"],
    "details_noted": ["detail 1", "detail 2"],
    "potential_issues": ["issue 1", "issue 2"],
    "notes": "General observations about the drawing"
}}

Return ONLY valid JSON."""


def build_spec_context(retrieved_chunks: list[dict]) -> str:
    """Build numbered context from retrieved specification chunks."""
    if not retrieved_chunks:
        return "No relevant specifications found in the knowledge base."

    parts = []
    for chunk in retrieved_chunks:
        source_num = chunk.get("source_num", 0)
        file_name = chunk.get("file", "unknown")
        content = chunk.get("content", "")
        page = chunk.get("page", "")
        score = chunk.get("score", 0)

        header = f"[Source {source_num}] {file_name}"
        if page:
            header += f" (Page {page})"
        header += f" [Relevance: {score:.2f}]"

        parts.append(f"{header}\n{content}\n---")

    return "\n\n".join(parts)


def build_review_prompt(
    submittal_content: str,
    submittal_type: str,
    spec_context: str,
    additional_context: str = "",
) -> str:
    """Build the complete review prompt."""
    context_section = ""
    if additional_context:
        context_section = f"\nADDITIONAL CONTEXT:\n{additional_context}"

    return REVIEW_USER_PROMPT.format(
        submittal_type=submittal_type,
        submittal_content=submittal_content[:12000],  # Limit content length
        spec_context=spec_context,
        additional_context=context_section,
    )
