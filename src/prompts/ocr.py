"""Prompts for OCR using vision-capable LLMs.

These prompts are used with GPT-4o or Claude to extract text from
slide images, particularly focusing on academic references.
"""

from textwrap import dedent


OCR_EXTRACTION_PROMPT = dedent("""
    Extract all text from this slide image. Include any URLs, DOIs,
    arXiv IDs, paper titles, author names, and references you can find.
    Return only the extracted text, no commentary.
""").strip()


# JSON schema for slide OCR output
SLIDE_OCR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "slide_text": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Slide title/header"},
                "bullet_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Bullet points preserving hierarchy (use indentation for sub-bullets)"
                },
                "equations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Mathematical equations in LaTeX notation"
                },
                "other_text": {"type": "string", "description": "Any other text on the slide"},
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs, DOIs, arXiv IDs found on slide"
                }
            },
            "required": ["title"]
        },
        "visual_elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Type: chart, diagram, figure, image, table"},
                    "description": {"type": "string", "description": "Detailed description of the visual"},
                    "data": {"type": "string", "description": "Data shown (axes, values, trends)"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels, legends, annotations"
                    }
                },
                "required": ["type", "description"]
            }
        },
        "key_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "explanation": {"type": "string"}
                },
                "required": ["concept"]
            },
            "description": "Main concepts presented on the slide"
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Paper title"},
                    "authors": {"type": "string", "description": "Author names"},
                    "venue": {"type": "string", "description": "Conference/journal"},
                    "year": {"type": "string", "description": "Publication year"},
                    "url": {"type": "string", "description": "URL, DOI, or arXiv ID"}
                }
            },
            "description": "Academic references visible on slide"
        }
    },
    "required": ["slide_text"]
}


# Context-aware slide OCR prompt for extracting structured content (JSON output)
SLIDE_OCR_PROMPT = dedent("""
    You are analyzing a slide from an academic presentation.

    **Presentation Title:** {title}

    **Transcript context (what the speaker is saying during this slide):**
    {transcript_context}

    ---

    Analyze this slide image and extract structured information. Return your response as a JSON object with the following structure:

    {{
      "slide_text": {{
        "title": "The slide title/header",
        "bullet_points": ["bullet 1", "  - sub-bullet", "bullet 2"],
        "equations": ["LaTeX equation 1", "LaTeX equation 2"],
        "other_text": "Any other text on the slide",
        "urls": ["any URLs, DOIs, arXiv IDs"]
      }},
      "visual_elements": [
        {{
          "type": "chart|diagram|figure|image|table",
          "description": "Detailed description of what the visual shows",
          "data": "Data being presented (axes labels, trends, values)",
          "labels": ["label1", "label2"]
        }}
      ],
      "key_concepts": [
        {{
          "concept": "Main concept name",
          "explanation": "Brief explanation based on slide and transcript context"
        }}
      ],
      "references": [
        {{
          "title": "Paper title",
          "authors": "Author names",
          "venue": "Conference/Journal",
          "year": "2024",
          "url": "DOI or arXiv ID"
        }}
      ]
    }}

    Important:
    - Use LaTeX notation for equations (e.g., "\\\\alpha", "\\\\frac{{a}}{{b}}")
    - For bullet point hierarchy, use indentation like "  - " for sub-bullets
    - If a section has no content, use an empty array [] or null
    - Only include references that are actually visible on the slide
    - Be thorough in describing visual elements - charts, diagrams, figures

    Return ONLY the JSON object, no other text.
""").strip()


# Simpler prompt for slides without transcript context (JSON output)
SLIDE_OCR_SIMPLE_PROMPT = dedent("""
    You are analyzing a slide from an academic presentation titled: "{title}"

    Analyze this slide image and extract structured information. Return your response as a JSON object with the following structure:

    {{
      "slide_text": {{
        "title": "The slide title/header",
        "bullet_points": ["bullet 1", "  - sub-bullet", "bullet 2"],
        "equations": ["LaTeX equation 1", "LaTeX equation 2"],
        "other_text": "Any other text on the slide",
        "urls": ["any URLs, DOIs, arXiv IDs"]
      }},
      "visual_elements": [
        {{
          "type": "chart|diagram|figure|image|table",
          "description": "Detailed description of what the visual shows",
          "data": "Data being presented (axes labels, trends, values)",
          "labels": ["label1", "label2"]
        }}
      ],
      "key_concepts": [
        {{
          "concept": "Main concept name",
          "explanation": "Brief explanation"
        }}
      ],
      "references": [
        {{
          "title": "Paper title",
          "authors": "Author names",
          "venue": "Conference/Journal",
          "year": "2024",
          "url": "DOI or arXiv ID"
        }}
      ]
    }}

    Important:
    - Use LaTeX notation for equations (e.g., "\\\\alpha", "\\\\frac{{a}}{{b}}")
    - For bullet point hierarchy, use indentation like "  - " for sub-bullets
    - If a section has no content, use an empty array [] or null
    - Only include references that are actually visible on the slide
    - Be thorough in describing visual elements - charts, diagrams, figures

    Return ONLY the JSON object, no other text.
""").strip()


def format_slide_ocr_prompt(
    title: str,
    transcript_context: str | None = None,
) -> str:
    """Format the slide OCR prompt with context.

    Args:
        title: The presentation title.
        transcript_context: Optional transcript text around the slide timestamp.

    Returns:
        Formatted prompt string.
    """
    if transcript_context and transcript_context.strip():
        return SLIDE_OCR_PROMPT.format(
            title=title,
            transcript_context=transcript_context,
        )
    else:
        return SLIDE_OCR_SIMPLE_PROMPT.format(title=title)
