"""Prompts for audio transcription using LLMs.

These prompts are used with multimodal LLMs (like Claude) to transcribe
audio content from conference presentations.
"""

from textwrap import dedent


TRANSCRIPTION_PROMPT = dedent("""
    You are an expert transcriptionist. Listen to the provided audio and
    transcribe it accurately.

    Instructions:
    1. Transcribe all spoken words verbatim
    2. Include speaker labels if multiple speakers are detected
       (Speaker 1:, Speaker 2:, etc.)
    3. Note any significant non-speech sounds in brackets
       [applause], [laughter], etc.
    4. Preserve the natural flow of speech including filler words
    5. Use proper punctuation and paragraph breaks for readability

    Transcribe the audio now:
""").strip()


TRANSCRIPTION_WITH_CONTEXT_PROMPT = dedent("""
    You are an expert transcriptionist. Listen to the provided audio and
    transcribe it accurately.

    Context about this audio:
    {context}

    Instructions:
    1. Transcribe all spoken words verbatim
    2. Include speaker labels if multiple speakers are detected
       (Speaker 1:, Speaker 2:, etc.)
    3. Note any significant non-speech sounds in brackets
       [applause], [laughter], etc.
    4. Preserve the natural flow of speech including filler words
    5. Use proper punctuation and paragraph breaks for readability
    6. Use the context to help with technical terms and proper nouns

    Transcribe the audio now:
""").strip()


def format_transcription_prompt_with_context(context: str) -> str:
    """Format the transcription prompt with context.

    Args:
        context: Additional context about the audio (e.g., presentation
                 title, topic, speaker names).

    Returns:
        Formatted prompt string ready for LLM.
    """
    return TRANSCRIPTION_WITH_CONTEXT_PROMPT.format(context=context)
