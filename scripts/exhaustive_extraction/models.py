"""
Model selection, API wrappers, and retry logic for extraction.

Uses litellm for unified interface across all LLM providers (Claude, Gemini, OpenRouter).

Handles:
- Model selection based on segment size
- API calls with retry and exponential backoff
- Rate limiting and concurrency management
- JSON mode enforcement across providers
"""

import json
import os
import re
import time
from collections import deque
from threading import Lock, Semaphore
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import litellm
from litellm import completion

from conference_pipeline.model_governance.registry import load_registry
from conference_pipeline.provider_receipts import litellm_completion, record_provider_call

# Drop unsupported params automatically (e.g., temperature for GPT-5 models)
litellm.drop_params = True


class JSONExtractionError(Exception):
    """Raised when JSON cannot be extracted from model response."""
    def __init__(self, message: str, response_preview: str):
        super().__init__(message)
        self.response_preview = response_preview


class EmptyResponseError(Exception):
    """Raised when model returns empty response (e.g., GPT-5 reasoning token exhaustion)."""
    pass


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from model response text.

    Tries multiple strategies in order:
    1. Direct JSON parse (response is pure JSON)
    2. Extract from ```json ... ``` code block
    3. Extract from ``` ... ``` code block
    4. Find JSON object/array with regex
    5. Try to repair common JSON issues

    Args:
        response_text: Raw response text from model

    Returns:
        Parsed JSON as dict

    Raises:
        JSONExtractionError: If no valid JSON can be extracted
    """
    if not response_text or not response_text.strip():
        raise JSONExtractionError(
            "Empty response from model",
            response_preview="<empty>"
        )

    text = response_text.strip()

    # Strategy 1: Try direct parse (response is pure JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json ... ``` code block
    json_block_match = re.search(r'```json\s*([\s\S]*?)```', text)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract from ``` ... ``` code block (without json tag)
    code_block_match = re.search(r'```\s*([\s\S]*?)```', text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find JSON object with regex (handles text before/after JSON)
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 5: Find JSON array with regex
    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 6: Handle truncated JSON (missing closing braces)
    if '{' in text:
        json_start = text.find('{')
        partial_json = text[json_start:]
        open_braces = partial_json.count('{') - partial_json.count('}')
        open_brackets = partial_json.count('[') - partial_json.count(']')

        if open_braces > 0 or open_brackets > 0:
            repaired = partial_json + (']' * open_brackets) + ('}' * open_braces)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # All strategies failed
    preview = text[:500] + "..." if len(text) > 500 else text
    raise JSONExtractionError(
        f"Could not extract valid JSON from response. "
        f"Response length: {len(text)} chars. "
        f"Contains '{{': {'Yes' if '{' in text else 'No'}. "
        f"Contains '```': {'Yes' if '```' in text else 'No'}.",
        response_preview=preview
    )


# =============================================================================
# Model Configuration (litellm format)
# =============================================================================


def _litellm_model(task_id: str) -> str:
    """LiteLLM id (``<provider>/<model>``) of a governed extraction task.

    OpenRouter models are registered with their ``openrouter/`` prefix already.
    """
    policy = load_registry().get(task_id)
    if policy.model.startswith(f"{policy.provider}/"):
        return policy.model
    return f"{policy.provider}/{policy.model}"


# Every route is read from the model-governance registry by task id.
# GPT-5-mini is used as fallback for lower-risk cleanup/recovery.
GPT5_MINI = _litellm_model("exhaustive_extraction.gpt5_mini")

# GPT-5.2 - powerful model for complex tasks like quote extraction fallback
GPT5_2 = _litellm_model("exhaustive_extraction.gpt52_quote_fallback")

# Direct API models (litellm prefixes)
CLAUDE_SONNET = _litellm_model("exhaustive_extraction.claude_sonnet")
# The primary (<200K token) route. Its task id and these constant names keep
# the historical "haiku" label, but Claude Haiku is retired in the registry and
# the task resolves to Claude Sonnet.
CLAUDE_HAIKU = _litellm_model("exhaustive_extraction.claude_haiku")
GEMINI_PRO = _litellm_model("exhaustive_extraction.gemini_pro")

# OpenRouter models
OPENROUTER_SONNET = _litellm_model("exhaustive_extraction.openrouter_sonnet")
OPENROUTER_GEMINI = _litellm_model("exhaustive_extraction.openrouter_gemini")
OPENROUTER_GPT5_MINI = _litellm_model("exhaustive_extraction.openrouter_gpt5_mini")

# OpenRouter fallback for the primary route (same retired-Haiku note as above)
OPENROUTER_HAIKU = _litellm_model("exhaustive_extraction.openrouter_haiku")

# Local MLX default for ICLR content-fidelity recovery pilots.
LOCAL_MLX_QWEN35B_A3B = "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"
LOCAL_MLX_DEFAULT_MAX_OUTPUT = 8_000

# Mapping from direct API models to OpenRouter equivalents
OPENROUTER_FALLBACK_MAP = {
    GPT5_MINI: OPENROUTER_GPT5_MINI,
    CLAUDE_SONNET: OPENROUTER_SONNET,
    CLAUDE_HAIKU: OPENROUTER_HAIKU,
    GEMINI_PRO: OPENROUTER_GEMINI,
}

# Token limits for model selection
HAIKU_TOKEN_LIMIT = 200_000  # Claude Haiku 4.5 context window
GPT5_MINI_TOKEN_LIMIT = 400_000  # GPT-5-mini planning context window
SONNET_TOKEN_LIMIT = 50_000  # Use Sonnet for segments < 50K tokens
GEMINI_TOKEN_LIMIT = 1_000_000  # Use Gemini for segments < 1M tokens

# Max output tokens per model
HAIKU_MAX_OUTPUT = 64_000  # Claude Haiku 4.5 max output
DEFAULT_MAX_OUTPUT = 128_000  # Default for other models

# Rate limits
SONNET_RATE_LIMIT = 50
GEMINI_RATE_LIMIT = 30
MAX_WORKERS = 3

# Global throttle
GLOBAL_MAX_CONCURRENT = 3
GLOBAL_RPM_LIMIT = 45

_MLX_CACHE_LOCK = Lock()
_MLX_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}

# System prompt for JSON extraction
JSON_SYSTEM_PROMPT = (
    "You are a JSON extraction assistant. You MUST respond with valid JSON only. "
    "Do not include any text before or after the JSON object. "
    "Do not wrap the JSON in markdown code blocks. "
    "Output a single JSON object with an 'extractions' array."
)


class ModelSelector:
    """Select appropriate model based on segment characteristics."""

    def __init__(self):
        self.provider_override = os.getenv("EXHAUSTIVE_EXTRACTION_PROVIDER", "").strip().lower()
        self.local_mlx_model = (
            os.getenv("EXHAUSTIVE_EXTRACTION_MODEL", "").strip()
            or LOCAL_MLX_QWEN35B_A3B
        )
        self.openai_available = bool(os.getenv("OPENAI_API_KEY"))
        self.anthropic_available = bool(os.getenv("ANTHROPIC_API_KEY"))
        self.gemini_available = bool(os.getenv("GOOGLE_API_KEY"))
        self.openrouter_available = bool(os.getenv("OPENROUTER_API_KEY"))

    def select_model(self, token_count: int) -> str:
        """
        Select best model for given token count.

        Fallback chain: primary Claude route → Gemini → GPT-5-mini → OpenRouter

        The primary route (CLAUDE_HAIKU, governed task
        exhaustive_extraction.claude_haiku) handles segments under
        HAIKU_TOKEN_LIMIT; the registry decides which Claude model that is.

        Args:
            token_count: Estimated token count for segment

        Returns:
            Model ID to use (litellm format)
        """
        if self.provider_override in {"mlx", "local-mlx", "local_mlx"}:
            return self.local_mlx_model

        # Try the primary Claude route first
        if token_count < HAIKU_TOKEN_LIMIT and self.anthropic_available:
            return CLAUDE_HAIKU

        # Try Gemini for large segments (200K - 1M tokens)
        if token_count < GEMINI_TOKEN_LIMIT and self.gemini_available:
            return GEMINI_PRO

        # Guard: If segment exceeds Gemini limit, fail fast
        if token_count >= GEMINI_TOKEN_LIMIT:
            raise RuntimeError(
                f"Segment too large: {token_count:,} tokens exceeds maximum supported limit "
                f"({GEMINI_TOKEN_LIMIT:,} tokens). Reduce segment size."
            )

        # Try GPT-5-mini as fallback.
        if token_count < GPT5_MINI_TOKEN_LIMIT and self.openai_available:
            return GPT5_MINI

        # Try the primary Claude route via OpenRouter as fallback
        if token_count < HAIKU_TOKEN_LIMIT and self.openrouter_available:
            return OPENROUTER_HAIKU

        # Try OpenRouter GPT-5-mini for larger segments
        if token_count < GPT5_MINI_TOKEN_LIMIT and self.openrouter_available:
            return OPENROUTER_GPT5_MINI

        raise RuntimeError(
            "No available models - check API keys "
            "(ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY)"
        )


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses simple heuristic: ~1.3 tokens per word.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    words = len(text.split())
    return int(words * 1.3)


def _mlx_max_output_tokens() -> int:
    raw = os.getenv("EXHAUSTIVE_EXTRACTION_MAX_OUTPUT_TOKENS")
    if not raw:
        return LOCAL_MLX_DEFAULT_MAX_OUTPUT
    try:
        return max(1, int(raw))
    except ValueError:
        return LOCAL_MLX_DEFAULT_MAX_OUTPUT


def _load_mlx_model(model_id: str) -> tuple[Any, Any]:
    with _MLX_CACHE_LOCK:
        cached = _MLX_MODEL_CACHE.get(model_id)
        if cached is not None:
            return cached

        from mlx_lm import load  # type: ignore[import-not-found]

        loaded = load(model_id)
        _MLX_MODEL_CACHE[model_id] = loaded
        return loaded


def _token_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return estimate_tokens(text)


def _apply_mlx_chat_template(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": JSON_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"{JSON_SYSTEM_PROMPT}\n\n{prompt}"


def _strip_qwen_thinking(response_text: str) -> str:
    if "</think>" in response_text:
        return response_text.split("</think>", 1)[1].strip()
    return response_text


def _call_mlx(
    prompt: str,
    model: str,
    max_tokens: int | None = None,
) -> Tuple[Dict[str, Any], int, int]:
    """Call local MLX generation with provider receipts and JSON parsing."""

    if max_tokens is None:
        max_tokens = _mlx_max_output_tokens()

    def invoke() -> Any:
        from mlx_lm import generate  # type: ignore[import-not-found]

        model_obj, tokenizer = _load_mlx_model(model)
        formatted_prompt = _apply_mlx_chat_template(tokenizer, prompt)
        response_text = generate(
            model_obj,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        response_text = _strip_qwen_thinking(str(response_text or ""))
        input_tokens = _token_count(tokenizer, formatted_prompt)
        output_tokens = _token_count(tokenizer, response_text)
        usage = SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        message = SimpleNamespace(content=response_text)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(id=None, choices=[choice], usage=usage)

    response = record_provider_call(
        invoke,
        provider="local",
        model=model,
        operation="mlx_lm.generate",
        caller=__file__,
        request_kwargs={
            "model": model,
            "max_tokens": max_tokens,
            "enable_thinking": False,
            "prompt": prompt,
        },
    )

    response_text = response.choices[0].message.content
    if not response_text or not response_text.strip():
        raise EmptyResponseError(
            f"Model returned empty response (model={model}, provider=mlx)."
        )
    parsed = extract_json_from_response(response_text)
    usage = response.usage
    return parsed, usage.prompt_tokens, usage.completion_tokens


def _call_litellm(
    prompt: str,
    model: str,
    max_tokens: int = None
) -> Tuple[Dict[str, Any], int, int]:
    """
    Call LLM via litellm with JSON mode and retry logic.

    Args:
        prompt: Extraction prompt
        model: Model ID (litellm format, e.g., "anthropic/claude-sonnet-4-5")
        max_tokens: Maximum output tokens (auto-detected from model if None)

    Returns:
        Tuple of (parsed_json, input_tokens, output_tokens)
    """
    # Auto-detect max_tokens based on model
    if max_tokens is None:
        if "haiku" in model.lower():
            max_tokens = HAIKU_MAX_OUTPUT
        else:
            max_tokens = DEFAULT_MAX_OUTPUT

    max_retries = 5
    initial_delay = 5.0
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            # litellm handles JSON mode via response_format for supported providers
            response = litellm_completion(
                completion,
                caller=__file__,
                model=model,
                messages=[
                    {"role": "system", "content": JSON_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            # Extract response text
            response_text = response.choices[0].message.content

            # Check for empty response (GPT-5 reasoning token exhaustion)
            if not response_text or not response_text.strip():
                finish_reason = response.choices[0].finish_reason
                output_tokens = response.usage.completion_tokens if response.usage else 0
                # Don't retry empty responses - they indicate model limitation
                raise EmptyResponseError(
                    f"Model returned empty response (finish_reason={finish_reason}, "
                    f"output_tokens={output_tokens}). This model may not be suitable for this prompt."
                )

            # Parse JSON using robust extraction
            parsed = extract_json_from_response(response_text)

            # Get token counts
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return parsed, input_tokens, output_tokens

        except EmptyResponseError:
            # Don't retry empty responses - immediately propagate for model fallback
            raise

        except Exception as e:
            last_exception = e
            is_rate_limit = "rate" in str(e).lower()
            is_json_error = isinstance(e, JSONExtractionError)

            if attempt < max_retries - 1:
                if is_rate_limit:
                    error_type = "Rate limit"
                elif is_json_error:
                    error_type = "JSONExtractionError"
                else:
                    error_type = type(e).__name__

                print(f"      API error (attempt {attempt + 1}/{max_retries}): {error_type}")

                if is_json_error:
                    preview = e.response_preview[:200] + "..." if len(e.response_preview) > 200 else e.response_preview
                    print(f"      Response preview: {preview}")

                print(f"      Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"      API error after {max_retries} attempts: {e}")
                if is_json_error:
                    print(f"      Final response preview: {e.response_preview}")

    raise last_exception


class ExtractionRunner:
    """High-level runner for extraction with model selection and failover."""

    _semaphore = Semaphore(GLOBAL_MAX_CONCURRENT)
    _lock = Lock()
    _call_timestamps = deque()

    def __init__(self):
        self.selector = ModelSelector()

    @classmethod
    def _throttle(cls):
        """Enforce max concurrent calls and RPM limit."""
        cls._semaphore.acquire()
        with cls._lock:
            now = time.time()
            while cls._call_timestamps and now - cls._call_timestamps[0] > 60:
                cls._call_timestamps.popleft()
            if len(cls._call_timestamps) >= GLOBAL_RPM_LIMIT:
                sleep_for = 60 - (now - cls._call_timestamps[0]) + 0.01
                time.sleep(max(0.01, sleep_for))
            cls._call_timestamps.append(time.time())

    @classmethod
    def _release(cls):
        cls._semaphore.release()

    def run_extraction(
        self,
        prompt: str,
        estimated_tokens: Optional[int] = None
    ) -> Tuple[Dict[str, Any], str, int, int]:
        """
        Run extraction with automatic model selection and dynamic failover.

        If primary model fails after retries, automatically tries OpenRouter equivalent.

        Args:
            prompt: Extraction prompt
            estimated_tokens: Pre-computed token estimate (or None to compute)

        Returns:
            Tuple of (parsed_json, model_used, input_tokens, output_tokens)
        """
        self._throttle()
        try:
            # Estimate tokens if not provided
            if estimated_tokens is None:
                estimated_tokens = estimate_tokens(prompt)

            # Select model
            model = self.selector.select_model(estimated_tokens)

            # Log model selection for monitoring (only for non-default models)
            if model == CLAUDE_HAIKU:
                pass  # Default model, no need to log
            elif self.selector.provider_override in {"mlx", "local-mlx", "local_mlx"}:
                print(f"      📊 Using local MLX ({model})")
            elif model == GEMINI_PRO:
                print(f"      📊 Using GEMINI (large context: {estimated_tokens:,} tokens)")
            elif model == GPT5_MINI:
                print("      📊 Using GPT-5-mini (Anthropic unavailable)")
            elif "openrouter" in model:
                model_name = model.split("/")[-1]
                print(f"      📊 Using OPENROUTER ({model_name})")

            # Try primary model
            try:
                if self.selector.provider_override in {"mlx", "local-mlx", "local_mlx"}:
                    parsed, input_tok, output_tok = _call_mlx(prompt, model)
                else:
                    parsed, input_tok, output_tok = _call_litellm(prompt, model)
                return parsed, model, input_tok, output_tok

            except EmptyResponseError as empty_error:
                # Model returned empty response - try different model family
                print(f"      ⚠️  {model} returned empty response")

                # Try Gemini first (best for large contexts)
                if self.selector.gemini_available and model != GEMINI_PRO:
                    print("      🔄 Falling back to Gemini...")
                    try:
                        parsed, input_tok, output_tok = _call_litellm(prompt, GEMINI_PRO)
                        print("      ✅ Gemini fallback succeeded")
                        return parsed, GEMINI_PRO, input_tok, output_tok
                    except Exception as gemini_error:
                        print(f"      ❌ Gemini fallback failed: {type(gemini_error).__name__}")

                # Try GPT-5-mini (different model family)
                if self.selector.openai_available and estimated_tokens < GPT5_MINI_TOKEN_LIMIT and model != GPT5_MINI:
                    print("      🔄 Falling back to GPT-5-mini...")
                    try:
                        parsed, input_tok, output_tok = _call_litellm(prompt, GPT5_MINI)
                        print("      ✅ GPT-5-mini fallback succeeded")
                        return parsed, GPT5_MINI, input_tok, output_tok
                    except Exception as gpt_error:
                        print(f"      ❌ GPT-5-mini fallback failed: {type(gpt_error).__name__}")

                # All fallbacks failed
                raise RuntimeError(
                    f"Empty response from {model} and all fallbacks failed"
                ) from empty_error

            except Exception as primary_error:
                # Check if OpenRouter fallback is available
                fallback_model = OPENROUTER_FALLBACK_MAP.get(model)

                if fallback_model and self.selector.openrouter_available:
                    print(f"      ⚠️  Primary model failed: {type(primary_error).__name__}")
                    print(f"      🔄 Trying OpenRouter fallback: {fallback_model}")

                    try:
                        parsed, input_tok, output_tok = _call_litellm(prompt, fallback_model)
                        print(f"      ✅ OpenRouter fallback succeeded")
                        return parsed, fallback_model, input_tok, output_tok

                    except Exception as fallback_error:
                        print(f"      ❌ OpenRouter fallback also failed: {type(fallback_error).__name__}")
                        raise RuntimeError(
                            f"Both primary ({model}) and fallback ({fallback_model}) failed. "
                            f"Primary: {primary_error}, Fallback: {fallback_error}"
                        ) from primary_error
                else:
                    raise

        finally:
            self._release()


def call_gpt52_for_quote(
    source_text: str,
    start_anchor: str,
    end_anchor: str,
    summary: str,
    topic_name: Optional[str] = None,
    timestamp_start_sec: Optional[float] = None,
    timestamp_end_sec: Optional[float] = None,
    relevance_score: Optional[float] = None,
    max_source_chars: int = 15000
) -> Optional[str]:
    """
    Use GPT-5.2 to directly find and extract a quote from source text.

    Instead of asking the LLM to return corrected anchors, this function
    asks GPT-5.2 to find and return the actual verbatim quote text.

    Args:
        source_text: Full transcript/abstract text
        start_anchor: Approximate start anchor (may have transcription errors)
        end_anchor: Approximate end anchor (may have transcription errors)
        summary: Summary describing the quote's content
        topic_name: Research topic name for semantic context (e.g., "LLM Scaling Laws")
        timestamp_start_sec: Start time in seconds to narrow search window
        timestamp_end_sec: End time in seconds to narrow search window
        relevance_score: Confidence score (0-1) from extraction model
        max_source_chars: Maximum source text to include in prompt

    Returns:
        Verbatim quote text if found, None if not found or error
    """
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("      ⚠️  GPT-5.2 unavailable (no OPENAI_API_KEY)")
        return None

    # Truncate source text if needed
    source_preview = source_text[:max_source_chars]
    truncated = len(source_text) > max_source_chars

    # Build contextual information
    context_lines = []
    if topic_name:
        context_lines.append(f"RESEARCH TOPIC: {topic_name}")
    if timestamp_start_sec is not None and timestamp_end_sec is not None:
        duration = timestamp_end_sec - timestamp_start_sec
        start_mm_ss = f"{int(timestamp_start_sec//60)}:{int(timestamp_start_sec%60):02d}"
        end_mm_ss = f"{int(timestamp_end_sec//60)}:{int(timestamp_end_sec%60):02d}"
        context_lines.append(f"TIME RANGE IN TRANSCRIPT: {start_mm_ss} - {end_mm_ss} ({duration:.0f}s duration)")
        context_lines.append(f"  → Focus your search on this time window in the transcript")
    if relevance_score is not None:
        context_lines.append(f"EXTRACTION CONFIDENCE: {relevance_score:.2f}/1.0 (how relevant this passage is)")

    context_section = "\n".join(context_lines) if context_lines else ""

    prompt = f"""Extract the EXACT verbatim quote from the source text based on the context below.

SOURCE TEXT:
{source_preview}{"..." if truncated else ""}

=== PRIMARY SEARCH CRITERIA (Use these to find the right passage) ===
{context_section}

CONTENT DESCRIPTION: {summary}

=== APPROXIMATE ANCHORS (Hints only - may have transcription errors!) ===
These are CLUES, not exact text. The actual quote may have different wording:
- Start words (approximate): "{start_anchor}"
- End words (approximate): "{end_anchor}"

⚠️  IMPORTANT: The anchors above are NOT reliable - they may have transcription errors,
missing words, or different punctuation. Use them only as rough hints.

=== YOUR TASK ===
STEP 1: Find passage(s) in the source text that match:
   • The TOPIC: {topic_name if topic_name else 'general content'}
   • The CONTENT DESCRIPTION above
{f"   • The TIME RANGE: {start_mm_ss} - {end_mm_ss}" if timestamp_start_sec is not None and timestamp_end_sec is not None else ""}

STEP 2: Among matching passages, find the one that ALSO has words SIMILAR to the approximate anchors
   • Don't require exact anchor matches - they may have errors!
   • Look for semantic similarity, not word-for-word matching

STEP 3: Return the COMPLETE QUOTE, not just the anchor words
   • Include the full sentence or paragraph that contains the relevant content
   • Copy it EXACTLY as it appears in the source text
   • The quote should be self-contained and make sense on its own

Return JSON:
{{"verbatim_quote": "exact text copied from source", "found": true, "reasoning": "brief explanation of why this passage matches"}}

If no matching passage exists, return:
{{"verbatim_quote": null, "found": false, "reasoning": "why no match was found"}}"""

    try:
        max_retries = 3
        delay = 2.0

        for attempt in range(max_retries):
            try:
                response = litellm_completion(
                    completion,
                    caller=__file__,
                    model=GPT5_2,
                    messages=[
                        {"role": "system", "content": "You are a precise text extraction assistant. Use semantic understanding and context to find relevant passages, then return exact verbatim quotes from source text. Prioritize topic relevance and time range over exact anchor matching."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8000,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                response_text = response.choices[0].message.content
                if not response_text or not response_text.strip():
                    return None

                parsed = extract_json_from_response(response_text)

                if parsed.get("found") and parsed.get("verbatim_quote"):
                    quote = parsed["verbatim_quote"]
                    reasoning = parsed.get("reasoning", "")

                    # Log the model's reasoning
                    if reasoning:
                        print(f"      💡 GPT-5.2 reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")

                    # Validate quote is actually in source text (normalized comparison)
                    quote_norm = " ".join(quote.split()).lower()
                    source_norm = " ".join(source_text.split()).lower()
                    if quote_norm in source_norm:
                        return quote
                    else:
                        print(f"      ⚠️  GPT-5.2 returned quote not found in source")
                        if reasoning:
                            print(f"         Reasoning was: {reasoning}")
                        return None
                elif not parsed.get("found"):
                    reasoning = parsed.get("reasoning", "no reason provided")
                    print(f"      ℹ️  GPT-5.2 no match: {reasoning[:80]}{'...' if len(reasoning) > 80 else ''}")
                    return None

                return None

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      GPT-5.2 attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    except Exception as e:
        print(f"      ❌ GPT-5.2 quote extraction failed: {e}")
        return None
