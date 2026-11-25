# Guardrails System: Complete Technical Explanation

## Overview

The guardrails system provides topic-specific quality control for content extraction. It works in two phases: **guidance** (prompts) and **enforcement** (QA validation).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  conferences/neurips2025/analysis/topic_guardrails.json         │
│  {                                                               │
│    "01": {                                                       │
│      "required_cues": ["benchmark", "framework"],               │
│      "min_summary_len": 6,                                      │
│      "allow_generic": false                                     │
│    }                                                             │
│  }                                                               │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   │ load_topic_guardrails()
                   ↓
┌──────────────────────────────────────────────────────────────────┐
│  run_exhaustive_extraction.py                                    │
│  - Loads guardrails at startup                                   │
│  - Passes to PresentationProcessor                               │
│  - Passes to PresentationQAValidator                             │
└──────────┬────────────────────────┬──────────────────────────────┘
           │                        │
           │ PHASE 1: GUIDANCE      │ PHASE 2: ENFORCEMENT
           ↓                        ↓
┌──────────────────────┐   ┌──────────────────────┐
│  Processor           │   │  QA Validator        │
│  (prompts.py)        │   │  (qa_validator.py)   │
└──────────────────────┘   └──────────────────────┘
```

## Phase 1: Guidance (Prompts)

Guardrails are **injected into extraction prompts** to guide the model toward topic-appropriate content.

### Code Flow

```python
# 1. Processor loads guardrails
processor = PresentationProcessor(
    guardrails=topic_guardrails  # loaded from JSON
)

# 2. Format guardrails for prompt
guardrail_text = format_guardrails_for_prompt(self.guardrails)

# 3. Include in extraction prompt
prompt = build_presentation_extraction_prompt(
    ...,
    topic_guardrails=guardrail_text
)
```

### Example Prompt Addition

If you have these guardrails:
```json
{
  "01": {
    "required_cues": ["benchmark", "framework", "infrastructure"],
    "min_summary_len": 6,
    "allow_generic": false
  }
}
```

The prompt will include:
```
TOPIC-SPECIFIC QUALITY REQUIREMENTS:
Topic 01 cues: benchmark, framework, infrastructure | min summary words: 6 | reject generic statements
```

This **guides** the model to:
- Focus on content containing those keywords
- Write summaries with at least 6 words
- Avoid generic phrases like "they discussed"

## Phase 2: Enforcement (QA Validation)

After the model generates extractions, **QA validation enforces** the rules. Any extraction failing validation is **rejected** (not stored).

### Validation Pipeline

```
Extraction Generated
       ↓
┌──────────────────────────────────────────┐
│  1. Timestamp Check (presentations only) │
│     - Within segment bounds              │
│     - Start < End                        │
│     - Slide alignment if marked          │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  2. Quote Verbatim Check                 │
│     - Quote exists in transcript/abstract│
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  3. Slide Reference Check                │
│     - Referenced slides exist            │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  4. Hallucination/Generic Check          │
│     - Check allow_generic setting        │
│     - Reject if contains generic phrases │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  5. Topic Guardrail Check (if defined)   │
│     - Required cues present?             │
│     - Min quote/summary length met?      │
└──────────────────────────────────────────┘
       ↓
   All Valid? ──Yes──> Store Extraction
       │
      No
       ↓
   Reject & Log Failure Reason
```

## Detailed Validation Rules

### 1. Generic Statement Detection

**Purpose**: Filter out vague, non-specific content

**Code Location**: `qa_validator.py:227-279`

```python
generic_phrases = [
    "they discussed",
    "the paper talks about",
    "this presentation covers",
    "they mentioned",
    "they talked about",
    "the speaker said",
    "it was about"
]

allow_generic = guardrails.get(topic_id, {}).get("allow_generic", False)
if not allow_generic:
    for phrase in summary.lower():
        if phrase in summary:
            # REJECT
```

**Key Insight**: This applies to **ALL topics** unless `allow_generic: true` is set!

**Default Behavior**:
- Topics **WITH** guardrails: Use `allow_generic` setting
- Topics **WITHOUT** guardrails: Default to `allow_generic: false` (reject generic)

### 2. Required Cues Check

**Purpose**: Ensure content matches topic-specific keywords

**Code Location**: `qa_validator.py:320-328`

```python
cues = rules.get("required_cues") or []
if cues:
    text_blob = (extraction.verbatim_quote + " " + extraction.summary).lower()
    if not any(cue.lower() in text_blob for cue in cues):
        # REJECT - Missing required keywords
```

**Example**:
- Topic 01 requires: `["benchmark", "framework", "infrastructure", "runtime", "serving"]`
- Extraction must contain **at least ONE** of these keywords (substring match)

### 3. Minimum Length Checks

**Purpose**: Ensure extractions are substantive

**Code Location**: `qa_validator.py:330-348`

```python
# Check quote length
min_quote = rules.get("min_quote_len")
if min_quote and extraction.verbatim_quote:
    if len(extraction.verbatim_quote.split()) < min_quote:
        # REJECT

# Check summary length
min_summary = rules.get("min_summary_len")
if min_summary and extraction.summary:
    if len(extraction.summary.split()) < min_summary:
        # REJECT
```

**Example**:
- Topic 19 (Notable Quotes): `min_quote_len: 12`, `min_summary_len: 10`
- Quote must be ≥12 words AND summary ≥10 words

## Configuration Parameters

### `required_cues` (array of strings)

**What it does**: Specifies keywords that must appear in the extraction

**How it works**:
- Case-insensitive substring match
- Checks both `verbatim_quote` AND `summary` fields
- **At least ONE** cue must be present (OR logic)

**Example**:
```json
"required_cues": ["benchmark", "framework", "infrastructure"]
```
✅ Passes: "We used TensorFlow **framework** for our experiments"
❌ Fails: "We used a new training method for our experiments"

### `min_quote_len` (integer)

**What it does**: Minimum word count for `verbatim_quote` field

**How it works**:
- Only checked if quote is provided
- Split by whitespace and count words
- Rejects if below threshold

**Example**:
```json
"min_quote_len": 12
```
✅ Passes: "We demonstrate that our method achieves state-of-the-art results on all benchmarks" (11 words, ≥12)
❌ Fails: "Our method works well" (4 words, <12)

### `min_summary_len` (integer)

**What it does**: Minimum word count for `summary` field

**How it works**:
- Always checked (summary is required)
- Split by whitespace and count words
- Rejects if below threshold

**Example**:
```json
"min_summary_len": 6
```
✅ Passes: "The paper introduces a novel architecture" (6 words)
❌ Fails: "Novel architecture introduced" (3 words)

### `allow_generic` (boolean)

**What it does**: Controls whether generic phrases are permitted

**Default**: `false` (reject generic content)

**How it works**:
- If `false`: Rejects extractions containing generic phrases
- If `true`: Allows generic phrases (useful for broad topics like scaling laws)

**Generic phrase list**:
- "they discussed"
- "the paper talks about"
- "this presentation covers"
- "they mentioned"
- "they talked about"
- "the speaker said"
- "it was about"

**Example**:
```json
"allow_generic": false
```
✅ Passes: "The paper introduces influence-aware clustering to estimate data value"
❌ Fails: "They discussed data selection methods"

```json
"allow_generic": true
```
✅ Passes: "They discussed scaling laws and efficiency improvements" (allowed for Topic 17)

## Topics Without Guardrails

If a topic is **not defined** in `topic_guardrails.json`:

1. ✅ **NO required keyword check** (any content accepted)
2. ✅ **NO minimum length check** (any length accepted)
3. ❌ **DEFAULT `allow_generic: false`** (generic phrases still rejected!)

**Important**: Even without explicit guardrails, generic content is filtered by default.

## Example: Topic 17 (LLM Scaling Laws)

### Configuration
```json
"17": {
  "required_cues": ["scaling", "efficiency", "compute", "tokens", "latency"],
  "allow_generic": true,
  "min_summary_len": 6
}
```

### Why `allow_generic: true`?

Scaling discussions are **inherently broad**. Valid extractions include:
- "They discussed scaling laws and compute efficiency"
- "The paper talks about token efficiency improvements"

Without `allow_generic: true`, these would be rejected as "generic."

### Validation Flow

```
Extraction: "They discussed scaling laws and achieved 2x efficiency"

1. Required cues check:
   - Contains "scaling"? ✅ YES
   - Contains "efficiency"? ✅ YES
   → PASS

2. Min summary length:
   - 8 words ≥ 6? ✅ YES
   → PASS

3. Generic statement check:
   - Contains "they discussed"? ✅ YES
   - allow_generic: true? ✅ YES
   → PASS (allowed due to setting)

Result: ✅ ACCEPTED
```

## Example: Topic 04 (Data Attribution) - No Guardrails

### Configuration
```json
// Topic 04 NOT defined in guardrails
```

### Validation Flow

```
Extraction: "They discussed data attribution methods"

1. Required cues check:
   → SKIP (no guardrails)

2. Min length check:
   → SKIP (no guardrails)

3. Generic statement check:
   - Contains "they discussed"? ✅ YES
   - allow_generic: false (default)? ✅ YES
   → FAIL

Result: ❌ REJECTED
```

This explains the **45% reduction** for Topic 04 in the comparison!

## Real-World Impact from V1 vs V2 Comparison

### Topic 17 (LLM Scaling): +12% extractions

**Why**: `allow_generic: true` captures legitimate scaling discussions
```
✅ V2 Accepted (V1 Rejected):
"The speaker highlights the importance of data selection as a method to improve cost efficiency"
```

### Topic 04 (Data Attribution): -45% extractions

**Why**: No guardrails = default `allow_generic: false` rejects generic content
```
❌ V2 Rejected (V1 Accepted):
"The researchers mention training a 'data influence model', which is a technique..."
```

### Topic 19 (Notable Quotes): -33% extractions

**Why**: `min_quote_len: 12` filters weak quotes
```
❌ V2 Rejected (V1 Accepted):
Quote: "Data selection matters" (3 words, <12)
```

### Topic 10 (Serendipity): +200% extractions

**Why**: `required_cues: ["unexpected", "surprising", "novel"]` targets actual serendipitous findings
```
✅ V2 Accepted (V1 Missed):
"The speaker reports a potentially surprising finding where a relational data model outperformed..."
```

## How to Adjust Guardrails

### Problem: Topic Over-Filtering (like Topic 04)

**Symptom**: Large reduction in extractions, legitimate content rejected

**Solution**: Add explicit `allow_generic: true`
```json
"04": {
  "required_cues": ["attribution", "provenance", "source"],
  "allow_generic": true,  // Allow broader data discussions
  "min_summary_len": 6
}
```

### Problem: Topic Under-Filtering

**Symptom**: Too many vague extractions

**Solution**: Add stricter requirements
```json
"06": {
  "required_cues": ["vision", "image", "visual", "CNN", "transformer"],
  "allow_generic": false,  // Reject generic content
  "min_summary_len": 8,
  "min_quote_len": 12
}
```

### Problem: Missing Important Content

**Symptom**: Required cues too specific, valid content rejected

**Solution**: Expand keyword list or use broader terms
```json
"05": {
  // Before (too specific)
  "required_cues": ["marketplace", "pricing", "compensation"],

  // After (broader coverage)
  "required_cues": ["market", "valuation", "licensing", "pricing", "compensat", "monetiz"],
}
```

## Best Practices

### 1. Start with Minimal Guardrails

For new topics, begin with:
```json
"XX": {
  "required_cues": [],
  "allow_generic": true,
  "min_summary_len": 5
}
```

Then tighten based on observed extraction quality.

### 2. Use `allow_generic: true` for Broad Topics

Topics that naturally include meta-discussion:
- Scaling laws and efficiency (17)
- Data preparation (03)
- GenAI architectures (02)

### 3. Use `allow_generic: false` for Specific Topics

Topics requiring concrete technical content:
- Data markets (05) - needs pricing/valuation specifics
- Serendipitous findings (10) - needs surprise/novelty
- Notable quotes (19) - needs strong phrasing

### 4. Test Changes on Small Sample

Before full run:
```bash
# Test with 5 presentations
python scripts/run_exhaustive_extraction.py \
  --mode presentations \
  --limit-presentations 5 \
  --output-dir test_output

# Compare extraction counts per topic
cat test_output/progress.json | jq '.topic_distribution'
```

### 5. Monitor QA Failure Reasons

During extraction, watch for:
```
⚠️  QA failed for extraction: xyz_01_extract001
   - topic_guardrail_check: Missing required cues for topic 01: benchmark, framework
```

High failure rates indicate guardrails may need adjustment.

## Code References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Load guardrails | `guardrails.py` | 24-45 | Load JSON config |
| Format for prompt | `prompts.py` | 278-304 | Add to extraction prompts |
| Generic detection | `qa_validator.py` | 227-279 | Check generic phrases |
| Guardrail enforcement | `qa_validator.py` | 307-348 | Validate rules |
| Integration | `processor.py` | 41-60 | Pass guardrails to QA |

## Summary

The guardrails system provides **two-layer quality control**:

1. **Guidance Layer** (Prompts): Hints to the model about topic-specific requirements
2. **Enforcement Layer** (QA Validation): Hard validation that rejects non-compliant extractions

**Key Insight**: Even without explicit guardrails, generic content is filtered by default (`allow_generic: false`). This explains unexpected filtering for topics like 02, 04, and 08.

**Recommendation**: Define guardrails for ALL 20 topics to have explicit control over quality rules rather than relying on defaults.
