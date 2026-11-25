"""Local image-embedding similarity between a rendered figure and its source.

Two embedding metrics with two distinct failure contracts:

- CLIP (secondary, co-recorded metric — unchanged contract): the model loads
  lazily and any failure degrades silently to ``None``. The score is recorded
  in receipts for comparison against DINOv2 during Phase 02 calibration; the
  Sonnet vision critique remains the structural gate.
- DINOv2 (primary scored metric, v2.4 gate stack): ``preflight_dinov2()``
  hard-fails with a ``RuntimeError`` BEFORE any provider spend when the model
  cannot load (D-02) — a calibration tuple without the primary metric is
  unlabeled. After a successful preflight, per-image scoring errors degrade
  to ``None`` plus a recorded warning so a batch the budget already paid for
  is never killed mid-run (D-02 second half).

DINOv2 weights are pinned to ``DINOV2_REVISION`` (D-04) and downloaded only
by the explicit ``prefetch_dinov2_weights()`` step; runs load offline from
the warm HF cache. Device selection is MPS-else-CPU (D-03) and is reported
alongside the pinned revision by ``dinov2_runtime_info()`` for receipts.
"""

from __future__ import annotations

import os
from pathlib import Path

_MODEL_CACHE: dict[str, object] = {}

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
DEFAULT_SIMILARITY_THRESHOLD = 0.75


def _load_clip() -> tuple[object, object, object] | None:
    if "bundle" in _MODEL_CACHE:
        return _MODEL_CACHE["bundle"]  # type: ignore[return-value]
    try:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        model.eval()
        bundle = (model, preprocess, torch)
    except Exception:
        bundle = None
    _MODEL_CACHE["bundle"] = bundle
    return bundle


def clip_image_similarity(image_a: Path, image_b: Path) -> float | None:
    """Cosine similarity of CLIP image embeddings, or None when CLIP is unavailable."""

    bundle = _load_clip()
    if bundle is None:
        return None
    model, preprocess, torch = bundle
    try:
        from PIL import Image

        with torch.no_grad():
            tensors = []
            for path in (image_a, image_b):
                image = Image.open(path).convert("RGB")
                tensors.append(preprocess(image).unsqueeze(0))
            features = [model.encode_image(t) for t in tensors]
            features = [f / f.norm(dim=-1, keepdim=True) for f in features]
            return float((features[0] @ features[1].T).item())
    except Exception:
        return None


DINOV2_MODEL_NAME = "facebook/dinov2-base"  # ViT-B/14 (D-01)
DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"  # pinned (D-04)


def _load_dinov2() -> dict[str, object]:
    """Lazily load DINOv2 at the pinned revision, cached in ``_MODEL_CACHE``.

    Mirrors ``_load_clip``'s module-cache shape but intentionally does NOT
    swallow exceptions: load failures PROPAGATE so ``preflight_dinov2()`` can
    hard-fail before any provider spend (the D-02 divergence from CLIP).
    """

    if "dinov2" in _MODEL_CACHE:
        return _MODEL_CACHE["dinov2"]  # type: ignore[return-value]
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        DINOV2_MODEL_NAME, revision=DINOV2_REVISION
    )
    model = AutoModel.from_pretrained(
        DINOV2_MODEL_NAME, revision=DINOV2_REVISION
    ).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # D-03
    model.to(device)
    bundle: dict[str, object] = {
        "processor": processor,
        "model": model,
        "torch": torch,
        "device": device,
    }
    _MODEL_CACHE["dinov2"] = bundle
    return bundle


def preflight_dinov2() -> dict:
    """Load DINOv2 once, BEFORE any provider spend (D-02 hard-fail contract).

    Returns ``dinov2_runtime_info()`` on success so callers can embed the
    model name, pinned revision, and device in receipts. Raises
    ``RuntimeError`` with prefetch remediation when the model cannot load.
    Callers (trial runner ``main``, replay harness) MUST invoke this before
    the first provider call.
    """

    try:
        _load_dinov2()
    except Exception as exc:  # noqa: BLE001 — re-raised with remediation
        raise RuntimeError(
            f"DINOv2 preflight failed: could not load {DINOV2_MODEL_NAME} at "
            f"pinned revision {DINOV2_REVISION} ({exc}). Run "
            "prefetch_dinov2_weights() once with network access "
            "(HF_HUB_OFFLINE unset) to warm the local HF cache, then retry. "
            "This hard-fail happens before any provider spend (D-02)."
        ) from exc
    return dinov2_runtime_info()


def prefetch_dinov2_weights() -> str:
    """Download the pinned DINOv2 snapshot into the standard HF cache.

    This is the ONLY network step in the DINOv2 path (D-04): trial and replay
    runs load offline from the warm cache afterwards. It must NOT be run with
    ``HF_HUB_OFFLINE=1`` set — offline mode blocks the download itself; unset
    the variable for this one explicit step. Returns the snapshot path.
    """

    from huggingface_hub import snapshot_download

    return str(
        snapshot_download(
            repo_id=DINOV2_MODEL_NAME,
            revision=DINOV2_REVISION,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            ],
        )
    )


def dinov2_runtime_info() -> dict:
    """Receipt-embeddable DINOv2 configuration (D-03/D-04 traceability).

    Phase 02 calibration data must be traceable to the exact embedding
    configuration, so receipts embed this dict. ``device`` is ``None`` until
    the model has been loaded (preflight or first similarity call).
    """

    bundle = _MODEL_CACHE.get("dinov2")
    device = bundle["device"] if isinstance(bundle, dict) else None
    return {
        "model_name": DINOV2_MODEL_NAME,
        "revision": DINOV2_REVISION,
        "device": device,
        # WR-03: truthy-string parse matching huggingface_hub's own env-var
        # semantics — falsy overrides like "0"/"false" must NOT read as
        # offline, or the receipt claims offline for an online run.
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "").strip().upper()
        in {"1", "ON", "YES", "TRUE"},
    }


def dinov2_image_similarity(
    image_a: Path, image_b: Path, warnings: list[str] | None = None
) -> float | None:
    """Cosine similarity of DINOv2 class embeddings (primary scored metric).

    Same two-Path signature as ``clip_image_similarity`` so the trial runner
    calls both metrics symmetrically. The model bundle comes from
    ``_load_dinov2()`` (cached after a successful preflight). Per-image
    scoring errors AFTER a successful load degrade to ``None`` plus a
    human-readable entry appended to ``warnings`` when provided — the
    post-preflight degrade-to-warning contract (D-02 second half).
    """

    bundle = _load_dinov2()
    processor = bundle["processor"]
    model = bundle["model"]
    torch = bundle["torch"]
    device = bundle["device"]
    try:
        from PIL import Image

        with torch.no_grad():  # type: ignore[attr-defined]
            features = []
            for path in (image_a, image_b):
                image = Image.open(path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)  # type: ignore[operator]
                out = model(**inputs)  # type: ignore[operator]
                vector = out.pooler_output  # layernormed CLS token, (1, 768)
                features.append(vector / vector.norm(dim=-1, keepdim=True))
            return float((features[0] @ features[1].T).item())
    except Exception as exc:  # noqa: BLE001 — post-preflight degrade (D-02)
        if warnings is not None:
            warnings.append(
                f"dinov2 scoring failed for ({image_a}, {image_b}): {exc!r}"
            )
        return None


def margin_sign_test(
    sim_chosen: float, distractor_sims: list[float]
) -> dict:
    """Margin-vs-distractors sign test (CALIB-03) — pure arithmetic.

    ``margin = sim_chosen - max(distractor_sims)`` and
    ``margin_sign = margin > 0.0`` (strict inequality: a tie is NOT a pass,
    per the CALIB-03 spec ``sim(render, chosen) > max sim(render,
    distractor)``). Sign test only — no tunable threshold (locked CALIB-03).

    An empty distractor pool returns ``{"margin": None, "margin_sign": None,
    "distractor_count": 0}`` so callers can record a warning instead of
    crashing. The distractor pool definition is supplied by the caller and is
    recorded in receipts under protocol id ``"bound_papers_v1"``; Phase 02
    re-derives pools from that protocol id. No torch import, no model
    coupling — this function must stay usable without any ML dependency.
    """

    if not distractor_sims:
        return {"margin": None, "margin_sign": None, "distractor_count": 0}
    margin = sim_chosen - max(distractor_sims)
    return {
        "margin": margin,
        "margin_sign": margin > 0.0,
        "distractor_count": len(distractor_sims),
    }
