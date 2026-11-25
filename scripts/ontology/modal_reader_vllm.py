"""Modal app: OpenAI-compatible vLLM endpoint for the smaller open-weights broad readers (v6.1 Phase 5, P5-D5).

Governed task: ``ontology.read_broad`` in src/conference_pipeline/model_governance/registry.yaml
(catalog aliases ``gemma4_12b_modal`` = Gemma 4 12B bf16 on L40S, ``qwen36_35b_a3b_modal`` =
Qwen3.6-35B-A3B bf16 on H100). The owner approved both for the pilot's broad arm on 2026-11-22
("all"), inside the broad arm cap of USD 10.

The variant is chosen at deploy time by ONTOLOGY_READER_VARIANT; each deploys as its own app:

    set -a; source <main checkout>/.env; set +a
    ONTOLOGY_READER_VARIANT=gemma4-12b modal deploy scripts/ontology/modal_reader_vllm.py
    modal app stop ontology-reader-gemma4-12b --yes   # the pilot harness does this in a finally

Cost control as in the v5.2 serving app: one container at most, five idle minutes before
scale-down. The API key is the Modal secret ``ai4science-qwen-api-key`` (MODAL_QWEN_API_KEY in
``.env``, never in receipts); the Hugging Face cache volume is shared with that app. Model
strings here are the serving definition, not a pipeline model selection (model-literal-ok: serving app).
"""
from __future__ import annotations

import os
import subprocess

import modal

VARIANTS = {
    "gemma4-12b": {"model": "google/gemma-4-12B-it", "gpu": "L40S",  # model-literal-ok: Modal serving app definition
                   "extra": ["--limit-mm-per-prompt", '{"image": 0, "audio": 0}']},
    "qwen36-a3b": {"model": "Qwen/Qwen3.6-35B-A3B", "gpu": "H100",  # model-literal-ok: Modal serving app definition
                   "extra": ["--limit-mm-per-prompt", '{"image": 0, "video": 0}', "--reasoning-parser", "qwen3"]},
}
VARIANT = os.environ.get("ONTOLOGY_READER_VARIANT", "gemma4-12b")
MODEL_NAME = VARIANTS[VARIANT]["model"]
GPU = VARIANTS[VARIANT]["gpu"]
VLLM_PORT = 8000
MAX_MODEL_LEN = 16384  # the Phase 4 broad pass sends at most six tier-A sections (~6k tokens) plus an 8k reply
SCALEDOWN_WINDOW_S = 5 * 60
TIMEOUT_S = 60 * 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "ONTOLOGY_READER_VARIANT": VARIANT})  # the container re-imports this module
)

app = modal.App(f"ontology-reader-{VARIANT}")
hf_cache = modal.Volume.from_name("ai4science-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("ontology-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW_S,
    max_containers=1,
    timeout=TIMEOUT_S,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("ai4science-qwen-api-key")],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", VARIANT,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "32",
        "--max-num-batched-tokens", "8192",
        "--api-key", os.environ["MODAL_QWEN_API_KEY"],
        *VARIANTS[VARIANT]["extra"],
    ]
    print("vllm version:", subprocess.run(["vllm", "--version"], capture_output=True, text=True).stdout.strip())
    subprocess.Popen(cmd)
