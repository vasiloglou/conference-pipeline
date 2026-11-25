#!/usr/bin/env bash
#
# Run the CVPR 2026 post-14A analysis chain end to end.
#
# The production runner's --stage analysis only *consumes* the 07A..07H
# sub-step receipts; it cannot produce them (there is no --substep flag), and
# its pre-spend gate refuses the stage while any of them is stale. Each
# sub-step is its own command. The order below is the order that actually
# reaches a fresh analysis receipt (learned the hard way on 2026-09-20):
#
#   14A llm-full-read (already FRESH)
#     -> 07A exhaustive-topic-summaries        [PAID: gemini]
#       -> 07B deep-dive-question-selection    [PAID: anthropic]
#         -> 07C dual-deep-dive-research       [local]
#           -> paper matching (12 topics)      [PAID: openai, unpriced]
#             -> analysis pass 1               [$0: rebuilds source packs]
#               -> 07D/07E/07H reconciliation  [local]
#                 -> analysis pass 2           [PAID]
#
# Why paper matching comes BEFORE reconciliation: the analysis stage builds
# analysis/narrative_source_packs/*.json from the matched papers, and those
# packs are the hashed input of 07D. Reconciling first and matching second
# always ends in `upstream_lineage_not_fresh` at the analysis gate. The $0
# first analysis pass exists only to rebuild the packs; it is EXPECTED to
# block at the reconciliation gate (or to succeed outright when the packs
# did not change), and any other blocker halts the chain.
#
# Everything else is straight-line: no retries, no polling. `set -e` halts
# the chain at the first failure so a broken step never feeds the next one,
# and every step is --resume so re-running after a halt picks up where it
# stopped rather than re-spending.
#
# Provider and model are deliberately NOT passed. They resolve from
# src/conference_pipeline/model_governance/registry.yaml, which is the single
# model-selection authority; passing literals here would fork that decision.
#
# Usage:  bash scripts/run_cvpr2026_post_14a_chain.sh
#
set -euo pipefail

WORKSPACE="${WORKSPACE:-conferences/cvpr2026}"
RUN_ID="${RUN_ID:-cvpr2026-live-production}"
PY="${PY:-.venv/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${WORKSPACE}/storage/logs"
LOG="${LOG_DIR}/post-14a-chain-${STAMP}.log"

cd "$(dirname "$0")/.."
mkdir -p "${LOG_DIR}"

# Credentials come from this worktree's own .env, never from the environment
# of whoever launched the script.
set -a
# shellcheck disable=SC1091
source .env
set +a

# The local lanes (07C researcher, 07D reconciliation) talk to an
# OpenAI-compatible server. The library default is 127.0.0.1:8080, which Docker
# Desktop squats on this machine (its 404s look like a provider outage), so
# pin the mlx_lm server port and refuse to start if nothing answers there.
export GPTR_LOCAL_LLM_BASE_URL="${GPTR_LOCAL_LLM_BASE_URL:-http://127.0.0.1:8081/v1}"
if ! curl -sf -m 5 "${GPTR_LOCAL_LLM_BASE_URL}/models" >/dev/null; then
  echo "no local LLM server answering at ${GPTR_LOCAL_LLM_BASE_URL}; start it with:" >&2
  echo "  ${PY} -m mlx_lm server --host 127.0.0.1 --port 8081 --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit" >&2
  exit 1
fi

step() {
  local label="$1"; shift
  echo ""                                               | tee -a "${LOG}"
  echo "=== ${label}  $(date -u +%H:%M:%SZ) ==="        | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

# Print what a sub-step actually cost, straight from its receipt, so spend is
# visible as the chain runs instead of only at the end.
receipt_cost() {
  "${PY}" - "${WORKSPACE}" "${RUN_ID}" "$1" <<'PY' 2>/dev/null | tee -a "${LOG}" || true
import json, os, sys
workspace, run_id, name = sys.argv[1], sys.argv[2], sys.argv[3]
path = os.path.join(workspace, "storage", "workflow-runs", run_id, name)
try:
    d = json.load(open(path))
except OSError:
    sys.exit(0)
cost = d.get("estimated_cost_usd")
print(
    f"    receipt: status={d.get('status')} "
    f"cost={'n/a' if cost is None else f'${cost:.2f}'} "
    f"elapsed={(d.get('elapsed_seconds') or 0)/60:.1f}min "
    f"paid_calls={d.get('paid_provider_calls_made')}"
)
PY
}

# The analysis stage command, shared by both passes.
analysis_stage() {
  "${PY}" scripts/run_cvpr2026_production.py \
    --workspace "${WORKSPACE}" --run-id "${RUN_ID}" \
    --stage analysis --mode live --resume --summary \
    --provider-cost-approved --env-file .env
}

# Accept the pass-1 outcome only if the stage succeeded or blocked solely on
# stale reconciliation lineage; anything else is a real failure. A blocked
# pass is written as stage-07-analysis.attempt-<stamp>.json (the runner never
# overwrites an existing main receipt with a blocker), so judge the newest
# receipt written after the pass started, never the main file by name.
assert_pack_rebuild_outcome() {
  "${PY}" - "${WORKSPACE}" "${RUN_ID}" "$1" <<'PY' | tee -a "${LOG}"
import glob, json, os, sys
workspace, run_id, marker = sys.argv[1], sys.argv[2], sys.argv[3]
run_dir = os.path.join(workspace, "storage", "workflow-runs", run_id)
started = os.path.getmtime(marker)
candidates = [
    p for p in glob.glob(os.path.join(run_dir, "stage-07-analysis*.json"))
    if os.path.getmtime(p) >= started
]
if not candidates:
    print("    analysis pass 1 wrote no receipt; halting the chain", file=sys.stderr)
    sys.exit(1)
path = max(candidates, key=os.path.getmtime)
d = json.load(open(path))
status = d.get("status")
codes = sorted({str(b.get("blocker_code") or b.get("code") or "") for b in (d.get("blockers") or []) if isinstance(b, dict)})
print(f"    analysis pass 1: status={status} blockers={codes} receipt={os.path.basename(path)}")
if status == "succeeded":
    sys.exit(0)
if status == "blocked" and codes == ["upstream_lineage_not_fresh"]:
    sys.exit(0)
print("    unexpected pass-1 outcome; halting the chain", file=sys.stderr)
sys.exit(1)
PY
}

echo "CVPR 2026 post-14A chain  run_id=${RUN_ID}  local_llm=${GPTR_LOCAL_LLM_BASE_URL}  started $(date -u +%FT%TZ)" | tee "${LOG}"

# 07A is a stop point: it refuses unless the operator chose the summary model
# (workflow.exhaustive_summary_model in config.yaml, or --provider here).
step "07A exhaustive-topic-summaries  [PAID; model = operator choice]" \
  "${PY}" scripts/summarize_llm_full_read_topics.py \
    --workspace "${WORKSPACE}" --run-id "${RUN_ID}" \
    --mode live --provider-cost-approved \
    --max-input-tokens 400000 --resume --summary
receipt_cost "07A-exhaustive-topic-summaries.json"

step "07B deep-dive-question-selection  [PAID: anthropic]" \
  "${PY}" scripts/select_deep_dive_questions.py \
    --workspace "${WORKSPACE}" --run-id "${RUN_ID}" \
    --mode live --provider-cost-approved --summary
receipt_cost "07B-deep-dive-question-selection.json"

step "07C dual-deep-dive-research  [local]" \
  "${PY}" scripts/run_dual_deep_dive_research.py \
    --workspace "${WORKSPACE}" --run-id "${RUN_ID}" \
    --mode live --resume --summary
receipt_cost "07C-dual-deep-dive-research-lanes.json"

# Topic ids come from the conference's own topic strategy, not a hardcoded
# list, so this stays correct if the topic set is ever revised.
# (a while-read loop, not `mapfile` -- macOS still ships bash 3.2, which has no
# mapfile, and this script must run under /bin/bash there.)
TOPIC_IDS=""
while IFS= read -r topic_id; do
  [ -n "${topic_id}" ] && TOPIC_IDS="${TOPIC_IDS} ${topic_id}"
done <<EOF
$("${PY}" -c "
import json
rows = json.load(open('${WORKSPACE}/analysis/topic_strategy.json'))['final_topics']
print('\n'.join(str(r.get('topic_id') or r.get('id')) for r in rows))
")
EOF
echo "" | tee -a "${LOG}"
echo "=== paper matching over $(echo ${TOPIC_IDS} | wc -w | tr -d ' ') topics  [PAID: openai, unpriced in the estimate] ===" | tee -a "${LOG}"
for topic_id in ${TOPIC_IDS}; do
  step "paper-matching ${topic_id}" \
    "${PY}" scripts/match_papers_from_artifact_keypoints.py \
      --workspace "${WORKSPACE}" --run-id "${RUN_ID}" --topic-id "${topic_id}"
done

echo "" | tee -a "${LOG}"
echo "=== analysis pass 1: rebuild source packs from the matched papers  [\$0; expected to block on reconciliation lineage]  $(date -u +%H:%M:%SZ) ===" | tee -a "${LOG}"
PASS1_MARKER="$(mktemp "${LOG_DIR}/.pass1-marker.XXXXXX")"
set +e
analysis_stage 2>&1 | tee -a "${LOG}"
set -e
assert_pack_rebuild_outcome "${PASS1_MARKER}"
rm -f "${PASS1_MARKER}"

step "07D/07E/07H reconciliation + reference workflow against the rebuilt packs  [local]" \
  "${PY}" scripts/run_topic_reconciliation_reference_workflow.py \
    --workspace "${WORKSPACE}" --run-id "${RUN_ID}" \
    --mode live --resume --summary
receipt_cost "07H-topic-reconciliation-reference-workflow.json"

step "analysis pass 2  [PAID]" analysis_stage
receipt_cost "stage-07-analysis.json"

echo "" | tee -a "${LOG}"
echo "chain finished $(date -u +%FT%TZ) -- log: ${LOG}" | tee -a "${LOG}"
echo "Next: ${PY} -m conference_pipeline.cli workflow stage-status --workspace ${WORKSPACE} --run-id ${RUN_ID}" | tee -a "${LOG}"
