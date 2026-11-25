"""Runs inside GND 0.4.0's interpreter for ``ontology.admission`` (v6.1 Phase 4, lane A).

``receipts <config>``: writes a consumer-pinned judgment policy and its receipts for the given
targets, with GND's own digests (``gnd.judgments.judgment_fact_digest``,
``judgment_receipt_digest``) so ``gnd judgments check`` binds them. Each judged target gets one
receipt from the one judge, quoting its evidence as a byte slice of a file under the trusted
root; a target with no verdict gets no receipt (``missing``). Prints ``{"policy": ...,
"receipts": ...}`` file names, or ``{"refused": reason}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gnd.artifacts import canonical_json_bytes, sha256_hex
from gnd.judgments import judgment_fact_digest, judgment_receipt_digest
from gnd.model import load_model


def receipts(config: dict) -> dict:
    model_bytes = Path(config["model"]).read_bytes()
    raw_bytes = Path(config["population"]).read_bytes()
    population = json.loads(raw_bytes)
    trusted = Path(config["trusted_root"])
    judge = config["judge"]
    # The identity GND's mechanical report binds (checking_service: reference_binding); None when
    # the model binds no reference population.
    references = getattr(load_model(Path(config["model"])), "_references", None)
    identity = {"identity_version": 1, "references": [{"id": o.id, "sha256": o.raw_sha256}
                                                      for o, _, _ in references._snapshots]} if references else None
    out, pins = [], []
    for n, target in enumerate(config["targets"]):
        verdict = target.get("verdict")
        if verdict is None:
            continue
        ft, index = target["fact_type"], target["index"]
        source = trusted / target["path"]
        data = source.read_bytes()
        start, end = target["start"], target["end"]
        receipt = {
            "receipt_id": f"{judge}-{n:05d}", "judge_id": judge, "verdict": verdict, "trust": "consumer-pinned",
            "subject": {"model_sha256": sha256_hex(model_bytes), "population_sha256": sha256_hex(raw_bytes),
                        "reference_identity": identity, "fact_type": ft, "index": index,
                        "fact_sha256": judgment_fact_digest(ft, index, population["facts"][ft][index])},
            "sources": [{"path": target["path"], "sha256": sha256_hex(data),
                         "quote": {"start": start, "end": end, "sha256": sha256_hex(data[start:end])}}],
            "external_evidence": None,
        }
        pins.append({"judge_id": judge, "receipt_sha256": judgment_receipt_digest(receipt)})
        out.append(receipt)
    policy = {"schema_version": 1, "policy_id": config["policy_id"], "judges": [judge],
              "admission": {"evidence_floor": None, "minimum_approvals": 1},
              "repair": {"evidence_floor": None, "minimum_rejections": 1},
              "targets": [{"fact_type": t["fact_type"], "index": t["index"]} for t in config["targets"]],
              "trust_mode": "consumer-pinned", "trusted_receipts": pins}
    folder = Path(config["out"])
    (folder / "policy.json").write_bytes(canonical_json_bytes(policy))
    (folder / "receipts.json").write_bytes(canonical_json_bytes({"schema_version": 1, "receipts": out}))
    return {"policy": "policy.json", "receipts": "receipts.json"}


def main(argv: list[str]) -> int:
    config = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    try:
        result = {"receipts": receipts}[argv[1]](config)
    except Exception as exc:  # noqa: BLE001 - GND's refusals become reasons
        result = {"refused": f"{type(exc).__name__}: {exc}"[:400]}
    sys.stdout.write(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
