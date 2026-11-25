"""v6.1 Phase 3: the gold lane CLI end to end on a tiny workspace (saved producer, zero calls)."""

from __future__ import annotations

import json

from conference_pipeline.ontology import gold_label as gl
from conference_pipeline.ontology import gold_run as gr
from tests.test_ontology_gold_label import PAPER, _extract


def _workspace(tmp_path, *, approved=True):
    ws = tmp_path / "ws"
    (ws / "ontology").mkdir(parents=True)
    (ws / "texts").mkdir()
    families, records, groups = [], [], {}
    for n in range(4):
        key = f"openreview:W{n}"
        families.append({"family_key": key, "kind": "main-paper", "topics": ["01_alpha"], "workshops": [],
                         "split_group": f"g:{key}", "artifacts": [f"paper_text:{n}"]})
        (ws / f"texts/{n}.txt").write_text(PAPER.replace("UnMaskFork", f"Method{n}") if n else PAPER)
        records.append({"artifact_id": f"paper_text:{n}", "material_type": "paper", "text_path": f"texts/{n}.txt"})
        groups[f"g:{key}"] = [key]
    roster = {"families": families, "unjoined": []}
    (tmp_path / "roster.json").write_text(json.dumps(roster))
    (ws / "storage/corpus").mkdir(parents=True)
    (ws / "storage/corpus/corpus_inventory.json").write_text(json.dumps({"artifact_records": records}))
    (ws / "ontology/splits.json").write_text(json.dumps({"splits": {"dev": ["g:openreview:W0", "g:openreview:W1"],
                                                                    "sealed_test": ["g:openreview:W2"],
                                                                    "growth": ["g:openreview:W3"]},
                                                         "groups": groups}))
    (ws / "ontology/competency-questions.json").write_text(json.dumps({"questions": [
        {"id": "Q-01-1", "area": "01", "priority": "core", "kind": "lookup", "evidence": '"HumanEval" 3'}]}))
    (ws / "ontology/workshop-areas.json").write_text(json.dumps({"schema": "ontology-workshop-areas/1",
                                                                 "status": "approved" if approved else "proposed",
                                                                 "workshops": []}))
    return ws, tmp_path / "roster.json"


def _save_all(root, works):
    for work in works:
        for role in gl.ROLES:
            for kind, text in (("extract", _extract(role)),
                               ("check", json.dumps({"labels": [{"id": f"{'s' if role == 'primary' else 'p'}s01",
                                                                 "support": "supported"},
                                                                {"id": f"{'s' if role == 'primary' else 'p'}d01",
                                                                 "support": "contradicted"}]}))):
                path = gl.response_path(root, role, kind, work)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"text": text, "stop_reason": "end"}))


def test_select_needs_an_approved_workshop_map(tmp_path, capsys):
    ws, roster = _workspace(tmp_path, approved=False)
    assert gr.main(["select", "--workspace", str(ws), "--roster", str(roster), "--salt", "s"]) == gr.EXIT_INPUT
    assert "not approved" in capsys.readouterr().out


def test_select_estimate_label_queue_build(tmp_path, capsys):
    ws, roster = _workspace(tmp_path)
    assert gr.main(["select", "--workspace", str(ws), "--roster", str(roster), "--salt", "s", "--per-area", "2"]) == 0
    selection = json.loads((ws / "ontology/gold-selection.json").read_text())
    assert selection["counts"] == {"dev": 2, "sealed_test": 1}
    capsys.readouterr()
    assert gr.main(["estimate", "--workspace", str(ws), "--roster", str(roster)]) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["works"] == 3 and estimate["usd"] > 0
    assert estimate["by_role"]["primary"]["calls"] == 6

    responses, labels = tmp_path / "responses", tmp_path / "labels"
    _save_all(responses, ["openreview:W0", "openreview:W1", "openreview:W2"])
    assert gr.main(["label", "--workspace", str(ws), "--roster", str(roster), "--out", str(labels),
                    "--responses", str(responses)]) == 0
    summary = json.loads((labels / "label-summary.json").read_text())
    assert summary == {"labelled": 3, "not_reached": [], "calls": 0, "spent_usd": 0.0,
                       "link_pairs": {"dev": 0, "sealed_test": 0}}
    assert gr.main(["label", "--workspace", str(ws), "--roster", str(roster), "--out", str(labels),
                    "--responses", str(responses), "--producer", "live"]) == gr.EXIT_INPUT

    queue = tmp_path / "queue.json"
    assert gr.main(["queue", "--labels", str(labels), "--salt", "s", "--out", str(queue)]) == 0
    queued = json.loads(queue.read_text())["items"]
    rulings = {q["key"]: {"ruling": "accept-author", "date": "2026-11-18", "words": "fixture"} for q in queued}
    # the two labellers wrote the same distractor; keep it once
    rulings_path = tmp_path / "rulings.json"
    rulings_path.write_text(json.dumps({"rulings": rulings}))
    out = tmp_path / "pairs"
    code = gr.main(["build", "--workspace", str(ws), "--roster", str(roster), "--labels", str(labels),
                    "--rulings", str(rulings_path), "--salt", "s", "--out", str(out)])
    assert code == 0, capsys.readouterr().out
    digests = json.loads((out / "digests.json").read_text())
    assert set(digests) == {"dev", "sealed_test"} and (out / "dev/corpus.json").is_file()


def test_a_bad_ruling_stops_the_build(tmp_path, capsys):
    ws, roster = _workspace(tmp_path)
    rulings = tmp_path / "rulings.json"
    rulings.write_text(json.dumps({"rulings": {"k": {"ruling": "maybe"}}}))
    assert gr.main(["build", "--workspace", str(ws), "--roster", str(roster), "--labels", str(tmp_path),
                    "--rulings", str(rulings), "--salt", "s", "--out", str(tmp_path / "o")]) == gr.EXIT_PROBLEM
