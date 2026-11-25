"""v6.0 Phase 7.1, review phase-7.1-r3: one regression test per finding (F1-F4)."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conference_pipeline.publishing.dataset_package import DatasetPackageError, verified_hf_upload
from tests.test_phase71_review_r1 import _listed_bundle

HOME_PATH = "/" + "Users" + "/reviewer/private-project"


# --- F1: the hub is scanned for private markers and secrets before it deploys ----------


def _hub(tmp_path: Path) -> Path:
    from conference_pipeline.publishing.hub_site import build_hub_site
    from tests.public_site_fixtures import make_alpha, make_beta

    hub = tmp_path / "hub"
    build_hub_site([make_alpha(tmp_path), make_beta(tmp_path)], hub, hub_title="Fixture hub")
    return hub


def test_f1_a_hub_holding_a_private_marker_is_a_finding(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_hub import hub_findings

    hub = _hub(tmp_path)
    assert hub_findings(hub, secret_scanner=lambda paths: []) == []
    (hub / "notes.txt").write_text(f"see {HOME_PATH}\n", encoding="utf-8")
    findings = hub_findings(hub, secret_scanner=lambda paths: [])
    assert any("notes.txt" in f and "private_marker" in f for f in findings)


def test_f1_the_secrets_gate_runs_over_the_hub(tmp_path: Path) -> None:
    from conference_pipeline.publishing.release_hub import hub_findings

    hub = _hub(tmp_path)
    seen: list[int] = []

    def scanner(paths):
        seen.append(len(paths))
        return [{"RuleID": "fake", "File": str(paths[0])}]

    findings = hub_findings(hub, secret_scanner=scanner)
    assert seen and any(f.startswith("secrets") for f in findings)


# --- F2: RELEASE.json is still checked against its listed digest -----------------------


def test_f2_a_release_json_changed_under_its_listing_refuses(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["family_licenses"] = {"decks": "CC0-1.0"}
    (bundle / "RELEASE.json").write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(DatasetPackageError, match="RELEASE.json"):
        verified_hf_upload(bundle, secret_scanner=lambda paths: [])


# --- F3: the semantic proof receipt neither lists itself nor uploads -------------------


def test_f3_an_index_built_with_a_semantic_proof_verifies(tmp_path: Path) -> None:
    from conference_pipeline.publishing import public_index as pi
    from conference_pipeline.publishing.dataset_package import DatasetPackageConfig, package_public_dataset
    from tests.public_site_fixtures import make_alpha

    bundle = make_alpha(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["included_families"] = release.get("included_families") or ["decks"]
    release["family_licenses"] = {family: "MIT" for family in release["included_families"]}
    (bundle / "RELEASE.json").write_text(json.dumps(release), encoding="utf-8")
    checksums = json.loads((bundle / "CHECKSUMS.json").read_text(encoding="utf-8"))
    checksums["RELEASE.json"] = {"sha256": hashlib.sha256((bundle / "RELEASE.json").read_bytes()).hexdigest()}
    (bundle / "CHECKSUMS.json").write_text(json.dumps(checksums), encoding="utf-8")

    def ones(chunks):
        return np.ones((len(chunks), 8), dtype=np.float32)

    def browser(chunks):
        vectors = np.ones((len(chunks), 768), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    for query in ("hello", "another query"):
        pi.build_public_index(
            pi.PublicIndexConfig(bundle_root=bundle, embedding_dim=8, canonical_embed_fn=ones, browser_embed_fn=browser, semantic_proof_query=query)
        )
    package_public_dataset(DatasetPackageConfig(bundle_root=bundle, metadata_dir=bundle / "metadata", receipt_path=tmp_path / "r.json"))
    relatives = verified_hf_upload(bundle, secret_scanner=lambda paths: [])
    assert not [r for r in relatives if r.endswith("semantic_query_proof.json")]


# --- F4: the SDK-free id rules match the Hub's on the namespace too --------------------


@pytest.mark.parametrize("repo", ["o-/a", "-o/a", "o./a", "o/a", "vasiloglou/iclr2026-data"])
def test_f4_without_the_sdk_the_rules_match_the_hub(repo: str, monkeypatch) -> None:
    from huggingface_hub.utils import validate_repo_id

    from conference_pipeline.publishing import release_layout

    try:
        validate_repo_id(repo)
        hub_ok = True
    except Exception:
        hub_ok = False
    real_import = builtins.__import__

    def no_sdk(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    assert release_layout._hub_accepts(repo) is hub_ok
