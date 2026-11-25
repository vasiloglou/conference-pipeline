"""v6.0 Phase 7.1: the HF dataset carries the bundle's data tree, tagged per version (T9, T10)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from conference_pipeline.publishing.dataset_package import hf_dataset_card, publish_hf_dataset_live


def _bundle(root: Path, bundle_id: str = "x-v1.1.0", licenses: dict | None = None) -> Path:
    bundle = root / bundle_id
    files = {
        "RELEASE.json": json.dumps(
            {
                "bundle_id": bundle_id,
                "conference": "X 2026",
                "family_licenses": licenses or {"decks": "MIT", "reader_catalog": "CC-BY-4.0", "transcripts": "CC-BY-NC-ND-4.0"},
                "public_site_url": "https://x-data.pages.dev",
                "previous_version_doi": "10.5281/zenodo.1",
            }
        ),
        "CHECKSUMS.json": "{}",
        "metadata/datapackage.json": "{}",
        "catalog/papers.jsonl": '{"paper_id": "p1"}\n',
        "index/corpus_chunks.parquet": "pq",
        "index/embeddings.parquet": "pq",
        "index/embeddings.npy": "npy",
        "index/bm25_index.pkl": "pkl",
        "decks/a.md": "# A",
        "site/index.html": "<html></html>",
        "public-repo/README.md": "repo",
    }
    for rel, text in files.items():
        path = bundle / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return relist(bundle)


def relist(bundle: Path) -> Path:
    """List every exported file in CHECKSUMS.json and the index files in the index manifest."""
    import hashlib

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    exported = {
        p.relative_to(bundle).as_posix(): {"sha256": sha(p)}
        for p in sorted(bundle.rglob("*"))
        if p.is_file() and p.relative_to(bundle).parts[0] not in ("index", "metadata", "site", "public-repo")
        and p.name != "CHECKSUMS.json"
    }
    (bundle / "CHECKSUMS.json").write_text(json.dumps(exported), encoding="utf-8")
    index = {p.name: sha(p) for p in sorted((bundle / "index").glob("*")) if p.name != "index_manifest.json"}
    (bundle / "index" / "index_manifest.json").write_text(json.dumps({"output_checksums": index}), encoding="utf-8")
    return bundle


def _publish(bundle: Path, api: "FakeApi") -> dict:
    return publish_hf_dataset_live(
        metadata_dir=bundle / "metadata", repo_id="o/x-data", token="t", hf_api_factory=api, secret_scanner=lambda paths: []
    )


class FakeApi:
    def __init__(self, remote: dict[str, str] | None = None, tags: list[str] | None = None):
        self.remote = dict(remote or {})
        self.tags = list(tags or [])
        self.calls: list[tuple] = []
        self.downloads: Path | None = None

    def __call__(self, token: str) -> "FakeApi":
        return self

    def create_repo(self, repo_id, *, repo_type, exist_ok):
        self.calls.append(("create_repo", repo_id))

    def list_repo_files(self, repo_id, *, repo_type):
        return sorted(self.remote)

    def list_repo_refs(self, repo_id, *, repo_type):
        return type("Refs", (), {"tags": [type("Ref", (), {"name": n})() for n in self.tags]})()

    def hf_hub_download(self, *, repo_id, filename, repo_type):
        path = self.downloads / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.remote[filename], encoding="utf-8")
        return str(path)

    def create_tag(self, repo_id, *, tag, revision, repo_type):
        self.calls.append(("create_tag", tag, revision))
        self.tags.append(tag)

    def create_commit(self, *, repo_id, repo_type, operations, commit_message):
        self.calls.append(("create_commit", [type(op).__name__ + ":" + op.path_in_repo for op in operations]))
        return type("Info", (), {"oid": f"c{len(self.calls)}"})()


def _ops(api: FakeApi) -> list[str]:
    return [op for call in api.calls if call[0] == "create_commit" for op in call[1]]


def test_t9_the_upload_is_the_data_tree_with_a_card_and_a_tag(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    api = FakeApi(remote={"old/stale.json": "{}", ".gitattributes": "x"})
    result = _publish(bundle, api)
    ops = _ops(api)
    added = sorted(op.split(":", 1)[1] for op in ops if op.startswith("CommitOperationAdd"))
    assert added == sorted(
        [
            "CHECKSUMS.json", "LICENSE", "README.md", "RELEASE.json", "catalog/papers.jsonl", "decks/a.md",
            "index/corpus_chunks.parquet", "index/embeddings.parquet", "index/index_manifest.json",
            "metadata/datapackage.json",
        ]
    )
    # Dropped files are deleted; the Hub's own .gitattributes is kept.
    assert [op for op in ops if op.startswith("CommitOperationDelete")] == ["CommitOperationDelete:old/stale.json"]
    assert ("create_tag", "x-v1.1.0", result["commit_sha"]) in api.calls
    assert result["tag"] == "x-v1.1.0" and result["tagged_previous"] is None


def test_t9_large_trees_are_split_into_commits_of_at_most_100_operations(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    for number in range(250):
        (bundle / "decks" / f"d{number}.md").write_text("x", encoding="utf-8")
    relist(bundle)
    api = FakeApi()
    result = _publish(bundle, api)
    commits = [call[1] for call in api.calls if call[0] == "create_commit"]
    assert result["commit_count"] == len(commits) == 3
    assert all(len(ops) <= 100 for ops in commits)
    # The tag points at the last commit, which holds the whole tree.
    assert api.calls[-1] == ("create_tag", "x-v1.1.0", result["commit_sha"])


def test_t10_the_replaced_version_is_tagged_before_the_upload(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    api = FakeApi(remote={"RELEASE.json": json.dumps({"bundle_id": "x-v1.0.0"})})
    api.downloads = tmp_path / "dl"
    _publish(bundle, api)
    names = [call[0] for call in api.calls]
    first_tag = api.calls[names.index("create_tag")]
    assert first_tag == ("create_tag", "x-v1.0.0", "main")
    assert names.index("create_tag") < names.index("create_commit")


def test_t10_an_already_tagged_version_is_not_tagged_again(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    api = FakeApi(remote={"RELEASE.json": json.dumps({"bundle_id": "x-v1.0.0"})}, tags=["x-v1.0.0"])
    api.downloads = tmp_path / "dl"
    _publish(bundle, api)
    assert [call[1] for call in api.calls if call[0] == "create_tag"] == ["x-v1.1.0"]


def test_t9_the_card_parses_and_states_the_licence_per_d4() -> None:
    mixed = {"bundle_id": "x", "conference": "X", "family_licenses": {"a": "MIT", "b": "CC-BY-4.0"}}
    card = hf_dataset_card(mixed, ["catalog/papers.jsonl", "index/corpus_chunks.parquet"])
    front = yaml.safe_load(card.split("---")[1])
    assert front["license"] == "other" and front["license_name"] == "per-family" and front["license_link"] == "LICENSE"
    assert [c["config_name"] for c in front["configs"]] == ["catalog", "chunks"]
    single = hf_dataset_card({"bundle_id": "x", "family_licenses": {"a": "MIT"}}, [])
    assert yaml.safe_load(single.split("---")[1]) == {"pretty_name": "Conference public release x", "license": "mit"}
