"""Stage public release bundles as read-only public data repositories."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import os
import shutil
import sys
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from conference_pipeline._security.private_markers import has_publishing_marker, redact_publishing_markers
from conference_pipeline.publishing.release_layout import github_excluded
from conference_pipeline.publishing.export_gate import GateClosed, no_secrets
from conference_pipeline.publishing.public_index import (
    PublicIndexError,
    sha256_file,
    validate_public_bundle_root,
)
from conference_pipeline.publishing.release_licenses import (
    ReleaseLicenseError,
    family_licenses,
    license_summary,
    license_text,
)


RECEIPT_TYPE = "conference_pipeline.public_repo_push"


class PublicRepoError(RuntimeError):
    """Raised when public repository staging or push gates fail closed."""


PUBLIC_REPO_DIRNAME = "public-repo"


@dataclass(frozen=True)
class PublicRepoConfig:
    bundle_root: Path
    target_dir: Path
    # None: "<conference>-data", from the bundle's RELEASE.json (v6.0 Phase 3, O14).
    repo_name: str | None = None
    github_repo: str | None = None
    dry_run: bool = True
    receipt_path: Path | None = None
    reviewed_dry_run_receipt: Path | None = None
    live: bool = False
    source_repo_root: Path = Path(".")
    secret_scanner: Callable[[list[Path]], list[dict]] | None = None
    github_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run


@dataclass(frozen=True)
class PublicRepoResult:
    target_dir: Path
    manifest_path: Path
    receipt_path: Path
    output_checksums: dict[str, str]
    public_safety_status: str


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PublicRepoError(f"JSON object expected: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def output_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def text_has_private_marker(text: str) -> bool:
    return has_publishing_marker(text)


def file_has_private_marker(path: Path, rel: str) -> bool:
    if has_publishing_marker(rel):
        return True
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return text_has_private_marker(text)


def redact_private_markers(text: str) -> str:
    return redact_publishing_markers(text)


def assert_public_safe_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if has_publishing_marker(rel):
            raise PublicRepoError(f"private marker found in staged path: {rel}")
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if text_has_private_marker(text):
            raise PublicRepoError(f"private marker found in staged file: {rel}")


def validate_stage_target(bundle_root: Path, target_dir: Path) -> Path:
    bundle = bundle_root.resolve()
    target = target_dir.resolve()
    unsafe = {bundle, bundle.parent, Path.cwd().resolve(), Path.home().resolve(), Path("/")}
    if target in unsafe or target in bundle.parents:
        raise PublicRepoError(f"refusing unsafe public repo target: {target}")
    if not target.name:
        raise PublicRepoError("target directory must be a named staging directory")
    return target


def is_managed_public_repo_target(path: Path) -> bool:
    manifest_path = path / "public_repo_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return False
    return manifest.get("read_only") is True and isinstance(manifest.get("bundle_id"), str)


def copy_bundle(bundle_root: Path, target_dir: Path) -> list[dict[str, str]]:
    target_resolved = validate_stage_target(bundle_root, target_dir)
    if target_resolved.exists():
        if not is_managed_public_repo_target(target_resolved):
            raise PublicRepoError(f"refusing to overwrite unmanaged public repo target: {target_resolved}")
        shutil.rmtree(target_resolved)
    # The bundle's own staged repository (<bundle>/public-repo/) is never copied into a
    # repository: re-staging into another directory would nest a copy of it (v6.0
    # Phase 3, IF7).
    own_repo = (bundle_root / PUBLIC_REPO_DIRNAME).resolve()
    sources = [
        source
        for source in sorted(bundle_root.rglob("*"), key=lambda item: item.relative_to(bundle_root).as_posix())
        if source.resolve() != target_resolved
        and target_resolved not in source.resolve().parents
        and source.resolve() != own_repo
        and own_repo not in source.resolve().parents
        # The site and the search index are published on their own surfaces (Pages, HF);
        # the repository holds the small documents only (v6.0 Phase 7.1, spec 6.1).
        and not github_excluded(source.relative_to(bundle_root).as_posix())
    ]
    target_resolved.mkdir(parents=True)
    excluded: list[dict[str, str]] = []
    for source in sources:
        rel = source.relative_to(bundle_root)
        destination = target_resolved / rel
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            rel_text = rel.as_posix()
            if file_has_private_marker(source, rel_text):
                excluded.append(
                    {
                        "path": redact_private_markers(rel_text),
                        "path_redacted": redact_private_markers(rel_text) != rel_text,
                        "reason": "private_marker",
                    }
                )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return excluded


def staged_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and ".git" not in path.relative_to(root).parts
    ]


def run_staged_secret_gate(
    root: Path,
    *,
    scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> dict[str, Any]:
    try:
        receipt = no_secrets(staged_files(root), _scanner=scanner)
    except GateClosed as exc:
        raise PublicRepoError(f"staged public repo secret scan failed: {exc.reason}") from exc
    return dataclasses.asdict(receipt)


def _blocked(reason: str, code: int | None) -> dict[str, Any]:
    return {
        "status": "operator_blocked",
        "exit_code": 2,
        "blocked_reason": reason,
        "provider_command_exit_code": code,
        "public_url": None,
        "commit_sha": None,
    }


def _release_bundle_id(text: str | None) -> str | None:
    try:
        value = json.loads(text or "").get("bundle_id")
    except (TypeError, ValueError, AttributeError):
        return None
    from conference_pipeline.publishing.release_layout import valid_tag

    return value if valid_tag(value) else None


def publish_github_repo(
    target_dir: Path,
    repo_name: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create or update the artifact-only GitHub repository through gh.

    An existing repository keeps its history (v6.0 Phase 7.1, spec section 6.6): the new
    version is committed on top of the remote's ``main``, and the version it replaces is
    tagged with its bundle id first when no such tag exists. Nothing is force-pushed.
    The new version is tagged with its own bundle id.
    """

    def run(*command: str) -> subprocess.CompletedProcess[str]:
        return runner(["git", "-C", str(target_dir), *command], capture_output=True, text=True, check=False)

    from conference_pipeline.publishing.surface_limits import check_tree

    if check_tree("github", target_dir):
        # Every GitHub publish meets the file limit, not only release publish (pr-49-r1 F3).
        return _blocked("github_surface_limit", None)
    release_file = target_dir / "RELEASE.json"
    new_id = _release_bundle_id(release_file.read_text(encoding="utf-8")) if release_file.is_file() else None
    if release_file.is_file() and new_id is None:
        # A release whose id is no git tag is refused before anything runs, not published
        # untagged (pr-49-r2 F2).
        return _blocked("github_release_id_invalid", None)
    init = run("init", "-b", "main")
    if init.returncode != 0:
        return _blocked("github_local_git_init_failed", init.returncode)
    view = runner(
        ["gh", "repo", "view", repo_name, "--json", "url"],
        capture_output=True,
        text=True,
        check=False,
    )
    exists = view.returncode == 0
    if exists:
        run("remote", "remove", "origin")
        for command in (
            ("remote", "add", "origin", f"https://github.com/{repo_name}.git"),
            ("fetch", "--force", "--tags", "origin", "main"),
            ("reset", "--soft", "FETCH_HEAD"),
        ):
            result = run(*command)
            if result.returncode != 0:
                return _blocked("github_provider_command_failed", result.returncode)
        def remote_tag(name: str) -> bool | None:
            # The remote, not this staged directory, says what is tagged: a retry after a
            # failed push finds its own local tags (phase-7.1-r2 F4).
            listed = run("ls-remote", "--tags", "origin", f"refs/tags/{name}")
            return None if listed.returncode != 0 else bool((listed.stdout or "").strip())

        if new_id is not None:
            present = remote_tag(new_id)
            if present is None:
                return _blocked("github_tag_lookup_failed", None)
            if present:
                # A bundle id names one set of bytes; the remote already has it as a tag
                # (phase-7.1-r1 F6).
                return _blocked("github_tag_exists", None)
        previous = run("show", "FETCH_HEAD:RELEASE.json")
        old_id = _release_bundle_id(previous.stdout) if previous.returncode == 0 else None
        if previous.returncode == 0 and old_id is None:
            return _blocked("github_previous_release_unreadable", None)
        old_present = remote_tag(old_id) if old_id is not None and old_id != new_id else True
        if old_present is None:
            return _blocked("github_tag_lookup_failed", None)
        if not old_present:
            tagged = run("tag", "-f", old_id, "FETCH_HEAD")
            if tagged.returncode != 0:
                return _blocked("github_tag_previous_failed", tagged.returncode)
            # The replaced version's tag reaches the remote before main moves, so a
            # failure here leaves the remote as it was (phase-7.1-r1 F7).
            pushed = run("push", "origin", f"refs/tags/{old_id}")
            if pushed.returncode != 0:
                return _blocked("github_tag_previous_push_failed", pushed.returncode)
    for command in (("add", "-A"), ("commit", "-m", "Publish public conference bundle")):
        result = run(*command)
        if result.returncode != 0 and not (
            command[0] == "commit" and "nothing to commit" in f"{result.stdout or ''}\n{result.stderr or ''}"
        ):
            return _blocked("github_local_git_commit_failed", result.returncode)
    if new_id is not None:
        tagged = run("tag", "-f", new_id, "HEAD")
        if tagged.returncode != 0:
            return _blocked("github_tag_release_failed", tagged.returncode)
    if not exists:
        result = runner(
            [
                "gh",
                "repo",
                "create",
                repo_name,
                "--public",
                "--source",
                str(target_dir),
                "--remote",
                "origin",
                "--push",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and new_id is not None:
            result = run("push", "origin", f"refs/tags/{new_id}")
    else:
        result = run("push", "origin", "HEAD:main")
        if result.returncode == 0 and new_id is not None:
            result = run("push", "origin", f"refs/tags/{new_id}")
    if result.returncode != 0:
        return _blocked("github_provider_command_failed", result.returncode)
    sha = runner(
        ["git", "-C", str(target_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit_sha = (sha.stdout or "").strip()
    if sha.returncode != 0 or not commit_sha or any(char not in "0123456789abcdef" for char in commit_sha.lower()):
        return {
            "status": "operator_blocked",
            "exit_code": 2,
            "blocked_reason": "github_commit_sha_unavailable",
            "provider_command_exit_code": sha.returncode,
            "public_url": None,
            "commit_sha": None,
        }
    public_url = None
    final_view = runner(
        ["gh", "repo", "view", repo_name, "--json", "url"],
        capture_output=True,
        text=True,
        check=False,
    )
    if final_view.returncode == 0:
        try:
            public_url = json.loads(final_view.stdout or "{}").get("url")
        except (TypeError, ValueError):
            public_url = None
    if not isinstance(public_url, str) or not public_url.startswith("https://github.com/"):
        return {
            "status": "operator_blocked",
            "exit_code": 2,
            "blocked_reason": "github_public_url_unavailable",
            "provider_command_exit_code": final_view.returncode,
            "public_url": None,
            "commit_sha": commit_sha,
        }
    return {
        "status": "succeeded",
        "exit_code": 0,
        "public_url": public_url,
        "commit_sha": commit_sha,
        "provider": "github",
    }




def readme_text(release: dict[str, Any], repo_name: str) -> str:
    bundle_id = release.get("bundle_id", "unknown")
    source_sha = str(release.get("source_sha", "unknown"))
    short_sha = source_sha[:12] if source_sha != "unknown" else source_sha
    public_repo_url = release.get("public_repo_url") or "not published"
    public_site_url = release.get("public_site_url") or "not published"
    cdn_base_url = release.get("cdn_base_url") or "not published"
    zenodo_doi = release.get("zenodo_doi") or "not published"
    zenodo_record_url = release.get("zenodo_record_url") or "not published"
    hf_dataset_url = release.get("hf_dataset_url") or "not published"
    return (
        f"# {repo_name}\n\n"
        f"Public data release for `{bundle_id}`.\n\n"
        "## Release Facts\n\n"
        f"- Bundle ID: `{bundle_id}`\n"
        f"- Source SHA: `{short_sha}`\n"
        f"- Licenses by family: {license_summary(release)}.\n"
        "- Repository policy: read-only public data mirror.\n"
        "- Live publication: operator-gated after dry-run receipts and staged public-tree safety checks pass.\n\n"
        "## Public Identifiers\n\n"
        f"- Public repository: {public_repo_url}\n"
        f"- Static site: {public_site_url}\n"
        f"- CDN base URL: {cdn_base_url}\n"
        f"- Zenodo DOI: {zenodo_doi}\n"
        f"- Zenodo record: {zenodo_record_url}\n"
        f"- Hugging Face dataset: {hf_dataset_url}\n\n"
        "## Contents\n\n"
        "- `RELEASE.json` and `CHECKSUMS.json` describe the exported bundle.\n"
        "- The search index, chunk texts and embeddings are in the Hugging Face dataset, and the reading site is the static site, both above.\n"
        "- `public_repo_manifest.json` lists staged files and checksums.\n"
        "- `.github/workflows/public-url-liveness.yml` runs public URL checks in CI.\n"
    )


def notice_text(release: dict[str, Any]) -> str:
    conference = release.get("conference", "conference")
    return (
        "# Notice\n\n"
        f"This release contains machine-assisted public artifacts for {conference}.\n"
        "Attribution to source papers and generated artifact provenance is preserved in release manifests.\n"
        "Raw private workspace files, provider secrets, Drive paths, and local cache state are excluded.\n"
    )


def citation_text(release: dict[str, Any]) -> str:
    bundle_id = release.get("bundle_id", "unknown")
    conference = release.get("conference", "Conference")
    year = str(datetime.now(UTC).year)
    text = (
        "cff-version: 1.2.0\n"
        f"title: \"{conference} public data release {bundle_id}\"\n"
        "message: \"If you use this dataset, please cite this public release.\"\n"
        "type: dataset\n"
        f"date-released: \"{datetime.now(UTC).date().isoformat()}\"\n"
        "authors:\n"
        "  - name: \"Conference Pipeline Team\"\n"
        f"version: \"{bundle_id}\"\n"
        f"year: {year}\n"
    )
    if release.get("public_repo_url"):
        text += f"url: \"{release['public_repo_url']}\"\n"
    if release.get("zenodo_doi"):
        text += f"doi: \"{release['zenodo_doi']}\"\n"
    return text


def workflow_text() -> str:
    return (
        "name: Public URL liveness\n\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "  push:\n"
        "    branches: [main]\n\n"
        "jobs:\n"
        "  public-url-liveness:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n"
        "          python-version: '3.11'\n"
        "      - name: Check public URLs\n"
        "        shell: python\n"
        "        run: |\n"
        "          import json\n"
        "          import re\n"
        "          import sys\n"
        "          import urllib.error\n"
        "          import urllib.request\n"
        "          from pathlib import Path\n"
        "          pattern = re.compile(r'(?:https?)://[^\\s\"\\'<>`]+')\n"
        "          urls = set()\n"
        "          for path in Path('.').rglob('*'):\n"
        "              if not path.is_file() or '.git' in path.parts:\n"
        "                  continue\n"
        "              try:\n"
        "                  text = path.read_text(encoding='utf-8')\n"
        "              except UnicodeDecodeError:\n"
        "                  continue\n"
        "              urls.update(url.rstrip('.,;:)]}`') for url in pattern.findall(text))\n"
        "          results = []\n"
        "          for url in sorted(urls):\n"
        "              request = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'conference-pipeline-public-liveness/1.0'})\n"
        "              try:\n"
        "                  with urllib.request.urlopen(request, timeout=15) as response:\n"
        "                      status = int(response.status)\n"
        "              except urllib.error.HTTPError as exc:\n"
        "                  status = int(exc.code)\n"
        "              except Exception:\n"
        "                  status = None\n"
        "              results.append({'url': url, 'status': 'passed' if status == 200 else 'failed', 'http_status': status})\n"
        "          failed = [item for item in results if item['status'] == 'failed']\n"
        "          Path('public-url-liveness-receipt.json').write_text(json.dumps({'status': 'failed' if failed else 'succeeded', 'checked_count': len(results), 'failed_count': len(failed), 'results': results, 'secret_values_recorded': False}, indent=2) + '\\n', encoding='utf-8')\n"
        "          sys.exit(1 if failed else 0)\n"
    )


REVIEW_CHECKSUM_EXCLUSIONS = {"public_repo_manifest.json", "CITATION.cff"}


def reviewable_checksums(checksums: dict[str, str]) -> dict[str, str]:
    return {
        path: digest
        for path, digest in checksums.items()
        if path not in REVIEW_CHECKSUM_EXCLUSIONS
    }


def verify_reviewed_receipt(config: PublicRepoConfig) -> dict[str, Any] | None:
    if not config.live:
        return None
    if config.reviewed_dry_run_receipt is None:
        raise PublicRepoError("--live requires --reviewed-dry-run-receipt")
    receipt = read_json(config.reviewed_dry_run_receipt)
    if receipt.get("receipt_type") != RECEIPT_TYPE or receipt.get("status") != "succeeded":
        raise PublicRepoError("reviewed dry-run receipt is not a successful public repo receipt")
    if receipt.get("dry_run") is not True or receipt.get("repo_name") != config.repo_name:
        raise PublicRepoError("reviewed dry-run receipt does not match public repo target")
    return receipt


def verify_reviewed_receipt_matches_current(
    receipt: dict[str, Any] | None,
    release: dict[str, Any],
    checksums: dict[str, str],
) -> None:
    if receipt is None:
        return
    if receipt.get("bundle_id") != release.get("bundle_id"):
        raise PublicRepoError("reviewed dry-run bundle_id does not match current bundle")
    if receipt.get("source_sha") != release.get("source_sha"):
        raise PublicRepoError("reviewed dry-run source_sha does not match current bundle")
    reviewed_checksums = receipt.get("output_checksums")
    if not isinstance(reviewed_checksums, dict) or not reviewed_checksums:
        raise PublicRepoError("reviewed dry-run receipt missing output checksums")
    reviewed = reviewable_checksums({str(path): str(digest) for path, digest in reviewed_checksums.items()})
    current = reviewable_checksums(checksums)
    if set(reviewed) != set(current):
        missing = sorted(set(reviewed) - set(current))
        extra = sorted(set(current) - set(reviewed))
        detail = ", ".join(
            part
            for part in (
                f"missing={missing[:5]}" if missing else "",
                f"extra={extra[:5]}" if extra else "",
            )
            if part
        )
        raise PublicRepoError(f"reviewed dry-run output checksum set mismatch: {detail}")
    for rel, reviewed_digest in reviewed.items():
        if current.get(rel) != reviewed_digest:
            raise PublicRepoError(f"reviewed dry-run output checksum mismatch: {rel}")


def default_repo_name(release: dict[str, Any]) -> str:
    """The data repository's default name, ``<conference>-data``."""
    conference = release.get("conference")
    if not isinstance(conference, str) or not conference.strip():
        raise PublicRepoError("RELEASE.json names no conference; pass --repo-name")
    return f"{conference.strip()}-data"


def stage_public_repo(config: PublicRepoConfig) -> PublicRepoResult:
    try:
        bundle_root = validate_public_bundle_root(config.bundle_root)
    except PublicIndexError as exc:
        raise PublicRepoError(str(exc)) from exc
    target_dir = config.target_dir.resolve()
    release = read_json(bundle_root / "RELEASE.json")
    try:
        family_licenses(release)
    except ReleaseLicenseError as exc:
        raise PublicRepoError(str(exc)) from exc
    if config.repo_name is None:
        config = dataclasses.replace(config, repo_name=default_repo_name(release))

    reviewed_receipt = verify_reviewed_receipt(config)
    public_safety = {
        "status": "clean",
        "scope": "staged_public_repo_tree",
        "private_marker_count": 0,
        "source_repo_history_published": False,
    }

    excluded_files = copy_bundle(bundle_root, target_dir)
    write_text(target_dir / "README.md", readme_text(release, config.repo_name))
    write_text(target_dir / "LICENSE", license_text(release))
    write_text(target_dir / "NOTICE.md", notice_text(release))
    write_text(target_dir / "CITATION.cff", citation_text(release))
    write_text(target_dir / ".github" / "workflows" / "public-url-liveness.yml", workflow_text())

    manifest_path = target_dir / "public_repo_manifest.json"
    relative_source = bundle_root.as_posix()
    if bundle_root.is_absolute():
        try:
            relative_source = bundle_root.relative_to(config.source_repo_root.resolve()).as_posix()
        except ValueError:
            relative_source = bundle_root.name
    manifest = {
        "schema_version": "1.0.0",
        "repo_name": config.repo_name,
        "read_only": True,
        "bundle_id": release.get("bundle_id"),
        "source_sha": release.get("source_sha"),
        "source_bundle_root": relative_source,
        "operator_live_step_required": True,
        "public_repo_url": release.get("public_repo_url"),
        "public_site_url": release.get("public_site_url"),
        "cdn_base_url": release.get("cdn_base_url"),
        "zenodo_doi": release.get("zenodo_doi"),
        "zenodo_record_url": release.get("zenodo_record_url"),
        "hf_dataset_url": release.get("hf_dataset_url"),
        "excluded_files": excluded_files,
        "license_families": family_licenses(release),
        "generated_at": utc_now(),
        "output_checksums": {},
    }
    write_json(manifest_path, manifest)
    assert_public_safe_tree(target_dir)
    checksums = output_checksums(target_dir)
    manifest["output_checksums"] = checksums
    write_json(manifest_path, manifest)
    assert_public_safe_tree(target_dir)
    secret_gate = run_staged_secret_gate(target_dir, scanner=config.secret_scanner)
    checksums = output_checksums(target_dir)
    verify_reviewed_receipt_matches_current(reviewed_receipt, release, checksums)

    receipt_path = config.receipt_path or target_dir / "public_repo_push_receipt.json"
    provider_result = (
        publish_github_repo(target_dir, config.github_repo or config.repo_name, runner=config.github_runner)
        if config.live
        else {"status": "succeeded", "exit_code": 0}
    )
    status = provider_result["status"]
    exit_code = provider_result["exit_code"]
    receipt = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "status": status,
        "exit_code": exit_code,
        "dry_run": config.dry_run,
        "live": config.live,
        "repo_name": config.repo_name,
        "bundle_id": release.get("bundle_id"),
        "source_sha": release.get("source_sha"),
        "target_dir": str(target_dir),
        "operator_live_step_required": True,
        "blocked_reason": provider_result.get("blocked_reason"),
        "provider_command_exit_code": provider_result.get("provider_command_exit_code"),
        "public_url": provider_result.get("public_url"),
        "commit_sha": provider_result.get("commit_sha"),
        "public_safety": public_safety,
        "secret_gate": secret_gate,
        "output_checksums": checksums,
        "excluded_files": excluded_files,
        "secret_values_recorded": False,
        "generated_at": utc_now(),
    }
    write_json(receipt_path, receipt)
    if config.live and status != "succeeded":
        raise PublicRepoError("live GitHub publication provider command failed")
    return PublicRepoResult(
        target_dir=target_dir,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        output_checksums=checksums,
        public_safety_status=str(public_safety.get("status", "clean")),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage and gate a public data repository")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--repo-name", default=None, help="default: <conference>-data from RELEASE.json")
    parser.add_argument("--github-repo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--reviewed-dry-run-receipt", type=Path)
    parser.add_argument("--live", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = stage_public_repo(
            PublicRepoConfig(
                bundle_root=args.bundle_root,
                target_dir=args.target_dir,
                repo_name=args.repo_name,
                github_repo=args.github_repo,
                dry_run=args.dry_run or not args.live,
                receipt_path=args.receipt_path,
                reviewed_dry_run_receipt=args.reviewed_dry_run_receipt,
                live=args.live,
                source_repo_root=Path.cwd(),
                github_runner=subprocess.run,
            )
        )
    except PublicRepoError as exc:
        print(str(exc), file=sys.stderr)
        return 2 if args.live else 1
    print(f"Public repo staged: {result.target_dir}")
    print(f"Receipt: {result.receipt_path}")
    print(f"Public safety: {result.public_safety_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
