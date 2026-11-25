"""Material-cluster, section-spine, stable-reference, and slide-break validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from conference_pipeline.presentation.editorial_contract import validate_storyboard_rows  # noqa: F401

CLUSTER_SCHEMA_VERSION = 1

_REQUIRED_SECTION_FIELDS = (
    "section_id",
    "section_thesis",
    "source_clusters",
    "slide_roles",
    "reference_set",
    "transition_purpose",
)


@dataclass(frozen=True)
class ClusterFinding:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class ClusterResult:
    status: str  # "pass" | "blocked"
    findings: tuple[ClusterFinding, ...]
    schema_version: int = CLUSTER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
        }


def validate_section_spine(sections: list[dict[str, Any]]) -> ClusterResult:
    if not isinstance(sections, list):
        return ClusterResult(
            status="blocked",
            findings=(_finding("section_spine_missing_field", "sections must be a list", "$"),),
        )
    findings: list[ClusterFinding] = []
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            findings.append(_finding("section_spine_missing_field", "section must be an object", f"sections[{i}]"))
            continue
        for field in _REQUIRED_SECTION_FIELDS:
            val = section.get(field)
            if val is None or (isinstance(val, (str, list)) and not val):
                findings.append(
                    _finding(
                        "section_spine_missing_field",
                        f"{field} is absent or empty",
                        f"sections[{i}].{field}",
                    )
                )
    status = "blocked" if findings else "pass"
    return ClusterResult(status=status, findings=tuple(findings))


def validate_stable_reference_ids(refs: list[dict[str, Any]]) -> ClusterResult:
    if not isinstance(refs, list):
        return ClusterResult(
            status="blocked",
            findings=(_finding("cluster_missing_stable_ref_id", "refs must be a list", "$"),),
        )
    findings: list[ClusterFinding] = []
    seen: set[str] = set()
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict) or "stable_ref_id" not in ref:
            findings.append(
                _finding("cluster_missing_stable_ref_id", "stable_ref_id key is absent", f"refs[{i}]")
            )
            continue
        ref_id = ref["stable_ref_id"]
        if ref_id in seen:
            findings.append(
                _finding(
                    "cluster_duplicate_stable_ref_id",
                    f"duplicate stable_ref_id: {ref_id!r}",
                    f"refs[{i}].stable_ref_id",
                )
            )
        else:
            seen.add(ref_id)
    status = "blocked" if findings else "pass"
    return ClusterResult(status=status, findings=tuple(findings))


def validate_orphaned_references(
    refs: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
) -> ClusterResult:
    if not isinstance(refs, list) or not isinstance(clusters, list):
        return ClusterResult(
            status="blocked",
            findings=(_finding("cluster_orphaned_reference", "refs and clusters must be lists", "$"),),
        )
    claimed: set[str] = set()
    for cluster in clusters:
        if isinstance(cluster, dict):
            for ref_id in cluster.get("reference_set", []):
                if isinstance(ref_id, str):
                    claimed.add(ref_id)
    findings: list[ClusterFinding] = []
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        ref_id = ref.get("stable_ref_id")
        if ref_id is not None and ref_id not in claimed:
            findings.append(
                _finding(
                    "cluster_orphaned_reference",
                    f"stable_ref_id {ref_id!r} not found in any cluster reference_set",
                    f"refs[{i}].stable_ref_id",
                )
            )
    status = "blocked" if findings else "pass"
    return ClusterResult(status=status, findings=tuple(findings))


def validate_missing_section_membership(
    clusters: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> ClusterResult:
    if not isinstance(clusters, list) or not isinstance(sections, list):
        return ClusterResult(
            status="blocked",
            findings=(
                _finding("cluster_missing_section_membership", "clusters and sections must be lists", "$"),
            ),
        )
    section_ids: set[str] = {
        s.get("section_id") for s in sections if isinstance(s, dict) and s.get("section_id")
    }
    findings: list[ClusterFinding] = []
    for i, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            continue
        sec_id = cluster.get("section_id")
        if sec_id not in section_ids:
            findings.append(
                _finding(
                    "cluster_missing_section_membership",
                    f"cluster section_id {sec_id!r} not found in sections list",
                    f"clusters[{i}].section_id",
                )
            )
    status = "blocked" if findings else "pass"
    return ClusterResult(status=status, findings=tuple(findings))


def validate_cluster_to_slide_breaks(clusters: list[dict[str, Any]]) -> ClusterResult:
    if not isinstance(clusters, list):
        return ClusterResult(
            status="blocked",
            findings=(_finding("cluster_missing_slide_break", "clusters must be a list", "$"),),
        )
    findings: list[ClusterFinding] = []
    for i, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            continue
        is_multi = cluster.get("multi_slide") is True or len(cluster.get("slides", [])) > 1
        if is_multi and cluster.get("slide_boundary") is None:
            findings.append(
                _finding(
                    "cluster_missing_slide_break",
                    f"cluster {cluster.get('cluster_id', i)!r} spans multiple slides but has no slide_boundary",
                    f"clusters[{i}].slide_boundary",
                )
            )
    status = "blocked" if findings else "pass"
    return ClusterResult(status=status, findings=tuple(findings))


def validate_material_clusters(clusters: list[dict[str, Any]]) -> ClusterResult:
    if not isinstance(clusters, list):
        return ClusterResult(
            status="blocked",
            findings=(_finding("cluster_missing_slide_break", "clusters must be a list", "$"),),
        )
    findings: list[ClusterFinding] = []
    findings.extend(validate_cluster_to_slide_breaks(clusters).findings)
    for i, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            continue
        contributions = cluster.get("contributions")
        if not isinstance(contributions, list) or not contributions:
            findings.append(
                _finding(
                    "cluster_missing_slide_break",
                    f"cluster {cluster.get('cluster_id', i)!r} has empty or missing contributions",
                    f"clusters[{i}].contributions",
                )
            )
    status = "blocked" if findings else "pass"
    return ClusterResult(status=status, findings=tuple(findings))


def _finding(code: str, message: str, path: str) -> ClusterFinding:
    return ClusterFinding(code=code, message=message, path=path)
