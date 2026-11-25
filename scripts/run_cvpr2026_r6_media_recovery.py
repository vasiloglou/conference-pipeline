#!/usr/bin/env python3
"""Capability-sealed CLI shell for the fixed CVPR 2026 r6 media owner.

The runner intentionally contains no conference-pipeline imports, owner
helpers, receipt constructors, or mutable continuation facts.  The fixed
owner is imported only when this exact CLI boundary is invoked; it owns all
admission and every later capability in its private implementation module.
"""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import stat
import sys

try:
    import _cvpr2026_r6_successor_admission as _legacy_admission
except ModuleNotFoundError:
    import importlib.util as _importlib_util
    _spec = _importlib_util.spec_from_file_location("_cvpr2026_r6_successor_admission", pathlib.Path(__file__).with_name("_cvpr2026_r6_successor_admission.py"))
    _legacy_admission = _importlib_util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_legacy_admission)



def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    """Run the reviewed, fixed r6 CLI surface.

    This function deliberately makes no claim that Python caller provenance is
    an authorization boundary.  Its concrete boundary is instead its exact
    argv policy plus the descriptor-safe immutable control records validated
    below.  Tests may invoke this public CLI seam directly with disposable
    filesystem fixtures.
    """

    # Test-only observation is intentionally opt-in and process-local.  It is
    # not a runner API, is never persisted, and sees only the finite decision
    # fields below.  Production has no observer installed.
    def observe(phase: str, reason_code: str, exit_code: int, binding_state: str) -> None:
        observer = getattr(sys, "_cvpr2026_r6_cli_decision_observer", None)
        if observer is not None:
            observer((phase, reason_code, exit_code, binding_state))

    dispositions = {
        "owner_import_failed": ("pre_admission", 1),
        "owner_initialization_failed": ("pre_admission", 1),
        "worker_interrupted": ("supervisor", 2),
    }
    canonical_name = "cli_decision.json"
    quarantine_name = "cli_decision.quarantine.json"
    successor_root_name = "decision-successor"
    successor_archive_name = "archive"
    successor_admission_name = "admission.json"
    successor_claim_name = "claim.json"
    successor_run_id = "cvpr2026-live-production-r6-expanded-media"
    control_names = {canonical_name, quarantine_name, successor_root_name}
    # These fixed sibling names are owned and independently validated by the
    # private r6 owner.  They are not successor authority, but their expected
    # presence must not make the public control namespace unusable.  Unknown
    # siblings remain terminal so this is never an arbitrary-name allowance.
    owner_artifact_names = {
        "recovery_inventory.json",
        "audio_recovery_receipt.json",
        "audio-recovery-archive",
        "material_discovery_receipt.json",
        "material_acquisition_receipt.json",
        "cost_estimate_receipt.json",
        "benchmark_receipt.json",
        "provider_receipts.json",
        "budget_ledger.json",
        "r6_processing_inventory.json",
        "material-payloads",
        "raw-audio",
        "speech-text",
        "paper-text",
        "slide-ocr",
        "auth-pages",
        "logs",
        "session",
    }
    permitted_names = control_names | owner_artifact_names
    # Do not resolve this path: resolving would follow an attacker-controlled
    # script-path link before the descriptor-rooted walk below can reject it.
    root = pathlib.Path(__file__).parent.parent
    relative_parts = (
        "conferences", "cvpr2026", "storage", "transcription-ocr",
        "cvpr2026", "r6",
    )

    def closed_envelope(raw: bytes) -> bool:
        """Accept only one exact, self-binding unavailable decision envelope."""

        try:
            text = raw.decode("utf-8")
            document = json.loads(text)
            if type(document) is not dict or set(document) != {
                "schema_version", "receipt_type", "disposition",
                "unavailable_decision", "unavailable_decision_sha256",
            }:
                return False
            if (
                type(document["schema_version"]) is not int
                or type(document["receipt_type"]) is not str
                or type(document["disposition"]) is not str
                or type(document["unavailable_decision"]) is not dict
                or type(document["unavailable_decision_sha256"]) is not str
            ):
                return False
            digest = document["unavailable_decision_sha256"]
            unavailable = document["unavailable_decision"]
            if set(unavailable) != {
                "schema_version", "receipt_type", "phase", "reason_code", "exit_code",
                "binding_state", "recovery_inventory_sha256", "audio_policy_digest",
                "source_scope_sha256", "record_count", "checkpoint_sha256", "lineage_state",
                "historical_checkpoint_lineage_sha256", "historical_checkpoint_archive_sha256",
            }:
                return False
            required_text = ("receipt_type", "phase", "reason_code", "binding_state")
            nullable_fields = (
                "recovery_inventory_sha256", "audio_policy_digest", "source_scope_sha256", "record_count",
                "checkpoint_sha256", "lineage_state", "historical_checkpoint_lineage_sha256",
                "historical_checkpoint_archive_sha256",
            )
            if (
                type(unavailable["schema_version"]) is not int
                or type(unavailable["exit_code"]) is not int
                or any(type(unavailable[key]) is not str for key in required_text)
                or any(unavailable[key] is not None for key in nullable_fields)
            ):
                return False
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or document["schema_version"] != 2
                or document["receipt_type"] != "conference_pipeline.cvpr2026_r6_cli_decision"
                or document["disposition"] != "unavailable"
            ):
                return False
            if (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8") != raw:
                return False
            reason = unavailable["reason_code"]
            if reason not in dispositions or unavailable["phase"] != dispositions[reason][0] or unavailable["exit_code"] != dispositions[reason][1]:
                return False
            if unavailable["schema_version"] != 1 or unavailable["receipt_type"] != "conference_pipeline.cvpr2026_r6_cli_decision" or unavailable["binding_state"] != "unavailable":
                return False
            canonical_unavailable = (json.dumps(unavailable, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            return digest == hashlib.sha256(canonical_unavailable).hexdigest()
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return False

    def legacy_unbound_terminal_decision(raw: bytes) -> bool:
        """Recognize only the audited, non-authorizing schema-v1 terminal leaf.

        This is deliberately not a compatibility migration.  The returned
        value supplies one finite observer diagnosis for the present legacy
        shape, but never supplies predecessor binding, successor eligibility,
        or any authority to reach the owner.
        """

        try:
            document = json.loads(raw.decode("utf-8"))
            binding_fields = (
                "recovery_inventory_sha256", "audio_policy_digest", "source_scope_sha256",
                "record_count", "checkpoint_sha256", "lineage_state",
                "historical_checkpoint_lineage_sha256",
                "historical_checkpoint_archive_sha256",
            )
            if type(document) is not dict or set(document) != {
                "schema_version", "receipt_type", "phase", "reason_code", "exit_code",
                "binding_state", *binding_fields,
            }:
                return False
            if (
                type(document["schema_version"]) is not int
                or type(document["receipt_type"]) is not str
                or type(document["phase"]) is not str
                or type(document["reason_code"]) is not str
                or type(document["exit_code"]) is not int
                or type(document["binding_state"]) is not str
                or any(document[field] is not None for field in binding_fields)
            ):
                return False
            return (
                document["schema_version"] == 1
                and document["receipt_type"] == "conference_pipeline.cvpr2026_r6_cli_decision"
                and document["phase"] == "pre_admission"
                and document["reason_code"] == "admission_invalid"
                and document["exit_code"] == 1
                and document["binding_state"] == "unavailable"
                and (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8") == raw
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return False

    def with_decision_directory(*, create: bool = True) -> tuple[int | None, bool]:
        """Open the fixed path component-by-component without following links."""

        descriptor: int | None = None
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            descriptor = os.open(root, flags)
            for part in relative_parts:
                try:
                    if create:
                        os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
                try:
                    os.close(descriptor)
                except (OSError, TypeError, ValueError):
                    try:
                        os.close(child)
                    except (OSError, TypeError, ValueError):
                        pass
                    return None, False
                descriptor = child
            return descriptor, True
        except (OSError, TypeError, ValueError):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except (OSError, TypeError, ValueError):
                    pass
            return None, False

    def close_fd(descriptor: int | None) -> bool:
        if descriptor is None:
            return True
        try:
            os.close(descriptor)
        except (OSError, TypeError, ValueError):
            return False
        return True

    def read_closed_raw_leaf(directory: int, name: str) -> bytes | None:
        """Read exactly one regular no-follow fixed leaf and validate all bytes."""

        descriptor: int | None = None
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 1:
                return False
            chunks: list[bytes] = []
            remaining = metadata.st_size + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            return raw if len(raw) == metadata.st_size else None
        except (OSError, TypeError, ValueError):
            return None
        finally:
            close_fd(descriptor)

    def read_closed_leaf(directory: int, name: str) -> bool:
        raw = read_closed_raw_leaf(directory, name)
        return raw is not None and closed_envelope(raw)

    def filesystem_identity(metadata: os.stat_result, *, directory: bool) -> tuple[int, int, int] | None:
        """Return the exact no-follow device/inode/type tuple we can retain."""

        required = stat.S_ISDIR if directory else stat.S_ISREG
        if not required(metadata.st_mode):
            return None
        return (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))

    def read_exact_descriptor(descriptor: int, size: int) -> bytes | None:
        """Read a retained regular descriptor once, rejecting size drift."""

        if size < 1:
            return None
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = size + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            return raw if len(raw) == size else None
        except (OSError, TypeError, ValueError):
            return None

    def capture_predecessor_binding(directory: int) -> tuple[tuple[int, int, int], tuple[int, int, int], bytes, str, int] | None:
        """Retain one canonical no-follow parent/leaf identity and byte binding."""

        leaf: int | None = None
        retained = False
        try:
            parent_identity = filesystem_identity(os.fstat(directory), directory=True)
            if parent_identity is None:
                return None
            leaf = os.open(canonical_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            leaf_stat = os.fstat(leaf)
            leaf_identity = filesystem_identity(leaf_stat, directory=False)
            raw = read_exact_descriptor(leaf, leaf_stat.st_size)
            if leaf_identity is None or raw is None or not (closed_envelope(raw) or _legacy_admission.is_exact_legacy_v1(raw)):
                return None
            retained = True
            return parent_identity, leaf_identity, raw, hashlib.sha256(raw).hexdigest(), leaf
        except (OSError, TypeError, ValueError):
            return None
        finally:
            # A successful binding retains its leaf descriptor for the entire
            # transition.  Every failed capture owns and closes its descriptor.
            if leaf is not None and not retained:
                close_fd(leaf)

    def revalidate_predecessor_binding(
        directory: int,
        binding: tuple[tuple[int, int, int], tuple[int, int, int], bytes, str, int],
    ) -> bool:
        """Prove the fixed pathname still names the retained parent and leaf.

        The held descriptors alone are not enough: a rename can leave them
        valid while the canonical path names an attacker-controlled A→B
        replacement.  Reopening the fixed path with no-follow semantics and
        comparing both identities plus bytes catches that replacement before
        each irreversible successor publication.
        """

        parent_identity, leaf_identity, expected_raw, expected_sha256, _held_leaf = binding
        current: int | None = None
        fresh_leaf: int | None = None
        try:
            if filesystem_identity(os.fstat(directory), directory=True) != parent_identity:
                return False
            current, opened = with_decision_directory(create=False)
            if not opened or current is None:
                return False
            if filesystem_identity(os.fstat(current), directory=True) != parent_identity:
                return False
            fresh_leaf = os.open(canonical_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
            fresh_stat = os.fstat(fresh_leaf)
            if filesystem_identity(fresh_stat, directory=False) != leaf_identity:
                return False
            fresh_raw = read_exact_descriptor(fresh_leaf, fresh_stat.st_size)
            if fresh_raw != expected_raw or hashlib.sha256(fresh_raw or b"").hexdigest() != expected_sha256:
                return False
            # One final no-follow directory-entry identity check rejects a
            # leaf replacement that happened while its bytes were read.
            entry = os.lstat(canonical_name, dir_fd=current)
            return filesystem_identity(entry, directory=False) == leaf_identity
        except (OSError, TypeError, ValueError):
            return False
        finally:
            close_fd(fresh_leaf)
            close_fd(current)

    def open_closed_directory(parent: int, name: str, *, create: bool) -> int | None:
        """Open one fixed child directory without granting path authority."""

        try:
            try:
                metadata = os.lstat(name, dir_fd=parent)
            except FileNotFoundError:
                if not create:
                    return None
                os.mkdir(name, 0o700, dir_fd=parent)
                metadata = os.lstat(name, dir_fd=parent)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                return None
            descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode) or (
                opened.st_dev, opened.st_ino
            ) != (metadata.st_dev, metadata.st_ino):
                close_fd(descriptor)
                return None
            return descriptor
        except (OSError, TypeError, ValueError):
            return None

    def closed_json(payload: dict[str, object]) -> bytes:
        return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")

    def read_control_snapshot(directory: int, name: str) -> tuple[bytes, str, tuple[int, int, int]] | None:
        """Retain bytes/digest/no-follow identity without retaining a pathname."""

        descriptor: int | None = None
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            metadata = os.fstat(descriptor)
            identity = filesystem_identity(metadata, directory=False)
            raw = read_exact_descriptor(descriptor, metadata.st_size)
            if identity is None or raw is None:
                return None
            return raw, hashlib.sha256(raw).hexdigest(), identity
        except (OSError, TypeError, ValueError):
            return None
        finally:
            close_fd(descriptor)

    def write_closed_immutable_leaf(directory: int, name: str, raw: bytes) -> bool:
        """O_EXCL-publish and read back one fixed finite control leaf."""

        descriptor: int | None = None
        try:
            if not name or "/" in name or "\\" in name or "\x00" in name:
                return False
            if os.lstat(name, dir_fd=directory) is not None:
                return False
        except FileNotFoundError:
            pass
        except (OSError, TypeError, ValueError):
            return False
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return False
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    return False
                view = view[written:]
            os.fsync(descriptor)
            if not close_fd(descriptor):
                descriptor = None
                return False
            descriptor = None
            os.fsync(directory)
            return read_closed_raw_leaf(directory, name) == raw
        except MemoryError:
            raise
        except (OSError, TypeError, ValueError):
            return False
        finally:
            close_fd(descriptor)

    def successor_namespace_is_empty(directory: int) -> bool:
        """Allow only an empty fixed successor root before the one transition."""

        successor = open_closed_directory(directory, successor_root_name, create=False)
        if successor is None:
            return successor_root_name not in os.listdir(directory)
        try:
            return os.listdir(successor) == []
        except (OSError, TypeError, ValueError):
            return False
        finally:
            close_fd(successor)

    def create_successor_transition(
        directory: int,
        binding: tuple[tuple[int, int, int], tuple[int, int, int], bytes, str, int],
    ) -> bool:
        """Seal the sole archive/admission/claim lineage before owner import.

        This deliberately returns only a finite success value.  The runner
        never transfers predecessor bytes, a witness, or any authority object
        to the owner; the sealed successor controls are the complete durable
        boundary for the later fixed invocation.
        """

        _parent_identity, _leaf_identity, predecessor_raw, predecessor_sha256, _held_leaf = binding
        successor: int | None = None
        archive: int | None = None
        try:
            if not revalidate_predecessor_binding(directory, binding):
                return None
            successor = open_closed_directory(directory, successor_root_name, create=True)
            if successor is None or os.listdir(successor) != []:
                return None
            archive = open_closed_directory(successor, successor_archive_name, create=True)
            if archive is None or os.listdir(archive) != []:
                return None
            archive_leaf = predecessor_sha256 + ".json"
            # Directory creation is not the archive publication.  Recheck
            # again after it so a replacement during namespace setup cannot
            # cross the first O_EXCL data boundary.
            if not revalidate_predecessor_binding(directory, binding):
                return None
            if not write_closed_immutable_leaf(archive, archive_leaf, predecessor_raw):
                return None
            if read_closed_raw_leaf(archive, archive_leaf) != predecessor_raw:
                return None
            admission = {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.cvpr2026_r6_successor_admission",
                "r6_run_id": successor_run_id,
                "predecessor_relation": "immutable_cli_decision_successor",
                "predecessor_sha256": predecessor_sha256,
                "archive_name": archive_leaf,
                "admission_schema_version": 1,
            }
            admission_raw = closed_json(admission)
            if not revalidate_predecessor_binding(directory, binding):
                return None
            if not write_closed_immutable_leaf(successor, successor_admission_name, admission_raw):
                return None
            if read_closed_raw_leaf(successor, successor_admission_name) != admission_raw:
                return None
            claim = {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.cvpr2026_r6_successor_claim",
                "r6_run_id": successor_run_id,
                "predecessor_relation": "immutable_cli_decision_successor",
                "predecessor_sha256": predecessor_sha256,
                "admission_schema_version": 1,
                "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
            }
            claim_raw = closed_json(claim)
            if not revalidate_predecessor_binding(directory, binding):
                return None
            if not write_closed_immutable_leaf(successor, successor_claim_name, claim_raw):
                return None
            archive_snapshot = read_control_snapshot(archive, archive_leaf)
            admission_snapshot = read_control_snapshot(successor, successor_admission_name)
            claim_snapshot = read_control_snapshot(successor, successor_claim_name)
            predecessor_snapshot = (predecessor_raw, predecessor_sha256, _leaf_identity)
            if (
                archive_snapshot is None
                or admission_snapshot is None
                or claim_snapshot is None
                or not revalidate_predecessor_binding(directory, binding)
            ):
                return None
            # The successor records are self-consistent at publication time;
            # the owner will repeat this check using *only* these retained
            # bytes.  The post-publication path namespace is never an input.
            try:
                archive_document = json.loads(archive_snapshot[0].decode("utf-8"))
                admission_document = json.loads(admission_snapshot[0].decode("utf-8"))
                claim_document = json.loads(claim_snapshot[0].decode("utf-8"))
                if (
                    archive_snapshot[0] != predecessor_raw
                    or not isinstance(archive_document, dict)
                    or not isinstance(admission_document, dict)
                    or not isinstance(claim_document, dict)
                    or admission_document.get("predecessor_sha256") != predecessor_sha256
                    or admission_document.get("archive_name") != archive_leaf
                    or claim_document.get("predecessor_sha256") != predecessor_sha256
                    or claim_document.get("admission_sha256") != admission_snapshot[1]
                ):
                    return None
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                return None
            return True
        except MemoryError:
            raise
        except (OSError, TypeError, ValueError):
            return False
        finally:
            close_fd(archive)
            close_fd(successor)

    def admit_fixed_namespace() -> tuple[str, object | None]:
        """Return absent, pristine predecessor, closed, or invalid before import."""

        directory, opened = with_decision_directory()
        if not opened or directory is None:
            return "invalid", None
        retained_directory = False
        try:
            names = os.listdir(directory)
            if any(name not in permitted_names for name in names):
                return "invalid", None
            present = set(names).intersection(control_names)
            if not present:
                return "absent", None
            if present in ({canonical_name}, {canonical_name, successor_root_name}):
                binding = capture_predecessor_binding(directory)
                if binding is not None and successor_namespace_is_empty(directory):
                    retained_directory = True
                    return "pristine", (directory, binding)
                if binding is not None:
                    close_fd(binding[4])
                return "invalid", None
            if present == {quarantine_name} and read_closed_leaf(directory, quarantine_name):
                try:
                    os.replace(quarantine_name, canonical_name, src_dir_fd=directory, dst_dir_fd=directory)
                    os.fsync(directory)
                except (OSError, TypeError, ValueError):
                    # A completed replacement remains a closed canonical state;
                    # failure before it remains finite and never reaches owner.
                    return ("closed", None) if read_closed_leaf(directory, canonical_name) else ("invalid", None)
                return ("closed", None) if read_closed_leaf(directory, canonical_name) else ("invalid", None)
            return "invalid", None
        except (OSError, TypeError, ValueError):
            return "invalid", None
        finally:
            if not retained_directory:
                close_fd(directory)

    def publish_unavailable(reason_code: str) -> bool:
        """Write the one pre-owner decision without importing project code.

        This intentionally lives inside ``main``: before the owner is
        importable the public shell has no project import graph that can be
        trusted to construct a receipt.  The path, schema, and finite reason
        mapping are all fixed here; no caller can supply operational data.
        """

        state, _predecessor = admit_fixed_namespace()
        if state == "canonical":
            return True
        if state != "absent":
            return False
        phase, exit_code = dispositions[reason_code]
        payload = {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
            "phase": phase,
            "reason_code": reason_code,
            "exit_code": exit_code,
            "binding_state": "unavailable",
            "recovery_inventory_sha256": None,
            "audio_policy_digest": None,
            "source_scope_sha256": None,
            "record_count": None,
            "checkpoint_sha256": None,
            "lineage_state": None,
            "historical_checkpoint_lineage_sha256": None,
            "historical_checkpoint_archive_sha256": None,
        }
        # Descriptor slots are deliberately retired *before* close is
        # attempted.  Apart from avoiding a double-close in ``finally``, this
        # makes a close failure just another finite publication failure: no
        # later cleanup branch can turn it into a raw exception at the public
        # CLI boundary.
        descriptors: dict[str, int | None] = {
            "directory": None,
            "child": None,
            "output": None,
        }

        def close_slot(slot: str) -> bool:
            descriptor = descriptors[slot]
            descriptors[slot] = None
            if descriptor is None:
                return True
            try:
                os.close(descriptor)
            except (OSError, TypeError, ValueError):
                return False
            return True

        durably_published = False
        try:
            descriptors["directory"], opened = with_decision_directory()
            if not opened or descriptors["directory"] is None:
                return False
            unavailable_raw = (
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8")
            # Both fixed names carry the same closed decision envelope.  The
            # filename is the publication state: quarantine is a valid but
            # not-yet-promoted unavailable decision; canonical is the exact
            # same byte sequence after the atomic name transition.  This
            # permits a single fixed O_EXCL source leaf without ever exposing
            # a PID/object/caller-derived temporary name.
            decision = {
                "schema_version": 2,
                "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
                "disposition": "unavailable",
                "unavailable_decision": payload,
                "unavailable_decision_sha256": hashlib.sha256(unavailable_raw).hexdigest(),
            }
            raw = (json.dumps(decision, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
                "utf-8"
            )

            descriptors["output"] = os.open(
                quarantine_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptors["directory"],
            )
            view = memoryview(raw)
            while view:
                written = os.write(descriptors["output"], view)
                if written <= 0:
                    raise OSError("fixed unavailable decision write failed")
                view = view[written:]
            os.fsync(descriptors["output"])
            if not close_slot("output"):
                return False
            os.replace(
                quarantine_name,
                canonical_name,
                src_dir_fd=descriptors["directory"],
                dst_dir_fd=descriptors["directory"],
            )
            durably_published = True
            os.fsync(descriptors["directory"])
            return True
        except MemoryError:
            raise
        except (OSError, TypeError, ValueError):
            # Ordinary local publication trouble is intentionally opaque at
            # this boundary.  The finite process disposition remains nonzero.
            # ``replace`` is the durable replacement point.  Later ordinary
            # finalization trouble cannot invalidate that closed fixed leaf.
            return durably_published
        finally:
            # Do not let finalization alter the finite route.  Each helper
            # call retires its descriptor before attempting close, including
            # the first one that fails.
            close_slot("output")
            close_slot("child")
            close_slot("directory")

    args = list(sys.argv[1:] if argv is None else argv)
    if args not in ([], ["--live"]):
        observe("cli", "invalid_cli", 2, "unavailable")
        raise SystemExit(2)

    # The entire fixed namespace is admitted before importing the owner.  A
    # prior finite decision is closed; a malformed or mixed tree is equally
    # closed but deliberately never repaired or described at this boundary.
    admission, predecessor = admit_fixed_namespace()
    transition_binding: tuple[tuple[int, int, int], tuple[int, int, int], bytes, str, int] | None = None
    transition_directory: int | None = None
    successor_claimed = False
    if admission == "pristine":
        if type(predecessor) is tuple and len(predecessor) == 2:
            transition_directory, transition_binding = predecessor
        if transition_directory is not None and transition_binding is not None:
            successor_claimed = create_successor_transition(
                transition_directory, transition_binding
            )
        if successor_claimed:
            admission = "successor_claimed"
        else:
            admission = "invalid"
    if admission not in {"absent", "successor_claimed"}:
        reason = {
            "closed": "prior_unavailable_decision",
            "legacy_unbound": "legacy_unbound_terminal_decision",
        }.get(admission, "admission_invalid")
        observe("pre_admission", reason, 1, "unavailable")
        if transition_binding is not None:
            close_fd(transition_binding[4])
        close_fd(transition_directory)
        return 1

    def unavailable_owner_outcome(reason_code: str) -> int:
        publish_unavailable(reason_code)
        phase, exit_code = {
            "owner_import_failed": ("pre_admission", 1),
            "owner_initialization_failed": ("pre_admission", 1),
            "worker_interrupted": ("supervisor", 2),
        }[reason_code]
        observe(phase, reason_code, exit_code, "unavailable")
        return exit_code

    try:
        # Successor controls were sealed before import.  The owner receives
        # no predecessor bytes, witness, or caller-supplied authority object.
        if admission == "successor_claimed" and not successor_claimed:
            observe("pre_admission", "admission_invalid", 1, "unavailable")
            return 1
        from conference_pipeline.acquisition import _cvpr_r6_media_owner_impl as owner
    except ImportError:
        return unavailable_owner_outcome("owner_import_failed")
    except Exception:
        return unavailable_owner_outcome("owner_initialization_failed")
    except KeyboardInterrupt:
        return unavailable_owner_outcome("worker_interrupted")
    except SystemExit:
        return unavailable_owner_outcome("worker_interrupted")
    finally:
        if transition_binding is not None:
            close_fd(transition_binding[4])
        close_fd(transition_directory)

    expected_owner_failure = getattr(owner, "R6OwnerExpectedFailure", ())
    expected_disposition = getattr(owner, "R6OwnerDisposition", ())
    expected_incomplete = getattr(owner, "R6RecoveryIncompleteError", ())
    try:
        # The only owner invocation has a fixed, capability-free shape.
        # The runner has already sealed successor controls above; no direct
        # owner call can receive or manufacture authority through this API.
        # This bridge has no caller-controlled parameters.  Its safety comes
        # from the sealed descriptor-bound control records, not a claim that
        # same-process Python callers or frames are unforgeable.
        outcome = owner.run_fixed_r6_media_owner(live=args == ["--live"])
        observe("cli", "completed", outcome, "unavailable")
        return outcome
    except KeyboardInterrupt:
        # The owner has already published its single descriptor-bound result
        # when it was reachable.  The shell only exposes the stable outcome.
        observe("supervisor", "worker_interrupted", 2, "unavailable")
        return 2
    except SystemExit:
        observe("supervisor", "worker_interrupted", 2, "unavailable")
        return 2
    except expected_incomplete as exc:
        # The fixed owner has already emitted a descriptor-bound, sanitized
        # reason receipt.  Preserve a stable nonzero process outcome without
        # echoing exception text or receipt internals.
        if exc.reason_code not in {
            "worker_no_progress",
            "continuation_bound_exhausted",
            "worker_interrupted",
        }:
            raise RuntimeError("invalid fixed-owner incomplete outcome") from exc
        observe("supervisor", exc.reason_code, 2, "reachable")
        return 2
    except expected_owner_failure as exc:
        # The owner—not exception text—owns the complete finite public
        # taxonomy.  A malformed/mutable disposition is a programming fault,
        # intentionally visible instead of being reclassified by this shell.
        disposition = exc.disposition
        allowed = {
            ("workspace_root", "root_or_lease_unavailable", 1),
            ("artifact", "artifact_unreadable_or_invalid", 1),
            ("pre_admission", "admission_invalid", 1),
            ("preflight", "fixed_preflight_blocked", 1),
        }
        outcome = (
            disposition.phase,
            disposition.reason_code,
            disposition.exit_code,
        )
        if type(disposition) is not expected_disposition or outcome not in allowed:
            raise RuntimeError("invalid fixed-owner expected disposition") from exc
        observe(*outcome, "reachable")
        return disposition.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
