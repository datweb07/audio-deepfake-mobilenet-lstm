"""Pinned, bounded Hugging Face acquisition with explicit legal-use and hash gates."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import shutil
from typing import Callable
from urllib.parse import quote
from urllib.request import urlopen

from .evidence import file_hash, object_hash, write_json

REPOSITORY = "mueller91/MLAAD-tiny"
LICENSE_FILES = ("LICENSE", "original/LICENSE", "NOTICE_NO_COMMERCIAL_USE.txt", "README.md")


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("Unsafe remote relative path")
    return path.as_posix()


def pinned_revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Immutable 40-character repository revision required")
    return value


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str = "UNKNOWN"
    git_blob: str = "UNKNOWN"

    def __post_init__(self):
        safe_relative(self.path)
        if self.size <= 0:
            raise ValueError("Positive expected file size required")
        if self.sha256 != "UNKNOWN" and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Invalid remote SHA-256")


def bounded_get(url: str, maximum: int = 15_000_000) -> bytes:
    with urlopen(url, timeout=60) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Response exceeds byte budget")
    return data


def resolve_url(repository: str, revision: str, path: str) -> str:
    if repository != REPOSITORY:
        raise ValueError("Only the reviewed MLAAD-tiny official repository is authorized")
    return f"https://huggingface.co/datasets/{repository}/resolve/{pinned_revision(revision)}/{quote(safe_relative(path))}"


def acquire_file(item: RemoteFile, *, repository: str, revision: str, destination: Path,
                 fetch: Callable[[str, int], bytes] = bounded_get) -> dict:
    """Never resume partial bytes, silently overwrite, or accept an unverified LFS payload."""
    destination = destination.resolve()
    target = destination / safe_relative(item.path)
    if not target.resolve().is_relative_to(destination):
        raise ValueError("Destination path escaped dataset root")
    if target.exists():
        raise FileExistsError(f"Immutable acquisition target exists: {item.path}")
    if item.path.lower().endswith((".wav", ".flac")) and item.sha256 == "UNKNOWN":
        raise ValueError("Audio requires upstream content SHA-256")
    url = resolve_url(repository, revision, item.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".part")
    if pending.exists():
        raise FileExistsError("Unresolved partial download; inspect before retrying")
    try:
        payload = fetch(url, item.size)
        pending.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != item.size or (item.sha256 != "UNKNOWN" and digest != item.sha256):
            raise ValueError("Downloaded size or checksum mismatch")
        if item.git_blob != "UNKNOWN":
            blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if blob != item.git_blob:
                raise ValueError("Git blob identity mismatch")
        pending.replace(target)
        return {**asdict(item), "repository": repository, "revision": revision, "url": url,
                "status": "VERIFIED", "checksum": digest, "downloaded_bytes": len(payload)}
    except Exception:
        # Preserve .part for diagnosis, but never expose it as a successful source file.
        raise


def license_gate(directory: Path, *, purpose: str) -> dict:
    if purpose != "noncommercial_research":
        raise ValueError("MLAAD-tiny use is restricted to reviewed noncommercial research")
    for name in LICENSE_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"Missing authoritative license/attribution file: {name}")
    fake = (directory / "LICENSE").read_text(encoding="utf8")
    real = (directory / "original/LICENSE").read_text(encoding="utf8")
    notice = (directory / "NOTICE_NO_COMMERCIAL_USE.txt").read_text(encoding="utf8")
    if "CC BY-NC 4.0" not in fake or "M-AILABS" not in real or "commercial" not in notice.lower():
        raise ValueError("License text differs from reviewed dataset terms")
    return {"purpose": purpose, "fake": "CC-BY-NC-4.0", "real": "M-AILABS original notice",
            "attribution": "Nicolas Mueller / Fraunhofer AISEC, MLAAD authors; M-AILABS original creators",
            "license_hashes": {name: file_hash(directory / name) for name in LICENSE_FILES},
            "commercial_permission": "NOT_GRANTED", "changes": "Selected subset; downstream resampling/cropping/channel transforms recorded separately"}


def discover(output: Path, *, revision: str | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    suffix = f"/revision/{pinned_revision(revision)}" if revision else ""
    info = json.loads(bounded_get(f"https://huggingface.co/api/datasets/{REPOSITORY}{suffix}?blobs=true"))
    if revision is not None and info['sha'] != revision:
        raise ValueError('Repository returned a different requested revision')
    revision = pinned_revision(info["sha"])
    if info.get("private") or info.get("gated") or info.get("disabled"):
        raise ValueError("Public ungated active source required")
    write_json(output / "repository.json", info)
    inventory = []
    for row in info["siblings"]:
        lfs = row.get("lfs") or {}
        inventory.append(asdict(RemoteFile(row["rfilename"], row["size"], lfs.get("sha256", "UNKNOWN"),
                                          row.get("blobId", "UNKNOWN") if not lfs else "UNKNOWN")))
    write_json(output / "inventory.json", inventory)
    selected = [row for row in inventory if row["path"] in LICENSE_FILES or row["path"].endswith("/meta.csv")]
    reports = []
    for row in selected:
        reports.append(acquire_file(RemoteFile(**row), repository=REPOSITORY, revision=revision, destination=output))
    terms = license_gate(output, purpose="noncommercial_research")
    report = {"status": "METADATA_ONLY", "repository": REPOSITORY, "revision": revision,
              "inventory_sha256": file_hash(output / "inventory.json"), "files": reports, "license": terms,
              "audio_bytes": 0, "repository_audio_files": sum(row["path"].endswith(".wav") for row in inventory)}
    write_json(output / "discovery.json", report)
    return report


def acquire_selection(discovery: Path, selection: dict, destination: Path, *, maximum_bytes: int,
                      purpose: str, fetch: Callable[[str, int], bytes] = bounded_get) -> dict:
    report = json.loads((discovery / "discovery.json").read_text())
    terms = license_gate(discovery, purpose=purpose)
    revision = pinned_revision(report["revision"])
    if selection["revision"] != revision or selection["scope"] != "PUBLIC_DEVELOPMENT":
        raise ValueError("Selection revision/scope mismatch; confirmation acquisition is not exposed")
    if file_hash(discovery / "inventory.json") != report["inventory_sha256"]:
        raise ValueError("Discovery inventory changed")
    inventory = {row["path"]: RemoteFile(**row) for row in json.loads((discovery / "inventory.json").read_text())}
    paths = [row["path"] for row in selection["records"]]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Empty or duplicate selection")
    items = [inventory[path] for path in paths]
    total = sum(item.size for item in items)
    free = shutil.disk_usage(destination.parent if destination.parent.exists() else discovery).free
    if maximum_bytes <= 0 or total > maximum_bytes or free < total * 2 + 1024**3:
        raise ValueError("Acquisition exceeds download budget or disk reserve")
    destination.mkdir(parents=True, exist_ok=False)
    state = {"status": "ACQUIRING", "repository": REPOSITORY, "revision": revision,
             "selection_hash": object_hash(selection), "estimated_bytes": total, "free_disk_before": free,
             "license": terms, "files": []}
    write_json(destination / "acquisition.json", state)
    try:
        for item in items:
            state["files"].append(acquire_file(item, repository=REPOSITORY, revision=revision, destination=destination, fetch=fetch))
            write_json(destination / "acquisition.json", state)
            print(f"Acquired {len(state['files'])}/{len(items)}", flush=True)
        state["status"] = "ACQUIRED_VERIFIED"
    except Exception as error:
        state.update(status="FAILED_PARTIAL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["downloaded_bytes"] = sum(row["downloaded_bytes"] for row in state["files"])
        write_json(destination / "acquisition.json", state)
    return state
