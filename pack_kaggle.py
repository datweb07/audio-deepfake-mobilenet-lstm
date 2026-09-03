"""Build a portable, production-only Kaggle archive for LAVA training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "lava_kaggle_input_clean.zip"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
SPLIT_MANIFEST = MANIFEST_DIR / "split_manifest.csv"
MANIFEST_METADATA = MANIFEST_DIR / "manifest_metadata.json"

ROOT_FILES = (
    ".gitignore",
    "app.py",
    "config.py",
    "evaluate.py",
    "pack_kaggle.py",
    "predict.py",
    "README.md",
    "requirements.txt",
    "requirements-torch.txt",
    "train.py",
)
PRODUCTION_DIRECTORIES = ("benchmark", "configs", "scripts", "src", "tests")
IGNORED_DIRECTORY_NAMES = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".vscode", ".idea",
}
VALID_SPLITS = {"train", "validation", "test"}
VALID_LABELS = {0, 1}
SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _stable_manifest_hash(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in sorted(rows, key=lambda value: value["sample_id"]):
        lines.append(
            "|".join(
                row[key]
                for key in (
                    "sample_id", "path", "label", "sha256", "duplicate_group_id", "split"
                )
            )
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _canonical_audio_files() -> tuple[list[Path], dict[str, object]]:
    if not SPLIT_MANIFEST.is_file() or not MANIFEST_METADATA.is_file():
        raise FileNotFoundError(
            "Canonical manifests are missing. Run: python -m src.lava.data.manifest check"
        )
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with MANIFEST_METADATA.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not rows:
        raise RuntimeError("split_manifest.csv contains no training samples")
    if any(row.get("integrity_status") != "INCLUDED" for row in rows):
        raise ValueError("Excluded/conflicting sample found in split_manifest.csv")
    if any(row.get("included", "").lower() != "true" for row in rows):
        raise ValueError("A non-included sample found in split_manifest.csv")
    if {row.get("split") for row in rows} - VALID_SPLITS:
        raise ValueError("Invalid split name in split_manifest.csv")
    if {int(row["label"]) for row in rows} != VALID_LABELS:
        raise ValueError("Manifest must preserve REAL=0 and FAKE=1")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample_id in split_manifest.csv")

    actual_manifest_hash = _stable_manifest_hash(rows)
    if actual_manifest_hash != metadata.get("manifest_hash"):
        raise ValueError(
            f"Manifest hash mismatch: metadata={metadata.get('manifest_hash')} "
            f"actual={actual_manifest_hash}"
        )
    if len(rows) != int(metadata.get("included_samples", -1)):
        raise ValueError("Manifest included-sample count does not match metadata")
    expected_splits = {key: int(value) for key, value in metadata.get("split_counts", {}).items()}
    actual_splits = dict(Counter(row["split"] for row in rows))
    if actual_splits != expected_splits:
        raise ValueError(f"Split-count mismatch: expected={expected_splits}, actual={actual_splits}")

    files: list[Path] = []
    seen_paths: set[str] = set()
    data_root = (PROJECT_ROOT / "data").resolve()
    for row in rows:
        relative = Path(row["path"])
        candidate = (PROJECT_ROOT / relative).resolve()
        try:
            candidate.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes production data root: {relative}") from exc
        if candidate.suffix.lower() not in SUPPORTED_AUDIO:
            raise ValueError(f"Unsupported manifest audio extension: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Manifest audio file is missing: {candidate}")
        archive_name = candidate.relative_to(PROJECT_ROOT).as_posix()
        if archive_name in seen_paths:
            raise ValueError(f"Duplicate audio path in manifest: {archive_name}")
        seen_paths.add(archive_name)
        files.append(candidate)
    return sorted(files), metadata


def _production_files() -> list[Path]:
    files: list[Path] = []
    for name in ROOT_FILES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Required production file is missing: {path}")
        files.append(path)
    for directory_name in PRODUCTION_DIRECTORIES:
        directory = PROJECT_ROOT / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Required production directory is missing: {directory}")
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(directory).parts
            if any(part in IGNORED_DIRECTORY_NAMES for part in relative_parts):
                continue
            if path.suffix.lower() in {".pyc", ".pyo", ".zip", ".keras", ".h5", ".pt", ".pth"}:
                continue
            files.append(path)
    files.extend(path for path in MANIFEST_DIR.rglob("*") if path.is_file())
    return sorted(set(files))


def _verify_audio_hashes(audio_files: list[Path]) -> None:
    expected: dict[str, str] = {}
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected[(PROJECT_ROOT / row["path"]).resolve().as_posix()] = row["sha256"]
    for index, path in enumerate(audio_files, start=1):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected[path.resolve().as_posix()]:
            raise ValueError(f"Audio checksum differs from canonical manifest: {path}")
        if index % 1000 == 0:
            print(f"[VERIFY] Audio hashes: {index:,}/{len(audio_files):,}")


def pack_for_kaggle(
    output: Path = DEFAULT_OUTPUT,
    *,
    dry_run: bool = False,
    verify_audio_hashes: bool = False,
) -> Path:
    output = output.resolve()
    audio_files, metadata = _canonical_audio_files()
    production_files = _production_files()
    all_files = production_files + audio_files
    archive_names = [path.relative_to(PROJECT_ROOT).as_posix() for path in all_files]
    if len(archive_names) != len(set(archive_names)):
        raise ValueError("Packaging plan contains duplicate archive paths")

    total_bytes = sum(path.stat().st_size for path in all_files)
    print(f"[PLAN] Production/manifests: {len(production_files):,} files")
    print(f"[PLAN] Canonical audio: {len(audio_files):,} files")
    print(f"[PLAN] Split counts: {metadata['split_counts']}")
    print(f"[PLAN] Manifest hash: {metadata['manifest_hash']}")
    print(f"[PLAN] Uncompressed input: {total_bytes / 1024**3:.2f} GiB")
    print("[PLAN] Reference repositories, models, outputs and excluded dataset samples: OMITTED")
    if dry_run:
        print("[DRY RUN] Validation passed; no ZIP was written.")
        return output

    if verify_audio_hashes:
        _verify_audio_hashes(audio_files)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    
    try:
        # Tách riêng khối với zipfile để đảm bảo giải phóng file lock hoàn toàn
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=6, allowZip64=True,
        ) as archive:
            for index, path in enumerate(all_files, start=1):
                archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
                if index % 1000 == 0 or index == len(all_files):
                    print(f"[PACK] {index:,}/{len(all_files):,} files")
        
        # Đổi tên file ngoài khối 'with'
        os.replace(temporary, output)
    except BaseException:
        # Thêm try-except cho unlink phòng trường hợp file vẫn đang bị lock
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass
        raise

    size_gib = output.stat().st_size / 1024**3
    print(f"[SUCCESS] Created: {output} ({size_gib:.2f} GiB)")
    print("Upload this exact file as a Kaggle Dataset, then copy/extract it to /kaggle/working.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Validate and show the plan only")
    parser.add_argument(
        "--verify-audio-hashes", action="store_true",
        help="Re-hash every audio file before packing (slow but strongest integrity check)",
    )
    arguments = parser.parse_args()
    pack_for_kaggle(
        arguments.output,
        dry_run=arguments.dry_run,
        verify_audio_hashes=arguments.verify_audio_hashes,
    )


if __name__ == "__main__":
    main()
