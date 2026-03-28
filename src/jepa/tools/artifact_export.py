"""Helpers for packaging notebook artifacts from remote runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import shutil
import zipfile


@dataclass(slots=True)
class ArtifactBundleResult:
    """Summary of a packaged artifact archive."""

    archive_path: Path
    file_count: int
    total_bytes: int
    included_files: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "archive_path": str(self.archive_path),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "included_files": list(self.included_files),
        }


def create_artifact_bundle(
    artifact_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    bundle_stem: str = "jepa_artifacts",
    base_dir: str | Path | None = None,
) -> ArtifactBundleResult:
    """Bundle selected files and directories into a single zip archive."""

    resolved_paths = [Path(path).resolve() for path in artifact_paths]
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = output_root / f"{bundle_stem}_{timestamp}.zip"
    archive_path = archive_path.resolve()

    if base_dir is not None:
        base_path = Path(base_dir).resolve()
    else:
        base_path = _infer_common_base(resolved_paths)

    included_files: List[str] = []
    total_bytes = 0

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for artifact_path in resolved_paths:
            if not artifact_path.exists():
                continue
            if artifact_path.is_dir():
                for file_path in sorted(path for path in artifact_path.rglob("*") if path.is_file()):
                    archive_name = _relative_archive_name(file_path, base_path)
                    bundle.write(file_path, archive_name)
                    included_files.append(archive_name)
                    total_bytes += int(file_path.stat().st_size)
            elif artifact_path.is_file():
                archive_name = _relative_archive_name(artifact_path, base_path)
                bundle.write(artifact_path, archive_name)
                included_files.append(archive_name)
                total_bytes += int(artifact_path.stat().st_size)

    return ArtifactBundleResult(
        archive_path=archive_path,
        file_count=len(included_files),
        total_bytes=total_bytes,
        included_files=included_files,
    )


def copy_artifact_bundle(archive_path: str | Path, destination_dir: str | Path) -> Path:
    """Copy an existing artifact bundle to another directory, such as Google Drive."""

    source = Path(archive_path).resolve()
    destination_root = Path(destination_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_path = destination_root / source.name
    shutil.copy2(source, destination_path)
    return destination_path


def _infer_common_base(paths: Iterable[Path]) -> Path:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return Path.cwd().resolve()

    anchor = existing[0] if existing[0].is_dir() else existing[0].parent
    anchor_parts = anchor.parts
    for path in existing[1:]:
        candidate = path if path.is_dir() else path.parent
        max_index = min(len(anchor_parts), len(candidate.parts))
        shared_parts: List[str] = []
        for index in range(max_index):
            if anchor_parts[index] != candidate.parts[index]:
                break
            shared_parts.append(anchor_parts[index])
        if shared_parts:
            anchor = Path(*shared_parts)
            anchor_parts = anchor.parts
        else:
            return Path(candidate.anchor or anchor.anchor or "/").resolve()
    return anchor.resolve()


def _relative_archive_name(path: Path, base_dir: Path) -> str:
    try:
        relative = path.resolve().relative_to(base_dir.resolve())
        return relative.as_posix()
    except Exception:
        return path.name
