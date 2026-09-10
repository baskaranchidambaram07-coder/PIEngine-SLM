"""Shared helpers for the large-file downloads both journeys perform.

The Studio caches models onto the server and the Runtime pulls models and LoRA
adapters onto the device. Both stream to a `.part` file and rename on success,
so both need the same cleanup on failure — kept here rather than triplicated.
"""
from __future__ import annotations

import shutil
from pathlib import Path

# Headroom left free after a download. The box also needs room for SQLite
# writes, logs and gradle scratch; filling the disk to the last byte corrupts
# those rather than merely failing the download.
SPACE_MARGIN_BYTES = 1 << 30   # 1 GB


def space_shortfall(dest_dir: Path, needed_bytes: int,
                    margin: int = SPACE_MARGIN_BYTES) -> int:
    """Bytes by which `dest_dir` is short of holding `needed_bytes` + margin.

    0 means there is room. Checking up front turns "ran the disk to zero, then
    failed at 64%" into an immediate, explainable refusal — and a full disk
    does not just lose the download, it can corrupt an in-flight SQLite write.

    An unknown size (0) or an unreadable filesystem returns 0: refusing a
    download because we could not measure it would be worse than attempting it,
    since the failure path now cleans up after itself.
    """
    if needed_bytes <= 0:
        return 0
    try:
        free = shutil.disk_usage(dest_dir).free
    except OSError:
        return 0
    return max(0, (needed_bytes + margin) - free)


def describe_shortfall(short: int, needed_bytes: int, dest_dir: Path) -> str:
    try:
        free = shutil.disk_usage(dest_dir).free
    except OSError:
        free = 0
    return (f"not enough disk space: needs {needed_bytes / 1e9:.2f} GB plus "
            f"{SPACE_MARGIN_BYTES / 1e9:.0f} GB headroom, but only "
            f"{free / 1e9:.2f} GB is free — {short / 1e9:.2f} GB short")


def discard_partial(tmp: Path) -> int:
    """Delete a failed download's .part file, returning the bytes reclaimed.

    Nothing resumes a .part, so leaving one behind is pure dead weight. The
    commonest failure for a multi-GB model is running out of disk — exactly
    when leaking gigabytes hurts most, and exactly when the next attempt is
    most likely to fail for want of the space the corpse is holding.
    """
    try:
        size = tmp.stat().st_size if tmp.exists() else 0
        tmp.unlink(missing_ok=True)
        return size
    except OSError:
        return 0
