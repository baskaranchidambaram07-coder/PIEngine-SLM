"""Shared helpers for the large-file downloads both journeys perform.

The Studio caches models onto the server and the Runtime pulls models and LoRA
adapters onto the device. Both stream to a `.part` file and rename on success,
so both need the same cleanup on failure — kept here rather than triplicated.
"""
from __future__ import annotations

from pathlib import Path


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
