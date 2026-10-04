"""Pre-registration freeze: hash the body of a vault note and check it against the frozen value.

The body of a pre-registration note is every byte before the first line that is exactly
``## Addenda``. Dated addenda are appended below that line after the freeze, so they
change the whole-file sha256 but never the body sha256.

The frozen value lives in the Code repository (``prereg/<name>_freeze.json``) so it is
part of every tagged commit that trains or predicts. ``check_prereg`` refuses a note whose
body hash differs from it. Scripts log both the body hash and the whole-file hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ADDENDA_HEADING = b"## Addenda"
AIM2_DL_FREEZE = CODE_ROOT / "prereg" / "aim2_dl_olsen_v1_freeze.json"


class PreregMismatch(RuntimeError):
    """The note's body differs from the frozen body, or the freeze record is missing."""


def note_body(path: str | Path) -> bytes:
    """Bytes before the first line that is exactly ``## Addenda`` (trailing whitespace ignored)."""
    data = Path(path).read_bytes()
    pos = 0
    for line in data.splitlines(keepends=True):
        if line.rstrip(b"\r\n").rstrip() == ADDENDA_HEADING:
            return data[:pos]
        pos += len(line)
    raise PreregMismatch(f"{path}: no '## Addenda' line; the body of a pre-registration is undefined without it")


def body_sha256(path: str | Path) -> str:
    return hashlib.sha256(note_body(path)).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(freeze_file: str | Path | None) -> Path:
    """``None`` means the module-level AIM2_DL_FREEZE, read at call time (tests patch it)."""
    return Path(freeze_file) if freeze_file is not None else Path(AIM2_DL_FREEZE)


def load_freeze(freeze_file: str | Path | None = None) -> dict:
    """The committed freeze record: {note, body_sha256, body_bytes, vault_commit, frozen_utc}."""
    p = _resolve(freeze_file)
    if not p.exists():
        raise PreregMismatch(f"no freeze record at {p}: the pre-registration has not been frozen")
    rec = json.loads(p.read_text())
    sha = rec.get("body_sha256")
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
        raise PreregMismatch(f"{p}: body_sha256 missing or not a sha256 hex digest")
    return rec


def freeze_summary(freeze_file: str | Path | None = None) -> dict:
    """What training records about the freeze (no note needed): body hash, vault commit, record hash."""
    freeze_file = _resolve(freeze_file)
    rec = load_freeze(freeze_file)
    return {
        "body_sha256": rec["body_sha256"],
        "vault_commit": rec.get("vault_commit"),
        "freeze_file": str(Path(freeze_file).resolve()),
        "freeze_file_sha256": file_sha256(freeze_file),
    }


def check_prereg(note: str | Path, freeze_file: str | Path | None = None) -> dict:
    """Refuse unless sha256(body of ``note``) equals the frozen body hash.

    Returns the provenance to log: note path, whole-file sha256, body sha256 and the
    freeze record's own sha256.
    """
    freeze_file = _resolve(freeze_file)
    rec = load_freeze(freeze_file)
    got = body_sha256(note)
    if got != rec["body_sha256"]:
        raise PreregMismatch(
            f"{note}: body sha256 {got[:12]} != frozen {rec['body_sha256'][:12]} ({freeze_file}); "
            "the body of a frozen pre-registration may not change (record decisions as dated addenda)")
    return {
        "prereg": str(Path(note).resolve()),
        "prereg_sha256": file_sha256(note),
        "prereg_body_sha256": got,
        "prereg_freeze_file": str(Path(freeze_file).resolve()),
        "prereg_freeze_sha256": file_sha256(freeze_file),
        "prereg_vault_commit": rec.get("vault_commit"),
    }
