"""thesis_pipeline/prereg.py: the body of a pre-registration note and its freeze record.

The body is every byte above the first '## Addenda' line, so dated addenda appended after the
freeze leave the body hash unchanged while any body edit changes it, including an edit of the
YAML frontmatter (``updated:`` / ``status:``), which is part of the body. A note without the
'## Addenda' line, a missing freeze record and a malformed one are refused.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_pipeline import prereg  # noqa: E402
from thesis_pipeline.prereg import PreregMismatch  # noqa: E402

BODY = "# Pre-registration (synthetic)\n\nClaim rule: A-E.\n\n"
NOTE = BODY + "## Addenda\n"


def _write(p: Path, text: str) -> Path:
    p.write_bytes(text.encode())
    return p


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A note plus a freeze record of its body; prereg.AIM2_DL_FREEZE points at the record."""
    note = _write(tmp_path / "note.md", NOTE)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"note": str(note), "body_sha256": hashlib.sha256(BODY.encode()).hexdigest(),
                                  "body_bytes": len(BODY.encode()), "vault_commit": "abc123",
                                  "frozen_utc": "2026-10-04T00:00:00Z"}))
    monkeypatch.setattr(prereg, "AIM2_DL_FREEZE", freeze)
    return note, freeze


def test_body_excludes_the_addenda(tmp_path):
    note = _write(tmp_path / "n.md", NOTE + "\n### 2026-10-05 aggregator lock\nmean\n")
    assert prereg.note_body(note) == BODY.encode()
    assert prereg.body_sha256(note) == hashlib.sha256(BODY.encode()).hexdigest()
    assert prereg.file_sha256(note) == hashlib.sha256(note.read_bytes()).hexdigest()


def test_only_an_exact_addenda_heading_ends_the_body(tmp_path):
    text = "# t\n### Addenda notes\n## Addenda and more\n## Addenda  \r\nafter\n"
    note = _write(tmp_path / "n.md", text)
    assert prereg.note_body(note) == b"# t\n### Addenda notes\n## Addenda and more\n"
    # the first '## Addenda' line wins; a later one is part of the addenda
    note2 = _write(tmp_path / "n2.md", "a\n## Addenda\nb\n## Addenda\nc\n")
    assert prereg.note_body(note2) == b"a\n"


def test_appending_an_addendum_keeps_the_body_hash(frozen):
    note, freeze = frozen
    first = prereg.check_prereg(note)
    note.write_bytes(note.read_bytes() + b"\n### 2026-10-06 recipe tier\nA (gate rule outcome)\n")
    second = prereg.check_prereg(note)
    assert first["prereg_body_sha256"] == second["prereg_body_sha256"]
    assert first["prereg_sha256"] != second["prereg_sha256"]  # the whole-file hash moves
    assert second["prereg_sha256"] == prereg.file_sha256(note)
    assert second["prereg_freeze_sha256"] == prereg.file_sha256(freeze)
    assert second["prereg_vault_commit"] == "abc123"
    assert second["prereg"] == str(note.resolve()) and second["prereg_freeze_file"] == str(freeze.resolve())


def test_editing_the_body_changes_its_hash_and_is_refused(frozen):
    note, _ = frozen
    before = prereg.body_sha256(note)
    note.write_bytes(NOTE.replace("A-E", "A-D").encode())
    assert prereg.body_sha256(note) != before
    with pytest.raises(PreregMismatch, match="may not change"):
        prereg.check_prereg(note)


def test_yaml_frontmatter_is_part_of_the_frozen_body(tmp_path, monkeypatch):
    """The body is every byte above '## Addenda', INCLUDING the YAML frontmatter: bumping the
    vault's ``updated:`` / ``status:`` fields after the freeze is a body edit and is refused, even
    when the edit only accompanies an appended addendum. (Pinned behaviour: the frontmatter of a
    frozen note must never be edited.)"""
    front = "---\ntags: [aim, apnoea]\nupdated: 2026-10-04\nstatus: frozen\n---\n"
    note = _write(tmp_path / "fm.md", front + NOTE)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"note": str(note), "body_sha256": prereg.body_sha256(note)}))
    monkeypatch.setattr(prereg, "AIM2_DL_FREEZE", freeze)
    assert prereg.note_body(note).startswith(front.encode())
    with open(note, "ab") as fh:  # the legitimate post-freeze edit: an appended addendum
        fh.write(b"\n### 2026-11-01 aggregator lock\nmean\n")
    prereg.check_prereg(note)
    note.write_bytes(note.read_bytes().replace(b"updated: 2026-10-04", b"updated: 2026-11-01"))
    with pytest.raises(PreregMismatch, match="may not change"):
        prereg.check_prereg(note)
    note.write_bytes(note.read_bytes().replace(b"updated: 2026-11-01", b"updated: 2026-10-04"))
    prereg.check_prereg(note)  # only an exact revert of the frontmatter recovers


def test_a_note_without_addenda_heading_is_refused(frozen, tmp_path):
    bare = _write(tmp_path / "bare.md", BODY)
    with pytest.raises(PreregMismatch, match="no '## Addenda' line"):
        prereg.body_sha256(bare)
    with pytest.raises(PreregMismatch, match="no '## Addenda' line"):
        prereg.check_prereg(bare)


def test_missing_or_malformed_freeze_record_is_refused(frozen, tmp_path, monkeypatch):
    note, _ = frozen
    monkeypatch.setattr(prereg, "AIM2_DL_FREEZE", tmp_path / "nope.json")
    with pytest.raises(PreregMismatch, match="has not been frozen"):
        prereg.check_prereg(note)
    with pytest.raises(PreregMismatch, match="has not been frozen"):
        prereg.freeze_summary()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"body_sha256": "XYZ"}))
    with pytest.raises(PreregMismatch, match="not a sha256"):
        prereg.check_prereg(note, freeze_file=bad)


def test_freeze_summary_reads_the_patched_record_at_call_time(frozen):
    _, freeze = frozen
    s = prereg.freeze_summary()
    assert s == {"body_sha256": hashlib.sha256(BODY.encode()).hexdigest(), "vault_commit": "abc123",
                 "freeze_file": str(freeze.resolve()), "freeze_file_sha256": prereg.file_sha256(freeze)}
