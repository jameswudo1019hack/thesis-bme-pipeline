"""SHHS dataset paths, subject discovery, and local/placeholder detection."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

SHHS_ROOT = Path(
    "/Users/jameswu/Library/CloudStorage/"
    "OneDrive-SharedLibraries-TheUniversityofSydney(Staff)/"
    "Philip de Chazal - SHHS"
)

PILOT_JSON = Path(__file__).resolve().parents[1] / "pilot_subjects.json"

# BSD stat flag for FileProvider placeholders on macOS. A file with this flag
# has metadata but no local bytes — opening it triggers an on-demand download.
UF_DATALESS = 0x40000000

_SHHS1_EDF_RX = re.compile(r"^shhs1-(\d{6})\.edf$")


@dataclass(frozen=True)
class SubjectPaths:
    """Canonical file paths for a single SHHS subject."""

    subject_id: int
    cohort: str  # "shhs1" or "shhs2"
    edf: Path
    nsrr_xml: Path
    profusion_xml: Path

    @property
    def display_id(self) -> str:
        return f"{self.cohort}-{self.subject_id}"


def paths_for(subject_id: int, cohort: str = "shhs1") -> SubjectPaths:
    """Construct canonical paths for a SHHS subject (does not check existence)."""
    stem = f"{cohort}-{subject_id}"
    return SubjectPaths(
        subject_id=subject_id,
        cohort=cohort,
        edf=SHHS_ROOT / f"{stem}.edf",
        nsrr_xml=SHHS_ROOT / f"{stem}-nsrr.xml",
        profusion_xml=SHHS_ROOT / f"{stem}-profusion.xml",
    )


def is_local(path: Path) -> bool:
    """True if the file is materialised locally (not a cloud placeholder).

    Placeholder files on OneDrive/macOS FileProvider have the UF_DATALESS flag
    set on their stat struct. Reading a placeholder triggers a download — so
    we skip them when iterating in placeholder-aware mode.
    """
    try:
        return not bool(os.stat(path).st_flags & UF_DATALESS)
    except FileNotFoundError:
        return False


def is_subject_local(sp: SubjectPaths, require_xml: bool = True) -> bool:
    """True if the subject's EDF (and optionally XML) is locally materialised."""
    if not is_local(sp.edf):
        return False
    if require_xml and not is_local(sp.nsrr_xml):
        return False
    return True


def discover_shhs1_subject_ids() -> list[int]:
    """Return every SHHS1 subject ID visible in the shared folder (local or placeholder)."""
    ids: set[int] = set()
    for name in os.listdir(SHHS_ROOT):
        m = _SHHS1_EDF_RX.match(name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def load_pilot_subject_ids() -> list[int]:
    """Load the fixed-seed 100-subject pilot list from pilot_subjects.json."""
    data = json.loads(PILOT_JSON.read_text())
    return list(data["subject_ids"])


def local_subjects(
    cohort: str = "shhs1",
    restrict_to: list[int] | None = None,
    require_xml: bool = True,
) -> list[SubjectPaths]:
    """Yield all subjects whose files are currently materialised locally.

    ``restrict_to`` optionally filters to a specific list of subject IDs
    (e.g. the pilot set). ``require_xml`` requires the NSRR annotation XML
    to also be local.
    """
    ids = restrict_to if restrict_to is not None else discover_shhs1_subject_ids()
    out: list[SubjectPaths] = []
    for sid in ids:
        sp = paths_for(sid, cohort=cohort)
        if is_subject_local(sp, require_xml=require_xml):
            out.append(sp)
    return out
