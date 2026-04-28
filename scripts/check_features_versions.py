"""Quick diagnostic: count how many parquets are at each FEATURES_VERSION.

Useful before/after a regen to verify the cohort is on the expected schema.

Usage:
    cd Code && python scripts/check_features_versions.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = CODE_ROOT / "features"


def main() -> None:
    v: Counter[str] = Counter()
    for f in sorted(FEATURES_DIR.glob("shhs1-*.parquet")):
        try:
            ver = pd.read_parquet(f, columns=["features_version"])["features_version"].iloc[0]
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        v[ver] += 1

    print(f"\n=== FEATURES_VERSION distribution across {sum(v.values()):,} parquets ===\n")
    for ver, n in sorted(v.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>5,}  {ver}")
    print()


if __name__ == "__main__":
    main()
