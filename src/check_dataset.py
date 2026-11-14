"""Check that the dataset file is present and is exactly the file this project was built on.

Research prototype, not a diagnostic tool.

The dataset is not stored in this repository. Download `diagnosed_cbc_data_v4.csv` from the Kaggle
dataset "Anemia Types Classification" (E. Aboelnaga, Apache 2.0 licence):
    https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification
and place it at data/diagnosed_cbc_data_v4.csv.

Run:  python src/check_dataset.py      (exit code 0 = file present and hash matches)
"""
from __future__ import annotations

import sys

from data import RAW, RAW_SHA256, sha256_of

KAGGLE_URL = "https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification"
EXPECTED_BYTES = 125590


def main() -> int:
    if not RAW.exists():
        print(f"missing: {RAW.relative_to(RAW.parents[1])}")
        print(f"download diagnosed_cbc_data_v4.csv from {KAGGLE_URL} and place it in data/")
        return 1
    size, digest = RAW.stat().st_size, sha256_of(RAW)
    print(f"file: {RAW.name}, bytes: {size} (expected {EXPECTED_BYTES})")
    print(f"sha256: {digest}")
    print(f"expected: {RAW_SHA256}")
    if digest != RAW_SHA256:
        print("MISMATCH: this is not the file the project was built on. Results will not reproduce.")
        return 1
    print("OK: dataset matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
