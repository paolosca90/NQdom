"""Shared utilities for depth-dom pipeline."""
import hashlib
import numpy as np
from pathlib import Path


def parse_ts_to_ms(ts_str: str) -> int:
    """Fast parse: '2026-01-08 09:30:00.123456 UTC' -> ms-from-midnight."""
    digits = "".join(c for c in ts_str if c.isdigit())
    return ((int(digits[8:10]) * 3600 +
             int(digits[10:12]) * 60 +
             int(digits[12:14])) * 1000 +
            int(digits[14:17]))


def memory_efficient_csv_read(path: Path, usecols: list[int], dtype=np.float64):
    """Memory-mapped CSV read for large files."""
    return np.loadtxt(path, usecols=usecols, dtype=dtype,
                      delimiter=",", skiprows=1)


def checksum_csv(path: Path) -> str:
    """MD5 checksum of a CSV file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
