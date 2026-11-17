"""
ML Dataset Quick Auditor
------------------------
A small, dependency-free utility for checking CSV datasets before model training.

Usage:
    python ml_dataset_audit.py path/to/dataset.csv
    python ml_dataset_audit.py path/to/dataset.csv --target category
    python ml_dataset_audit.py path/to/dataset.csv --sample 10

The script reports:
- row and column counts
- missing and duplicate values
- target-label distribution (optional)
- a small data preview

This is intended as a practical preprocessing/checking step for ML coursework.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file using UTF-8 (with BOM support)."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("The CSV file is empty or has no header row.")

        rows = []
        for row in reader:
            # Ignore fully blank lines.
            if any(value not in (None, "") for value in row.values()):
                rows.append({key: (value or "").strip() for key, value in row.items()})
        return list(reader.fieldnames), rows


def count_missing(rows: Iterable[dict[str, str]], columns: list[str]) -> dict[str, int]:
    """Count empty cells in each column."""
    counts = Counter()
    for row in rows:
        for column in columns:
            if not row.get(column, "").strip():
                counts[column] += 1
    return {column: counts[column] for column in columns}


def count_duplicates(rows: list[dict[str, str]]) -> int:
    """Count duplicate rows."""
    seen = set()
    duplicates = 0
    for row in rows:
        key = tuple(sorted(row.items()))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def print_report(
    path: Path,
    columns: list[str],
    rows: list[dict[str, str]],
    target: str | None,
    sample_size: int,
) -> None:
    print("\n=== ML DATASET AUDIT ===")
    print(f"File: {path}")
    print(f"Rows: {len(rows):,}")
    print(f"Columns: {len(columns)}")
    print("\nColumns:")
    for column in columns:
        print(f"  - {column}")

    print("\nMissing values:")
    missing = count_missing(rows, columns)
    for column, count in missing.items():
        percentage = (count / len(rows) * 100) if rows else 0
        print(f"  {column}: {count:,} ({percentage:.2f}%)")

    print(f"\nDuplicate rows: {count_duplicates(rows):,}")

    if target:
        if target not in columns:
            print(f"\nWarning: target column '{target}' was not found.")
        else:
            labels = Counter(row.get(target, "") or "<EMPTY>" for row in rows)
            print(f"\nLabel distribution for '{target}':")
            for label, count in labels.most_common():
                percentage = (count / len(rows) * 100) if rows else 0
                print(f"  {label}: {count:,} ({percentage:.2f}%)")

    print(f"\nPreview (first {sample_size} rows):")
    for index, row in enumerate(rows[:sample_size], start=1):
        print(f"\n[{index}]")
        for column in columns:
            value = row.get(column, "")
            if len(value) > 120:
                value = value[:117] + "..."
            print(f"  {column}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a CSV dataset for ML preprocessing.")
    parser.add_argument("csv_file", type=Path, help="Path to the CSV dataset")
    parser.add_argument("--target", help="Optional target/label column name")
    parser.add_argument("--sample", type=int, default=5, help="Number of preview rows (default: 5)")
    args = parser.parse_args()

    if args.sample < 0:
        parser.error("--sample must be zero or greater")
    if not args.csv_file.is_file():
        parser.error(f"File not found: {args.csv_file}")

    try:
        columns, rows = read_csv(args.csv_file)
        print_report(args.csv_file, columns, rows, args.target, args.sample)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
