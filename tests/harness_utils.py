from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PEER_REVIEW = ROOT / "peer_review"
TABLES_DIR = PEER_REVIEW / "tables"
FIGURES_DIR = PEER_REVIEW / "figures"
CODE_DIR = PEER_REVIEW / "code"


def read_text(relative_path: str) -> str:
    """Read a UTF-8 text file relative to the project root."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def read_csv_rows(relative_path: str) -> list[dict[str, str]]:
    """Load a CSV into a list of dict rows."""
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_columns(relative_path: str) -> list[str]:
    """Return CSV column names without reading the whole file elsewhere."""
    rows = read_csv_rows(relative_path)
    if not rows:
        return []
    return list(rows[0].keys())


def find_row(rows: list[dict[str, str]], key: str, expected_value: str) -> dict[str, str]:
    """Return the first row whose key matches the expected value."""
    for row in rows:
        if row.get(key) == expected_value:
            return row
    raise AssertionError(f"Could not find row where {key} == {expected_value!r}")


def parse_number(value: str | None) -> float:
    """Convert a display number from tables/manuscript snippets to float."""
    if value is None:
        raise AssertionError("Expected a numeric value, found None")

    cleaned = (
        value.strip()
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
    )

    if cleaned == "" or cleaned.lower() == "nan":
        raise AssertionError(f"Expected a numeric value, found {value!r}")

    return float(cleaned)
