from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


class CsvDecisionLogger:
    """Appends binary decisions to a CSV file immediately after each click."""

    fieldnames = ["timestamp_utc", "sample_id", "file_name", "source_path", "response", "response_bool"]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def append(
        self,
        sample_id: str,
        source_path: Path,
        response_text: str,
        response_bool: bool,
        update_if_exists: bool = False,
    ) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sample_id": sample_id,
            "file_name": source_path.name,
            "source_path": str(source_path),
            "response": response_text,
            "response_bool": str(response_bool),
        }

        if update_if_exists and self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            found_match = False
            for existing in rows:
                if existing.get("sample_id", "").strip() == sample_id:
                    existing.update(row)
                    found_match = True

            if found_match:
                with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                return

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
            f.flush()

    def labeled_ids(self) -> set[str]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return set()

        ids: set[str] = set()
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get("sample_id", "").strip()
                if sample_id:
                    ids.add(sample_id)
        return ids
