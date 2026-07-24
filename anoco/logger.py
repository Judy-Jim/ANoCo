"""Production inspection logger: rolling CSV + statistics + drift detection.

Usage:
    logger = InspectionLogger(log_dir="./logs")
    logger.log(result)               # record one inspection
    stats = logger.get_stats(last_n=100)  # recent statistics
    if logger.check_drift(window=500):   # distribution drift alert
        ...
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class InspectResult:
    """Result of a single image inspection."""
    score: float
    decision: str              # OK / NG / ERROR
    anomaly_map: object = None # np.ndarray (H, W) float32, or None
    latency_ms: float = 0.0
    timestamp: str = ""
    filename: str = ""
    error_msg: str = ""        # only when decision == "ERROR"

    def to_csv_row(self) -> list:
        return [
            self.timestamp,
            self.filename,
            f"{self.score:.6f}",
            self.decision,
            f"{self.latency_ms:.1f}",
            self.error_msg,
        ]


class InspectionLogger:
    """Thread-safe rolling CSV logger with statistics and drift detection.

    CSV columns: timestamp, filename, score, decision, latency_ms, error_msg
    """

    def __init__(self, log_dir: str, max_rows: int = 100_000):
        """
        Args:
            log_dir: directory for CSV log files.
            max_rows: max rows per CSV file before rotating to a new file.
        """
        self.log_dir = log_dir
        self.max_rows = max_rows
        os.makedirs(log_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._row_count = 0
        self._file_index = 0
        self._fh = None
        self._writer = None
        self._scores = deque(maxlen=10000)  # ring buffer for stats
        self._latencies = deque(maxlen=10000)
        self._decisions = deque(maxlen=10000)

        self._open_new_file()

    def _open_new_file(self):
        if self._fh is not None:
            self._fh.close()
        fname = f"inspection_{self._file_index:04d}.csv"
        path = os.path.join(self.log_dir, fname)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(
            ["timestamp", "filename", "score", "decision", "latency_ms", "error_msg"]
        )
        self._row_count = 0

    def log(self, result: InspectResult):
        """Record one inspection result."""
        if not result.timestamp:
            result.timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._writer.writerow(result.to_csv_row())
            self._fh.flush()
            self._row_count += 1
            self._scores.append(result.score)
            self._latencies.append(result.latency_ms)
            self._decisions.append(result.decision)

            if self._row_count >= self.max_rows:
                self._file_index += 1
                self._open_new_file()

    def get_stats(self, last_n: int = 100) -> dict:
        """Return statistics for the last N inspections."""
        with self._lock:
            scores = list(self._scores)[-last_n:]
            latencies = list(self._latencies)[-last_n:]
            decisions = list(self._decisions)[-last_n:]

        if not scores:
            return {"count": 0}

        n_ok = sum(1 for d in decisions if d == "OK")
        n_ng = sum(1 for d in decisions if d == "NG")
        n_err = sum(1 for d in decisions if d == "ERROR")

        return {
            "count": len(scores),
            "mean_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "ng_rate": n_ng / len(scores) if scores else 0,
            "ok_rate": n_ok / len(scores) if scores else 0,
            "error_rate": n_err / len(scores) if scores else 0,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
        }

    def check_drift(self, window: int = 500) -> bool:
        """Detect score distribution drift.

        Compares the mean of the last `window` scores against the mean of the
        preceding `window` scores. Returns True if the shift exceeds 2 standard
        deviations of the preceding window (potential process change).

        Args:
            window: number of samples per comparison window.

        Returns:
            True if drift detected.
        """
        with self._lock:
            scores = list(self._scores)

        if len(scores) < 2 * window:
            return False

        recent = scores[-window:]
        baseline = scores[-2 * window:-window]

        mean_b = sum(baseline) / len(baseline)
        std_b = (sum((x - mean_b) ** 2 for x in baseline) / len(baseline)) ** 0.5
        mean_r = sum(recent) / len(recent)

        if std_b < 1e-8:
            return False

        return abs(mean_r - mean_b) > 2 * std_b

    def close(self):
        """Close the current log file."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
