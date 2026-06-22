"""
Logging utilities for the Multi-Task Federated Learning experiment.

Provides:
  - Structured logging to console + rotating log files
  - CSVMetricsLogger for per-round / per-epoch metric tracking
  - Auto-creates log directory from config
"""
import csv
import logging
import os
import sys
from datetime import datetime


def setup_logger(name, log_dir, level=logging.INFO):
    """Create a logger that writes to both console and a timestamped log file.

    Args:
        name: logger name (e.g. "federated", "centralized")
        log_dir: directory for log files
        level: logging level

    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}.log"), mode="w")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class CSVMetricsLogger:
    """Append metrics row-by-row to a CSV file for easy post-hoc analysis.

    Usage:
        ml = CSVMetricsLogger("logs/fed_metrics.csv",
                              ["round", "fc_test_mse", "an_test_mse"])
        ml.log({"round": 1, "fc_test_mse": 0.25, "an_test_mse": 0.18})
    """

    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row_dict):
        self._writer.writerow(row_dict)
        self._file.flush()

    def close(self):
        self._file.close()
