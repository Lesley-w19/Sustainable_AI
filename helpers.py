# helpers.py
import os
import csv
import json
import datetime as dt


def _pct_change(before: float, after: float) -> float:
    """Return percentage change from before → after (negative = reduction)."""
    if before == 0:
        return 0.0
    return (after - before) / before * 100.0
