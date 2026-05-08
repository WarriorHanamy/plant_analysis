from __future__ import annotations

from pathlib import Path

from plant_analysis.core.schemas import FilterChainReport
from plant_analysis.utils.ulog_utils import read_ulog_filter_chain


def run(ulog_path: str | Path) -> FilterChainReport:
    return read_ulog_filter_chain(ulog_path)
