from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from plant_analysis.core.schemas import DiagnosisOutputPaths, FilterSpec
from plant_analysis.diagnostics.filter_analysis import run as run_filter_chain
from plant_analysis.diagnostics.filter_design import run as run_filter_design


def app(
    type: Annotated[Literal["filter-chain", "filter-design"], typer.Option("--type")],
    input: Annotated[Path | None, typer.Option("--input", exists=True, readable=True)] = None,
    fc: Annotated[float, typer.Option("--fc")] = 50.0,
    order: Annotated[int, typer.Option("--order")] = 4,
    fs: Annotated[float | None, typer.Option("--fs")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("_output"),
) -> None:
    if type == "filter-chain":
        if input is None:
            raise typer.BadParameter("--input is required for filter-chain diagnostics")
        paths = DiagnosisOutputPaths(output_dir=output_dir)
        report = run_filter_chain(input, paths)
        typer.echo(f"Found {len(report.signal_names)} filter-chain signals")
        return
    spec = FilterSpec(filter_type="butterworth", order=order, cutoff=fc, fs=fs)
    result = run_filter_design(spec, DiagnosisOutputPaths(output_dir=output_dir))
    typer.echo(f"Wrote filter_bode.png; points={len(result.frequencies_hz)}")
