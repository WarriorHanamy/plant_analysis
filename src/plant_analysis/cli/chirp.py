from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from plant_analysis.core.schemas import (
    ControllerSynthesisInputPaths,
    ControllerSynthesisOutputPaths,
)
from plant_analysis.pipeline.chirp import run


def app(
    input_dir: Annotated[Path, typer.Option("--input-dir", exists=True, readable=True)] = Path(
        "_data_input"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("_output"),
) -> None:
    sweep = run(
        ControllerSynthesisInputPaths(input_dir=input_dir),
        ControllerSynthesisOutputPaths(output_dir=output_dir),
    )
    typer.echo(f"SweepResult samples={len(sweep.time_s)} fs={sweep.fs_hz:.3f}")
