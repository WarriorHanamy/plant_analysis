from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from plant_analysis.core.schemas import ControllerSynthesisOutputPaths
from plant_analysis.pipeline.controller_synthesis import run


def app(
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("_output"),
) -> None:
    paths = ControllerSynthesisOutputPaths(output_dir=output_dir)
    table = run(paths)
    pass_count = sum(state == "pass" for state in table.status)
    typer.echo(f"Wrote tuning outputs to {output_dir}; pass={pass_count}/{len(table.status)}")
