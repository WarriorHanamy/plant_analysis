from __future__ import annotations

from typing import Annotated

import numpy as np
import typer

from plant_analysis.core.schemas import AngleGrid
from plant_analysis.pipeline.attitude_compare import run


def app(
    pitch_min: Annotated[float, typer.Option("--pitch-min")] = -30.0,
    pitch_max: Annotated[float, typer.Option("--pitch-max")] = 30.0,
    roll_min: Annotated[float, typer.Option("--roll-min")] = -30.0,
    roll_max: Annotated[float, typer.Option("--roll-max")] = 30.0,
    yaw_min: Annotated[float, typer.Option("--yaw-min")] = -30.0,
    yaw_max: Annotated[float, typer.Option("--yaw-max")] = 30.0,
    points: Annotated[int, typer.Option("--points", min=2)] = 11,
) -> None:
    grid = AngleGrid(
        pitch=np.linspace(pitch_min, pitch_max, points),
        roll=np.linspace(roll_min, roll_max, points),
        yaw=np.linspace(yaw_min, yaw_max, points),
    )
    values = run(grid)
    typer.echo(f"Angle error grid shape={values.shape} max_rad={float(values.max()):.6f}")
