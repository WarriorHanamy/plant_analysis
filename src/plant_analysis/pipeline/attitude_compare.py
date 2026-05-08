from __future__ import annotations

import numpy as np

from plant_analysis.core.schemas import AngleGrid
from plant_analysis.utils.so3_utils import geodesic_distance, ypr_to_rotation


def run(grid: AngleGrid, degrees: bool = True) -> np.ndarray:
    reference = np.eye(3)
    values = np.empty((len(grid.pitch), len(grid.roll), len(grid.yaw)), dtype=float)
    for i, pitch in enumerate(grid.pitch):
        for j, roll in enumerate(grid.roll):
            for k, yaw in enumerate(grid.yaw):
                values[i, j, k] = geodesic_distance(
                    reference, ypr_to_rotation(yaw, pitch, roll, degrees=degrees)
                )
    return values
