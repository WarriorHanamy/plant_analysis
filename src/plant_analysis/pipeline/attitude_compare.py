from __future__ import annotations

import numpy as np

from plant_analysis.core.schemas import AngleGrid, AttitudeOutputPaths
from plant_analysis.utils.plot_utils import save_contour, save_heatmap, save_time_plot
from plant_analysis.utils.so3_utils import geodesic_distance, ypr_to_rotation


def run(
    grid: AngleGrid,
    degrees: bool = True,
    output_paths: AttitudeOutputPaths | None = None,
) -> np.ndarray:
    reference = np.eye(3)
    values = np.empty((len(grid.pitch), len(grid.roll), len(grid.yaw)), dtype=float)
    for i, pitch in enumerate(grid.pitch):
        for j, roll in enumerate(grid.roll):
            for k, yaw in enumerate(grid.yaw):
                values[i, j, k] = geodesic_distance(
                    reference, ypr_to_rotation(yaw, pitch, roll, degrees=degrees)
                )

    if output_paths is None:
        output_paths = AttitudeOutputPaths()

    figs = output_paths.figs_dir
    figs.mkdir(parents=True, exist_ok=True)

    # 4.1 Error surface (2D slice at yaw=0)
    yaw_center = int(len(grid.yaw) // 2)
    save_heatmap(
        figs / "attitude_error_surface.png",
        grid.pitch,
        grid.roll,
        values[:, :, yaw_center],
        xlabel="Pitch",
        ylabel="Roll",
        clabel="Error [rad]",
    )

    # 4.2 Error contour for multiple yaw slices
    save_contour(
        figs / "attitude_error_contour.png",
        grid.pitch,
        grid.roll,
        values[:, :, yaw_center],
        xlabel="Pitch",
        ylabel="Roll",
    )

    # 4.3 Error profile: error vs pitch at roll=0, multiple yaw traces
    roll_center = int(len(grid.roll) // 2)
    save_time_plot(
        figs / "attitude_error_profile.png",
        grid.pitch,
        [("error", values[:, roll_center, 0])],
        ylabel="Error [rad]",
    )

    # 4.4 Max error across yaw for each pitch/roll
    max_error = values.max(axis=2)
    save_heatmap(
        figs / "attitude_error_max_slice.png",
        grid.pitch,
        grid.roll,
        max_error,
        xlabel="Pitch",
        ylabel="Roll",
        clabel="Max Error [rad]",
    )

    return values
