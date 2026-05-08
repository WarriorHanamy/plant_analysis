from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def set_rcparams() -> None:
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 120,
            "savefig.dpi": 160,
        }
    )


def bode_style(ax_mag, ax_phase) -> None:
    ax_mag.set_xscale("log")
    ax_phase.set_xscale("log")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.set_xlabel("Frequency [Hz]")


def bode_subplot_grid():
    set_rcparams()
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.0))
    bode_style(ax_mag, ax_phase)
    return fig, ax_mag, ax_phase


def save_bode_plot(path: str | Path, frequencies_hz, magnitude_db, phase_deg) -> None:
    fig, ax_mag, ax_phase = bode_subplot_grid()
    ax_mag.plot(frequencies_hz, magnitude_db)
    ax_phase.plot(frequencies_hz, phase_deg)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
