from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

if TYPE_CHECKING:
    import control


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
    _apply_decade_grid(ax_mag)
    _apply_decade_grid(ax_phase)


def _apply_decade_grid(ax) -> None:
    ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.grid(which="minor", linestyle="--", linewidth=0.3, alpha=0.25)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs="all"))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())


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


def save_time_plot(
    path: str | Path,
    time: np.ndarray,
    signals: list[tuple[str, np.ndarray]],
    ylabel: str = "Amplitude",
    xlabel: str = "Time [s]",
) -> None:
    set_rcparams()
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    for label, values in signals:
        ax.plot(time, values, label=label, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_fft_magnitude_plot(
    path: str | Path, freq_hz: np.ndarray, magnitude_db: np.ndarray, label: str = ""
) -> None:
    set_rcparams()
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.semilogx(freq_hz, magnitude_db, linewidth=0.8, label=label)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    if label:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_step_response_plot(
    path: str | Path, system: control.TransferFunction, t_end: float | None = None
) -> None:
    set_rcparams()
    t, y = _step_response(system, t_end)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(t, y, linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Step Response")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_impulse_response_plot(
    path: str | Path, system: control.TransferFunction, t_end: float | None = None
) -> None:
    set_rcparams()
    t, y = _impulse_response(system, t_end)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.stem(t, y, linefmt="C0-", markerfmt=" ", basefmt="k-")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Impulse Response")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_heatmap(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
    clabel: str = "",
) -> None:
    set_rcparams()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    mesh = ax.pcolormesh(x, y, z.T, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    if clabel:
        cbar.set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_contour(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> None:
    set_rcparams()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.contour(x, y, z.T, levels=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_histogram(path: str | Path, data: np.ndarray, bins: int, xlabel: str) -> None:
    set_rcparams()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.hist(data, bins=bins, edgecolor="white", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_scatter2d(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray | None = None,
    xlabel: str = "",
    ylabel: str = "",
    clabel: str = "",
) -> None:
    set_rcparams()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    scatter = ax.scatter(x, y, c=c, s=8, alpha=0.7)
    if c is not None and clabel:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_scatter3d(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c: np.ndarray | None = None,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    clabel: str = "",
) -> None:
    set_rcparams()
    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(x, y, z, c=c, s=8, alpha=0.7)
    if c is not None and clabel:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_pole_zero_plot(path: str | Path, system: control.TransferFunction) -> None:
    set_rcparams()
    poles = system.poles()
    zeros = system.zeros()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(np.real(poles), np.imag(poles), marker="x", s=60, color="red", label="Poles")
    ax.scatter(
        np.real(zeros),
        np.imag(zeros),
        marker="o",
        s=50,
        facecolors="none",
        edgecolors="blue",
        label="Zeros",
    )
    import control as _ctrl

    if _ctrl.isdtime(system, strict=True):
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.6)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_nyquist_plot(
    path: str | Path, system: control.TransferFunction, max_hz: float = 100.0
) -> None:
    set_rcparams()
    frequencies = np.logspace(-1, np.log10(max_hz), 2000)
    omega = 2.0 * np.pi * frequencies
    import control as _ctrl

    if _ctrl.isdtime(system, strict=True):
        ts = system.dt if system.dt and system.dt is not True else 0.01
        omega = np.clip(omega, 0, np.pi / ts * 0.99)
    resp = system(1j * omega)
    resp = np.atleast_1d(resp)
    real = np.real(resp)
    imag = np.imag(resp)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(real, imag, linewidth=1.0)
    ax.scatter(real[0], imag[0], marker="o", color="green", s=40, zorder=5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.scatter([-1], [0], marker="x", color="red", s=60, zorder=5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_aspect("equal")
    ax.set_title("Nyquist Diagram")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_sensitivity_plot(
    path: str | Path, open_loop: control.TransferFunction, max_hz: float = 100.0
) -> None:
    set_rcparams()
    frequencies = np.logspace(-1, np.log10(max_hz), 1000)
    omega = 2.0 * np.pi * frequencies
    loop = np.atleast_1d(open_loop(1j * omega))
    s_resp = 1.0 / (1.0 + loop)
    t_resp = loop / (1.0 + loop)
    s_mag = 20.0 * np.log10(np.maximum(np.abs(s_resp), np.finfo(float).tiny))
    t_mag = 20.0 * np.log10(np.maximum(np.abs(t_resp), np.finfo(float).tiny))
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.semilogx(frequencies, s_mag, label="|S|", linewidth=1.0)
    ax.semilogx(frequencies, t_mag, label="|T|", linewidth=1.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bode_overlay(
    path: str | Path,
    systems: list[control.TransferFunction],
    labels: list[str],
    max_hz: float = 100.0,
) -> None:
    set_rcparams()
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.0))
    bode_style(ax_mag, ax_phase)
    frequencies = np.logspace(-1, np.log10(max_hz), 1000)
    omega = 2.0 * np.pi * frequencies
    for sys, label in zip(systems, labels):
        resp = np.atleast_1d(sys(1j * omega))
        mag = 20.0 * np.log10(np.maximum(np.abs(resp), np.finfo(float).tiny))
        phase = np.unwrap(np.angle(resp)) * 180.0 / np.pi
        ax_mag.plot(frequencies, mag, linewidth=0.8, label=label)
        ax_phase.plot(frequencies, phase, linewidth=0.8, label=label)
    ax_mag.legend(fontsize=7)
    ax_phase.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bar_chart(path: str | Path, labels: list[str], values: np.ndarray, ylabel: str) -> None:
    set_rcparams()
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.bar(labels, values, edgecolor="white", alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_spectrogram(
    path: str | Path,
    signal: np.ndarray,
    fs: float,
    title: str = "",
) -> None:
    set_rcparams()
    f, t, sxx = scipy_signal.spectrogram(signal, fs=fs, nperseg=min(256, len(signal)))
    sxx_db = 10.0 * np.log10(np.maximum(sxx, np.finfo(float).tiny))
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    mesh = ax.pcolormesh(t, f, sxx_db, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Power [dB]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _step_response(
    system: control.TransferFunction, t_end: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    import control as _ctrl

    if t_end is None:
        t_end = _default_t_end(system)
    if _ctrl.isdtime(system, strict=True):
        dt_val = system.dt if system.dt and system.dt is not True else 0.01
        t = np.arange(0, t_end + dt_val / 2, dt_val)
    else:
        t = np.linspace(0, t_end, 500)
    response = system.step_response(t)
    return np.asarray(response.time).ravel(), np.asarray(response.outputs).ravel()


def _impulse_response(
    system: control.TransferFunction, t_end: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    import control as _ctrl

    if t_end is None:
        t_end = _default_t_end(system)
    if _ctrl.isdtime(system, strict=True):
        dt_val = system.dt if system.dt and system.dt is not True else 0.01
        t = np.arange(0, t_end + dt_val / 2, dt_val)
    else:
        t = np.linspace(0, t_end, 500)
    response = system.impulse_response(t)
    return np.asarray(response.time).ravel(), np.asarray(response.outputs).ravel()


def _default_t_end(system: control.TransferFunction) -> float:
    poles = system.poles()
    dominant = np.min(np.abs(poles[np.abs(poles) > 1e-10]), initial=10.0)
    return float(min(5.0 / dominant, 10.0))
