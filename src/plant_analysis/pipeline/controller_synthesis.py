from __future__ import annotations

import csv
from pathlib import Path

import control
import numpy as np

from plant_analysis.core.bode import freqresp_hz
from plant_analysis.core.metrics import system_performance
from plant_analysis.core.schemas import (
    ControllerSynthesisOutputPaths,
    PlantModel,
    SweepResult,
    TuningConstraints,
    TuningTable,
)
from plant_analysis.core.sysid import (
    fft_frd,
    fit_discrete_transfer,
    tfestimate,
    to_transfer_function,
)
from plant_analysis.utils.data_utils import load_npz, save_npz
from plant_analysis.utils.plot_utils import (
    save_bode_overlay,
    save_fft_magnitude_plot,
    save_histogram,
    save_impulse_response_plot,
    save_nyquist_plot,
    save_pole_zero_plot,
    save_scatter2d,
    save_scatter3d,
    save_sensitivity_plot,
    save_step_response_plot,
    save_time_plot,
)


def identify(sweep: SweepResult, na: int = 2, nb: int = 2) -> PlantModel:
    plant = fit_discrete_transfer(
        sweep.control.values, sweep.output_filtered.values, sweep.fs_hz, na=na, nb=nb
    )
    return PlantModel(
        transfer_function=plant.transfer_function,
        input_signal=sweep.control.signal_name,
        output_signal=sweep.output_filtered.signal_name,
        method=plant.method,
    )


def _grid(spec: tuple[float, float, int]) -> np.ndarray:
    start, stop, count = spec
    if count <= 1:
        return np.asarray([start], dtype=float)
    return np.linspace(start, stop, int(count))


def tune(plant: PlantModel, constraints: TuningConstraints | None = None) -> TuningTable:
    constraints = constraints or TuningConstraints()
    system = to_transfer_function(plant)
    parameters: list[dict[str, float]] = []
    metrics: list[dict[str, float]] = []
    status: list[str] = []
    for kp in _grid(constraints.kp):
        for ki in _grid(constraints.ki):
            for kd in _grid(constraints.kd):
                controller = control.TransferFunction([kd, kp, ki], [1.0, 0.0], system.dt)
                perf = system_performance(
                    controller * system, max_hz=(plant.fs_hz or 200.0) / 2.0 * 0.9
                )
                ok = (
                    np.isnan(perf["phase_margin_deg"])
                    or perf["phase_margin_deg"] >= constraints.min_phase_margin_deg
                ) and perf["max_sensitivity_db"] <= constraints.max_sensitivity_db
                parameters.append({"Kp": float(kp), "Ki": float(ki), "Kd": float(kd)})
                metrics.append(perf)
                status.append("pass" if ok else "fail")
    return TuningTable(parameters=parameters, metrics=metrics, status=status)


def write_tuning_csv(path: str | Path, table: TuningTable) -> None:
    fields = sorted({key for row in table.parameters + table.metrics for key in row}) + ["status"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for params, metric, state in zip(
            table.parameters, table.metrics, table.status, strict=True
        ):
            writer.writerow({**params, **metric, "status": state})


def _reconstruct_lsq_prediction(
    u: np.ndarray, y: np.ndarray, na: int, nb: int, delay: int, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    start = max(na, nb + delay)
    rows = []
    targets = []
    for k in range(start, len(y)):
        row = [-y[k - i] for i in range(1, na + 1)]
        row.extend(u[k - delay - j] for j in range(nb + 1))
        rows.append(row)
        targets.append(y[k])
    x_mat = np.asarray(rows)
    y_pred = x_mat @ theta
    return np.asarray(targets), y_pred


def _plot_sysid_diagnostics(
    sweep: SweepResult, plant: PlantModel, paths: ControllerSynthesisOutputPaths
) -> None:
    figs = paths.sysid_figs_dir
    figs.mkdir(parents=True, exist_ok=True)
    u = sweep.control.values
    y = sweep.output_filtered.values
    fs = sweep.fs_hz
    sys = to_transfer_function(plant)
    tf = plant.transfer_function

    # 2.1 Bode comparison: fitted TF vs tfestimate vs fft_frd
    f_tfe, h_tfe, coh = tfestimate(u, y, fs)
    f_fft, h_fft = fft_frd(u, y, fs)
    f_eval = np.logspace(np.log10(max(f_tfe[1], 0.1)), np.log10(min(fs / 2 * 0.95, f_tfe[-1])), 800)
    mag_fit, phase_fit = freqresp_hz(sys, f_eval)
    mag_tfe = 20.0 * np.log10(np.maximum(np.abs(h_tfe), 1e-15))
    mag_fft = 20.0 * np.log10(np.maximum(np.abs(h_fft), 1e-15))
    phase_tfe = np.unwrap(np.angle(h_tfe)) * 180.0 / np.pi
    phase_fft = np.unwrap(np.angle(h_fft)) * 180.0 / np.pi

    # Custom Bode overlay with raw FRD + TF
    import matplotlib.pyplot as plt

    from plant_analysis.utils.plot_utils import bode_subplot_grid

    fig, ax_mag, ax_phase = bode_subplot_grid()
    ax_mag.plot(f_eval, mag_fit, "C0-", linewidth=1.0, label="fitted TF")
    ax_mag.plot(f_tfe, mag_tfe, "C1--", linewidth=0.6, label="tfestimate")
    ax_mag.plot(f_fft, mag_fft, "C2:", linewidth=0.5, label="FFT FRD")
    ax_phase.plot(f_eval, phase_fit, "C0-", linewidth=1.0, label="fitted TF")
    ax_phase.plot(f_tfe, phase_tfe, "C1--", linewidth=0.6, label="tfestimate")
    ax_phase.plot(f_fft, phase_fft, "C2:", linewidth=0.5, label="FFT FRD")
    ax_mag.legend(fontsize=7)
    ax_phase.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "plant_bode_comparison.png")
    plt.close(fig)

    # 2.2 Coherence
    save_fft_magnitude_plot(figs / "plant_coherence.png", f_tfe, coh, label="coherence")

    # 2.3 Impulse response
    save_impulse_response_plot(figs / "plant_impulse_response.png", sys)

    # 2.4 Step response
    save_step_response_plot(figs / "plant_step_response.png", sys)

    # 2.5 Pole-zero map
    save_pole_zero_plot(figs / "plant_pole_zero_map.png", sys)

    # 2.6 Model vs measured time-domain
    t_sim, y_sim = control.forced_response(sys, T=sweep.time_s, U=u, X0=0.0)
    y_sim = np.asarray(y_sim).ravel()
    save_time_plot(
        figs / "plant_compare_time.png",
        sweep.time_s,
        [("measured y", y), ("simulated y", y_sim)],
    )

    # 2.7 Residuals
    residuals = y - y_sim
    save_time_plot(
        figs / "plant_residuals.png",
        sweep.time_s,
        [("residual", residuals)],
        ylabel="Residual",
    )

    # 2.8 LS regression fit scatter
    na_val = len(tf.denominator) - 1
    nb_val = len(tf.numerator) - 1
    theta = np.r_[tf.denominator[1:], tf.numerator]
    targets, y_pred = _reconstruct_lsq_prediction(u, y, na_val, nb_val, 0, theta)
    save_scatter2d(
        figs / "plant_lsq_regression_fit.png",
        y_pred,
        targets,
        xlabel="Predicted y[k]",
        ylabel="Actual y[k]",
    )
    # Overlay identity line
    lim_min = min(y_pred.min(), targets.min())
    lim_max = max(y_pred.max(), targets.max())
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(y_pred, targets, s=8, alpha=0.7)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.6)
    ax.set_xlabel("Predicted y[k]")
    ax.set_ylabel("Actual y[k]")
    fig.tight_layout()
    fig.savefig(figs / "plant_lsq_regression_fit.png")
    plt.close(fig)


def _plot_tuning_diagnostics(
    plant: PlantModel, table: TuningTable, paths: ControllerSynthesisOutputPaths
) -> None:
    figs = paths.tuning_figs_dir
    figs.mkdir(parents=True, exist_ok=True)
    sys = to_transfer_function(plant)
    max_hz = (plant.fs_hz or 200.0) / 2.0 * 0.95

    kp_vals = np.array([p["Kp"] for p in table.parameters])
    ki_vals = np.array([p["Ki"] for p in table.parameters])
    kd_vals = np.array([p["Kd"] for p in table.parameters])
    pm_vals = np.array([m["phase_margin_deg"] for m in table.metrics])
    gm_vals = np.array([m["gain_margin_db"] for m in table.metrics])
    s_vals = np.array([m["max_sensitivity_db"] for m in table.metrics])
    bw_vals = np.array([m["closed_loop_bandwidth_hz"] for m in table.metrics])
    pass_mask = np.array([s == "pass" for s in table.status])

    # 3.1 Kp-Ki heatmap (fixed kd=0)
    kd0_mask = np.abs(kd_vals) < 1e-10
    if np.any(kd0_mask):
        save_scatter2d(
            figs / "tuning_grid_kp_ki.png",
            kp_vals[kd0_mask],
            ki_vals[kd0_mask],
            c=pm_vals[kd0_mask],
            xlabel="Kp",
            ylabel="Ki",
            clabel="Phase Margin [deg]",
        )

    # 3.2-3.4 Histograms
    save_histogram(
        figs / "tuning_gain_margin_hist.png", gm_vals[np.isfinite(gm_vals)], 30, "Gain Margin [dB]"
    )
    save_histogram(
        figs / "tuning_phase_margin_hist.png",
        pm_vals[np.isfinite(pm_vals)],
        30,
        "Phase Margin [deg]",
    )
    save_histogram(
        figs / "tuning_sensitivity_hist.png",
        s_vals[np.isfinite(s_vals)],
        30,
        "Max Sensitivity [dB]",
    )

    # 3.5 Pass/fail 3D scatter
    colors = np.where(pass_mask, 1, 0)
    save_scatter3d(
        figs / "tuning_pass_fail_3d.png",
        kp_vals,
        ki_vals,
        kd_vals,
        c=colors,
        xlabel="Kp",
        ylabel="Ki",
        zlabel="Kd",
        clabel="pass=1 / fail=0",
    )

    # 3.6 Bandwidth vs phase margin
    save_scatter2d(
        figs / "tuning_bandwidth_vs_margin.png",
        bw_vals,
        pm_vals,
        c=colors,
        xlabel="Bandwidth [Hz]",
        ylabel="Phase Margin [deg]",
        clabel="pass",
    )

    # 3.7-3.10 Best candidate plots
    if np.any(pass_mask):
        best_idx = int(np.argmax(pm_vals[pass_mask]))
        best_params = [table.parameters[i] for i, p in enumerate(pass_mask) if p][best_idx]
        best_kp, best_ki, best_kd = best_params["Kp"], best_params["Ki"], best_params["Kd"]
        controller_tf = control.TransferFunction([best_kd, best_kp, best_ki], [1.0, 0.0], sys.dt)
        open_loop_tf = controller_tf * sys
        save_bode_overlay(
            figs / "tuning_loop_bode_best.png", [open_loop_tf], ["best L"], max_hz=max_hz
        )
        save_sensitivity_plot(figs / "tuning_sensitivity_best.png", open_loop_tf, max_hz=max_hz)
        closed_loop_tf = control.feedback(open_loop_tf, 1)
        save_step_response_plot(figs / "tuning_step_response_best.png", closed_loop_tf)
        save_nyquist_plot(figs / "tuning_nyquist_best.png", open_loop_tf, max_hz=max_hz)


def run(paths: ControllerSynthesisOutputPaths | None = None) -> TuningTable:
    paths = paths or ControllerSynthesisOutputPaths()
    sweep = load_npz(paths.sweep_npz, SweepResult)
    plant = identify(sweep)
    table = tune(plant)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(paths.plant_model_npz, plant)
    write_tuning_csv(paths.tuning_table_csv, table)

    _plot_sysid_diagnostics(sweep, plant, paths)
    _plot_tuning_diagnostics(plant, table, paths)

    return table
