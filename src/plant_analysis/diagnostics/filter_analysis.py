from __future__ import annotations

from pathlib import Path

import numpy as np

from plant_analysis.core.schemas import DiagnosisOutputPaths, FilterChainReport
from plant_analysis.utils.plot_utils import save_bar_chart
from plant_analysis.utils.ulog_utils import _ulog_spectra, read_ulog_filter_chain


def run(
    ulog_path: str | Path,
    paths: DiagnosisOutputPaths | None = None,
) -> FilterChainReport:
    report = read_ulog_filter_chain(ulog_path)
    spectra = _ulog_spectra(ulog_path)

    if paths is None:
        paths = DiagnosisOutputPaths()

    figs = paths.filter_chain_figs_dir
    figs.mkdir(parents=True, exist_ok=True)

    # 6.1 FFT magnitude overlay for all gyro/rate signals
    if spectra:
        import matplotlib.pyplot as plt

        from plant_analysis.utils.plot_utils import set_rcparams

        set_rcparams()
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        for name, (freq, mag) in spectra.items():
            ax.semilogx(freq, mag, linewidth=0.6, label=name)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figs / "filter_chain_fft_overlay.png")
        plt.close(fig)

    # 6.2 FFT peaks bar chart
    if report.fft_peaks:
        names = list(report.fft_peaks)
        peaks = np.array([report.fft_peaks[n] for n in names])
        peaks_db = 20.0 * np.log10(np.maximum(peaks, 1e-15))
        save_bar_chart(
            figs / "filter_chain_fft_peaks_bar.png",
            names,
            peaks_db,
            ylabel="Peak Magnitude [dB]",
        )

    # 6.3 Spectrogram (first available signal)
    if spectra:
        first_name = next(iter(spectra))
        freq_data, _ = spectra[first_name]
        ulog = _open_ulog(ulog_path)
        for dataset in ulog.data_list:
            for name, values in dataset.data.items():
                if "gyro" in name.lower() or "rate" in name.lower():
                    arr = np.asarray(values, dtype=float)
                    if arr.size < 32:
                        continue
                    from plant_analysis.utils.plot_utils import save_spectrogram

                    fs_est = 1.0 / float(
                        np.median(
                            np.diff(
                                np.asarray(dataset.data.get("timestamp", [0, 1000]), dtype=float)
                                * 1e-6
                            )
                        )
                    )
                    save_spectrogram(
                        figs / "filter_chain_spectrogram.png",
                        arr,
                        fs_est,
                        title=f"Spectrogram: {name}",
                    )
                    break
            break

    # 6.4 Noise floor per signal
    if spectra:
        noise_floor = {}
        for name, (freq, mag) in spectra.items():
            noise_floor[name] = float(np.median(mag))
        nf_names = list(noise_floor)
        nf_vals = np.array([noise_floor[n] for n in nf_names])
        save_bar_chart(
            figs / "filter_chain_noise_floor.png",
            nf_names,
            nf_vals,
            ylabel="Noise Floor [dB]",
        )

    return report


def _open_ulog(path: str | Path):
    from pyulog import ULog

    return ULog(str(path))
