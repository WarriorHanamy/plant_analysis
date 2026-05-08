from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type FloatArray1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type FloatMatrix2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
DEFAULT_INPUT_DIR = Path("_data_input")
DEFAULT_OUTPUT_DIR = Path("_output")

__all__ = [
    "AngleGrid",
    "ControllerSynthesisInputPaths",
    "ControllerSynthesisOutputPaths",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DiagnosisOutputPaths",
    "FilterChainReport",
    "FilterResult",
    "FilterSpec",
    "FloatArray1D",
    "FloatMatrix2D",
    "NumericModel",
    "PlantModel",
    "SisoTransferFunction",
    "SweepResult",
    "TimeSeries",
    "TuningConstraints",
    "TuningTable",
]


def _finite_array_1d(value: Any, name: str = "array") -> FloatArray1D:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


class NumericModel(BaseModel):
    """Base Pydantic model for numeric schemas that can be flattened to NPZ arrays."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def as_npz_dict(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in type(self).model_fields}


class TimeSeries(NumericModel):
    """One scalar signal sampled at a matching time vector."""

    time_s: FloatArray1D = Field(description="Monotonic sample times with shape (n,), unit [s]")
    values: FloatArray1D = Field(
        description="Signal samples with shape (n,), one value at each time_s[k]"
    )
    signal_name: str = Field(description="Human-readable signal name")
    unit: str = Field(default="", description="Physical unit of values")

    @field_validator("time_s", "values", mode="before")
    @classmethod
    def _validate_vector(cls, value: Any) -> FloatArray1D:
        return _finite_array_1d(value)

    @model_validator(mode="after")
    def _validate_time_series(self) -> TimeSeries:
        if self.time_s.shape != self.values.shape:
            raise ValueError("time_s and values must both have shape (n,)")
        if np.any(np.diff(self.time_s) <= 0):
            raise ValueError("time_s must be strictly increasing")
        return self


class SweepResult(NumericModel):
    """Aligned chirp sweep signals; every TimeSeries value is data at time_s[k]."""

    chirp: TimeSeries
    reference: TimeSeries
    control: TimeSeries
    output_filtered: TimeSeries
    output_raw: TimeSeries
    fs_hz: float = Field(gt=0, description="Nominal sampling frequency [Hz]")

    @model_validator(mode="after")
    def _validate_alignment(self) -> SweepResult:
        base = self.chirp.time_s
        for signal in (self.reference, self.control, self.output_filtered, self.output_raw):
            if signal.time_s.shape != base.shape or not np.allclose(signal.time_s, base):
                raise ValueError("all sweep TimeSeries objects must share the same time_s vector")
        return self

    @property
    def time_s(self) -> FloatArray1D:
        return self.chirp.time_s

    def as_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "time_s": self.time_s,
            "chirp": self.chirp.values,
            "reference": self.reference.values,
            "control": self.control.values,
            "output_filtered": self.output_filtered.values,
            "output_raw": self.output_raw.values,
            "fs_hz": np.asarray(self.fs_hz),
        }


class ControllerSynthesisInputPaths(BaseModel):
    """Input file contract for the controller synthesis flow."""

    input_dir: Path = Field(
        default=DEFAULT_INPUT_DIR, description="Directory for source input data"
    )
    recorded_csv_name: str = "recorded_data.csv"

    @property
    def recorded_csv(self) -> Path:
        return self.input_dir / self.recorded_csv_name


class ControllerSynthesisOutputPaths(BaseModel):
    """Output file contract for the controller synthesis flow."""

    output_dir: Path = Field(
        default=DEFAULT_OUTPUT_DIR, description="Directory for generated artifacts"
    )
    sweep_npz_name: str = "sweep_result.npz"
    plant_model_npz_name: str = "plant_id.npz"
    tuning_table_csv_name: str = "tuning_table.csv"

    @property
    def sweep_npz(self) -> Path:
        return self.output_dir / self.sweep_npz_name

    @property
    def plant_model_npz(self) -> Path:
        return self.output_dir / self.plant_model_npz_name

    @property
    def tuning_table_csv(self) -> Path:
        return self.output_dir / self.tuning_table_csv_name


class DiagnosisOutputPaths(BaseModel):
    """Output file contract for diagnostics reports and figures."""

    output_dir: Path = Field(
        default=DEFAULT_OUTPUT_DIR, description="Directory for generated artifacts"
    )
    figs_dir_name: str = "figs"
    filter_bode_png_name: str = "filter_bode.png"
    filter_step_png_name: str = "filter_step.png"
    filter_report_npz_name: str = "filter_report.npz"

    @property
    def figs_dir(self) -> Path:
        return self.output_dir / self.figs_dir_name

    @property
    def filter_bode_png(self) -> Path:
        return self.figs_dir / self.filter_bode_png_name

    @property
    def filter_step_png(self) -> Path:
        return self.figs_dir / self.filter_step_png_name

    @property
    def filter_report_npz(self) -> Path:
        return self.output_dir / self.filter_report_npz_name


class SisoTransferFunction(NumericModel):
    """Single-input single-output transfer function coefficients."""

    numerator: FloatArray1D = Field(
        description="Numerator b coefficients with shape (nb + 1,), descending powers"
    )
    denominator: FloatArray1D = Field(
        description="Denominator a coefficients with shape (na + 1,), denominator[0] != 0"
    )
    domain_variable: Literal["s", "z^-1"] = Field(
        description="Transfer-function domain variable used by the coefficients"
    )
    sample_time_s: float | None = Field(
        default=None, gt=0, description="Discrete sample time [s]; None for continuous models"
    )
    coefficient_order: Literal["descending"] = "descending"

    @field_validator("numerator", "denominator", mode="before")
    @classmethod
    def _validate_coefficients(cls, value: Any) -> FloatArray1D:
        return _finite_array_1d(value, "transfer-function coefficients")

    @model_validator(mode="after")
    def _validate_transfer_function(self) -> SisoTransferFunction:
        if np.isclose(self.denominator[0], 0.0):
            raise ValueError("denominator[0] must be non-zero")
        if self.domain_variable == "s" and self.sample_time_s is not None:
            raise ValueError("continuous transfer functions must not set sample_time_s")
        if self.domain_variable == "z^-1" and self.sample_time_s is None:
            raise ValueError("discrete transfer functions must set sample_time_s")
        return self


class PlantModel(NumericModel):
    """Identified plant with explicit input/output channels and transfer-function shape."""

    transfer_function: SisoTransferFunction
    input_signal: str = "control"
    output_signal: str = "output_filtered"
    method: str = "unknown"

    @property
    def fs_hz(self) -> float | None:
        if self.transfer_function.sample_time_s is None:
            return None
        return 1.0 / self.transfer_function.sample_time_s

    def as_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "numerator": self.transfer_function.numerator,
            "denominator": self.transfer_function.denominator,
            "sample_time_s": np.asarray(self.transfer_function.sample_time_s or np.nan),
            "domain_variable": np.asarray(self.transfer_function.domain_variable),
            "input_signal": np.asarray(self.input_signal),
            "output_signal": np.asarray(self.output_signal),
            "method": np.asarray(self.method),
        }


class FilterSpec(BaseModel):
    """Filter synthesis request with frequency parameters in Hz."""

    filter_type: Literal["butterworth", "notch", "pi_lead", "pi_leadlag", "boost", "differentiator"]
    order: int = Field(default=2, ge=0, le=8)
    cutoff: float = Field(gt=0)
    bandwidth: float | None = Field(default=None, gt=0)
    gain: float = Field(default=1.0, gt=0)
    fs: float | None = Field(default=None, gt=0)


class TuningTable(BaseModel):
    """Controller candidate table with one parameter row, metric row, and status per candidate."""

    parameters: list[dict[str, float]]
    metrics: list[dict[str, float]]
    status: list[str]

    @model_validator(mode="after")
    def _same_length(self) -> TuningTable:
        if not (len(self.parameters) == len(self.metrics) == len(self.status)):
            raise ValueError("parameters, metrics, and status must have the same length")
        return self


class FilterChainReport(BaseModel):
    """Frequency-domain diagnostic summary for named signals extracted from a log."""

    signal_names: list[str]
    fft_peaks: dict[str, float]
    spectrogram_data: dict[str, list[list[float]]] = Field(default_factory=dict)


class AngleGrid(NumericModel):
    """Pitch, roll, and yaw sample vectors used to evaluate attitude error surfaces."""

    pitch: FloatArray1D = Field(
        description="Pitch grid samples with shape (n,), unit [deg] or [rad]"
    )
    roll: FloatArray1D = Field(description="Roll grid samples with shape (n,), unit [deg] or [rad]")
    yaw: FloatArray1D = Field(description="Yaw grid samples with shape (n,), unit [deg] or [rad]")

    @field_validator("pitch", "roll", "yaw", mode="before")
    @classmethod
    def _validate_array(cls, value: Any) -> FloatArray1D:
        return _finite_array_1d(value)


class TuningConstraints(BaseModel):
    """PID grid search ranges and minimum performance limits."""

    kp: tuple[float, float, int] = (0.01, 1.0, 12)
    ki: tuple[float, float, int] = (0.0, 0.2, 5)
    kd: tuple[float, float, int] = (0.0, 0.0, 1)
    min_phase_margin_deg: float = 30.0
    max_sensitivity_db: float = 12.0


class FilterResult(NumericModel):
    """Synthesized filter coefficients and sampled Bode response."""

    numerator: FloatArray1D = Field(
        description="Filter numerator coefficients with shape (nb + 1,)"
    )
    denominator: FloatArray1D = Field(
        description="Filter denominator coefficients with shape (na + 1,)"
    )
    frequencies_hz: FloatArray1D = Field(
        description="Bode sample frequencies with shape (n,), unit [Hz]"
    )
    magnitude_db: FloatArray1D = Field(
        description="Bode magnitude samples with shape (n,), unit [dB]"
    )
    phase_deg: FloatArray1D = Field(description="Bode phase samples with shape (n,), unit [deg]")

    @field_validator(
        "numerator", "denominator", "frequencies_hz", "magnitude_db", "phase_deg", mode="before"
    )
    @classmethod
    def _validate_array(cls, value: Any) -> FloatArray1D:
        return _finite_array_1d(value)

    @model_validator(mode="after")
    def _validate_response_shapes(self) -> FilterResult:
        response_shape = self.frequencies_hz.shape
        if self.magnitude_db.shape != response_shape or self.phase_deg.shape != response_shape:
            raise ValueError("frequencies_hz, magnitude_db, and phase_deg must share shape (n,)")
        return self
