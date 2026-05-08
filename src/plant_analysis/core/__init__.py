"""Pure math API for plant analysis.

The core package has no file I/O and no plotting side effects. Public functions operate on
numpy arrays, schema objects, or python-control transfer functions only.
"""

from plant_analysis.core.bode import freqresp, freqresp_hz
from plant_analysis.core.filters import (
    boost,
    butter_lowpass,
    butterworth_lowpass,
    differentiator,
    notch_filter,
    pi_lead,
    pi_leadlag,
)
from plant_analysis.core.metrics import (
    bandwidth_3db,
    closed_loop_bandwidth,
    open_loop_margin,
    sensitivity_metrics,
    system_performance,
)
from plant_analysis.core.schemas import (
    AngleGrid,
    ControllerSynthesisInputPaths,
    ControllerSynthesisOutputPaths,
    DiagnosisOutputPaths,
    FilterChainReport,
    FilterResult,
    FilterSpec,
    PlantModel,
    SisoTransferFunction,
    SweepResult,
    TimeSeries,
    TuningConstraints,
    TuningTable,
)
from plant_analysis.core.signal import data_fft, kalman_filter, time_delay_vector
from plant_analysis.core.sysid import (
    fft_frd,
    fit_discrete_transfer,
    tfestimate,
    to_transfer_function,
)

__all__ = [
    "AngleGrid",
    "ControllerSynthesisInputPaths",
    "ControllerSynthesisOutputPaths",
    "DiagnosisOutputPaths",
    "FilterChainReport",
    "FilterResult",
    "FilterSpec",
    "PlantModel",
    "SisoTransferFunction",
    "SweepResult",
    "TimeSeries",
    "TuningConstraints",
    "TuningTable",
    "bandwidth_3db",
    "boost",
    "butter_lowpass",
    "butterworth_lowpass",
    "closed_loop_bandwidth",
    "data_fft",
    "differentiator",
    "fft_frd",
    "fit_discrete_transfer",
    "freqresp",
    "freqresp_hz",
    "kalman_filter",
    "notch_filter",
    "open_loop_margin",
    "pi_lead",
    "pi_leadlag",
    "sensitivity_metrics",
    "system_performance",
    "tfestimate",
    "time_delay_vector",
    "to_transfer_function",
]
