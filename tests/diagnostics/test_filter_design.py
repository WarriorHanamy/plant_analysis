from plant_analysis.core.schemas import DiagnosisOutputPaths, FilterSpec
from plant_analysis.diagnostics.filter_design import run


def test_filter_design_writes_bode_plot(tmp_path):
    paths = DiagnosisOutputPaths(output_dir=tmp_path)
    result = run(FilterSpec(filter_type="butterworth", order=2, cutoff=20, fs=200), paths)
    assert result.frequencies_hz.size > 0
    assert paths.filter_bode_png.exists()
