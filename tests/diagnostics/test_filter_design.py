from plant_analysis.core.schemas import DiagnosisOutputPaths, FilterSpec
from plant_analysis.diagnostics.filter_design import run


def test_filter_design_writes_all_figs(tmp_path):
    paths = DiagnosisOutputPaths(output_dir=tmp_path)
    result = run(FilterSpec(filter_type="butterworth", order=2, cutoff=20, fs=200), paths)
    assert result.frequencies_hz.size > 0

    assert paths.filter_bode_png.exists()
    assert paths.filter_step_png.exists()
    assert paths.filter_impulse_png.exists()
    assert paths.filter_pole_zero_png.exists()
    assert paths.filter_group_delay_png.exists()
    assert paths.filter_phase_delay_png.exists()
    assert paths.filter_time_sweep_png.exists()
    assert paths.filter_cascade_png.exists()

    for png in [
        paths.filter_bode_png,
        paths.filter_step_png,
        paths.filter_impulse_png,
        paths.filter_pole_zero_png,
        paths.filter_group_delay_png,
        paths.filter_phase_delay_png,
        paths.filter_time_sweep_png,
        paths.filter_cascade_png,
    ]:
        assert png.stat().st_size > 0, f"{png.name} is empty"
