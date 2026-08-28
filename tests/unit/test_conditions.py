from oceans_glwl.analysis.selection import enumerate_conditions
from oceans_glwl.config import WindowConfig, input_offsets_for_window


def test_condition_space_is_historically_compatible() -> None:
    conditions = enumerate_conditions()
    assert len(conditions) == 336
    assert len({item.condition_id for item in conditions}) == 336
    assert {item.model_name for item in conditions} == {"AutoLSTM", "AutoNBEATSx", "AutoNHITS", "AutoTFT"}
    assert {item.input_variant for item in conditions} == {
        "daily_smoothed_y_raw_exog", "daily_smoothed_y_smoothed_exog", "sparse_30d_smoothed_all"
    }
    assert {item.context_days for item in conditions} == {180, 360, 540, 720}
    assert len({item.output_mode for item in conditions}) == 7


def test_sparse_context_offsets_are_6_12_18_24() -> None:
    observed = [len(input_offsets_for_window(context_days=context, input_gap_days=30)) for context in (180, 360, 540, 720)]
    assert observed == [6, 12, 18, 24]
    assert WindowConfig(output_mode="seq2seq_6x30d").lead_days == (30, 60, 90, 120, 150, 180)

