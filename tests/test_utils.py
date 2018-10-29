from ab_eval.core.utils import generate_random_cvr_data, get_segments_sample_size, get_test_summary


def test_get_segments_sample_size():
    df = generate_random_cvr_data(1000, 0.3, 0.5)
    assert get_segments_sample_size(df) == 1000


def test_get_test_summary_without_segment():
    df = generate_random_cvr_data(1000, 0.3, 0.5, days=10, control_label='control', variation_label='variation')
    df1 = get_test_summary(df, 'CVR')
    assert df1['CVR'].control is not None


def test_get_test_summary_with_segment():
    df = generate_random_cvr_data(1000, 0.3, 0.5, days=10, control_label='control', variation_label='variation')
    df1 = get_test_summary(df, 'CVR', segment='new')
    assert df1['CVR'].control == df1['CVR'].control     # trick with NaN != NaN
