from ab_eval.core.utils import *


def test_get_segments_sample_size():
    df=generate_random_cvr_data(1000,0.3,0.5)
    assert get_segments_sample_size(df) == 1000
