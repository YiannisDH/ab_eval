from ab_eval.core.experiment_components import evaluation_metrics, variations
from ab_eval.core.experiment import experiment
from ab_eval.core.utils import generate_random_cvr_data


def test_primary_kpi_existance():
    metrics = evaluation_metrics('mCVR1')
    assert metrics.primary_KPI in metrics.kpis


def test_variations():
    variation = variations(variation_label='control')
    assert variation.variation_label == 'control'


def test_get_p_value_on_random_data():
    df = generate_random_cvr_data(1000, 0.3, 0.4, days=10)
    exp = experiment(df, segments=['new', 'returning'])
    assert exp.get_p_val() is not None
