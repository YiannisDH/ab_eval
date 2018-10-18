from ab_eval.core.experiment import evaluation_metrics , variations




def test_primary_kpi_existance():
    metrics=evaluation_metrics('mCVR1')
    assert metrics.primary_KPI in metrics.kpis

def test_variations():
    variation=variations(variation_label='control')
    assert variation.variation_label == 'control'