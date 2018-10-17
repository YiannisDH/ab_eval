from ab_eval.core.experiment import evaluation_metrics




def test_primary_kpi_existance():
    metrics=evaluation_metrics('mCVR1')
    assert metrics.primary_KPI in metrics.kpis