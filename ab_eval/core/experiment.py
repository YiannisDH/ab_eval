import json
from ab_eval.core.experiment_components import variations,evaluation_metrics
from ab_eval.core.utils import get_test_summary

class experiment(object):
    """
    Class that defines an experiment and all its characteristics
    :param data: the dataframe with data.
    :type  data: dataframe.
    :param kpis: evaluation_metrics object that holds information about the kpis that gonna be used for the evaluation.
    :type  kpis: evaluation_metrics
    :param variations: variations object that holds information about the variations of the test.
    :type  variatios: variations
    :param segments: list of segments that will be used for a specific segment evaluation
    :type  segments: list of strings
    """
    def __init__(
            self,
            data,
            kpis=evaluation_metrics(kpis=["CVR"]),
            variations= variations(),
            segments=[],
            *args, **kwargs):
        super(experiment, self).__init__(*args, **kwargs)
        self.data=data
        self.kpis=kpis
        self.variations=variations
        self.segments=segments

    def get_data(self):
        return self.data

    def get_expirement_kpis(self):
        return self.kpis.get_kpis()

    def get_experiment_column_name(self):
        return self.variations.get_column_name()

    def get_segments(self):
        return  self.segments

    def get_experiment_variations(self):
        return json.dumps({'control_label':self.variations.get_control_label(),
                'variation_label':self.variations.get_control_label()})


