import json
import logging
from ab_eval.core.utils import get_segments_sample_size , get_test_summary

logger = logging.getLogger(__name__)


class evaluation_metrics(object):
    """
    Class tha defines the KPIs that we need to use for evaluation
    :param kpis: the list of kpis
    :type  kpis: list of kpis
    :param primary_KPI: the primary KPI that will be used in every evaluation
    :type  primary_KPI: string
    """
    def __init__(
            self,
            kpis,
            primary_KPI="CVR",
            *args, **kwargs):
        super(evaluation_metrics, self).__init__(*args, **kwargs)
        #always append the business primary KPI
        self.primary_KPI=primary_KPI
        if not primary_KPI in kpis:
            kpis.append(primary_KPI)
        self.kpis = kpis

    def get_kpis(self):
        return  self.kpis


    def get_primary_KPI(self):
        return  self.primary_KPI


class variations(object):
    """
    Class that defines the variations characteristics inside the dataset
    :param column_name: the column name that contains the variation information inside the dataframe
    :type  column_name: string
    :param control_label: the name of the control group that can be found inside the variations column
    :type  control_label: string
    :param variation_label: the name of the variation group that can be found inside the variations column
    :type  control_label: string
    """
    def __init__(
            self,
            column_name='group',
            control_label='A',
            variation_label='B',
            *args, **kwargs):
        super(variations, self).__init__(*args, **kwargs)
        self.column_name=column_name
        self.control_label=control_label
        self.variation_label=variation_label

    def get_column_name(self):
        return self.column_name

    def get_control_label(self):
        return self.control_label

    def get_variation_label(self):
        return self.variation_label


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
        return {'control_label':self.variations.get_control_label(),
                'variation_label':self.variations.get_control_label()}