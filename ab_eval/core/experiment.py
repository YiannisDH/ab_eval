import json
import logging
from ab_eval.core.experiment_components import variations, evaluation_metrics
from ab_eval.core.utils import get_test_summary, get_standard_error
import scipy.stats as scs

logger = logging.getLogger(__name__)


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
    :param significance_level: the significance level that should be used in the experiment
    :type  significance_level: float
    """
    def __init__(
            self,
            data,
            kpis=evaluation_metrics(kpis=["CVR"]),
            variations=variations(),
            segments=None,
            significance_level=0.05,
            *args, **kwargs):
        super(experiment, self).__init__(*args, **kwargs)
        self.data = data
        self.kpis = kpis
        self.variations = variations
        self.segments = segments
        if significance_level > 1:
            raise ValueError("significance_level should be >0 and <1 : {}")
        self.significance_level = significance_level

    def get_data(self):
        return self.data

    def get_expirement_kpis(self):
        return self.kpis.get_kpis()

    def get_experiment_column_name(self):
        return self.variations.get_column_name()

    def get_segments(self):
        return self.segments

    def get_experiment_variations(self):
        return json.dumps({'control_label': self.variations.get_control_label(), 'variation_label': self.variations.get_control_label()})

    def get_p_val(self, kpi='CVR', segment=None, segment_column='segment', variation_column='group'):
        """Method that calculates the p-value for a given dataset and KPI


        :param   kpi: the KPI that should be used
        :param   segment: the segment that should be used
        :param   segment_column: the column name that contains the segment information
        :param   variation_column: the column name that contains the variation information
        :return: the p value
        :rtype:  float

        """

        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column, variations_column=variation_column)

        return scs.binom(df_summary['total'][self.variations.variation_label],
                         df_summary['rate'][self.variations.variation_label])\
            .pmf(df_summary['rate'][self.variations.control_label] * df_summary['total'][self.variations.control_label])

    def get_relative_conversion_uplift(self, kpi='CVR', segment=None, segment_column='segment', variation_column='group'):
        """Method that calculates the relative conversion_uplift

        :param   kpi: the KPI that should be used
        :param   segment: the segment that should be used
        :param   segment_column: the column name that contains the segment information
        :param   variation_column: the column name that contains the variation information
        :return: the relative conversion uplift
        :rtype:  float
        """
        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column, variations_column=variation_column)

        return (df_summary['rate'][self.variations.variation_label] - df_summary['rate'][self.variations.control_label]) / \
            df_summary['rate'][self.variations.control_label]

    def get_standard_errors_of_test(self, kpi='CVR', segment=None, segment_column='segment', variation_column='group'):
        """
        This method is calculating the standard error for variation and control and returns a tuple where the first
        element as the standard error of control and the second as the standard error of variation

        :param   kpi: the KPI that should be used
        :param   segment: the segment that should be used
        :param   segment_column: the column name that contains the segment information
        :param   variation_column: the column name that contains the variation information
        :return: standard error for variation and control
        :rtype:  tuple
        """
        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column, variations_column=variation_column)

        return (get_standard_error(df_summary['rate'][self.variations.variation_label],
                                   df_summary['total'][self.variations.variation_label]),
                get_standard_error(df_summary['rate'][self.variations.control_label],
                                   df_summary['total'][self.variations.control_label]))
