import json
import logging
from ab_eval.core.experiment_components import variations, evaluation_metrics
from ab_eval.core.utils import get_test_summary, get_standard_error, get_z_val
import statsmodels.api as sm
import numpy as np

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
            alternative='two-sided',
            significance_level=0.05,
            *args, **kwargs):
        super(experiment, self).__init__(*args, **kwargs)
        self.data = data
        self.kpis = kpis
        self.variations = variations
        self.segments = segments
        self.alternative = alternative
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
        :type    kpi: str
        :param   segment: the segment that should be used
        :type    segment: str
        :param   segment_column: the column name that contains the segment information
        :type    segment_column: str
        :param   variation_column: the column name that contains the variation information
        :type    variation_column
        :return: the p value
        :rtype:  dict

        """

        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column, variations_column=variation_column)

        zscore, pval = sm.stats.proportions_ztest([df_summary[kpi][self.variations.variation_label], df_summary[kpi][self.variations.control_label]],
                                                  [df_summary['total'][self.variations.variation_label],
                                                  df_summary['total'][self.variations.control_label]],
                                                  alternative=self.alternative)

        return {"z-score": zscore, 'p-value': pval}

    def get_relative_conversion_uplift(self, kpi='CVR', segment=None, segment_column='segment', variation_column='group'):
        """Method that calculates the relative conversion_uplift

        :param   kpi: the KPI that should be used
        :type    kpi: str
        :param   segment: the segment that should be used
        :type    segment: str
        :param   segment_column: the column name that contains the segment information
        :type    segment_column: str
        :param   variation_column: the column name that contains the variation information
        :type    variation_column
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
        This method is calculating the standard error for variation and control and returns a dict where the first
        element as the standard error of control and the second as the standard error of variation

        :param   kpi: the KPI that should be used
        :type    kpi: str
        :param   segment: the segment that should be used
        :type    segment: str
        :param   segment_column: the column name that contains the segment information
        :type    segment_column: str
        :param   variation_column: the column name that contains the variation information
        :type    variation_column
        :return: standard error for variation and control
        :rtype:  dict
        """
        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column, variations_column=variation_column)

        return {"control_standard_error": get_standard_error(df_summary['rate'][self.variations.variation_label],
                                                             df_summary['total'][self.variations.variation_label]),
                "variation_standard_error": get_standard_error(df_summary['rate'][self.variations.control_label],
                                                               df_summary['total'][self.variations.control_label])}

    def get_confidence_interval_of_test(self, kpi='CVR', segment=None, segment_column='segment', variation_column='group'):
        """
        This method returns the confidence_interval of test as dict. http://onlinestatbook.com/2/estimation/difference_means.html
        :param   kpi: the KPI that should be used
        :type    kpi: str
        :param   segment: the segment that should be used
        :type    segment: str
        :param   segment_column: the column name that contains the segment information
        :type    segment_column: str
        :param   variation_column: the column name that contains the variation information
        :type    variation_column
        :return: confidence_interval of the test summary as a tuple
        :rtype:  json
        """

        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column, variations_column=variation_column)

        M1 = df_summary['rate'][self.variations.variation_label]
        M2 = df_summary['rate'][self.variations.control_label]
        z = get_z_val(sig_level=self.significance_level, two_tailed=True if self.alternative == 'two-sided' else False)
        errors = self.get_standard_errors_of_test(kpi=kpi, segment=segment, segment_column=segment_column, variation_column=variation_column)
        std1 = errors.get('control_standard_error')
        std2 = errors.get('variation_standard_error')
        std1 *= std1
        std2 *= std2
        Sm1_m2 = (np.sqrt((std1 + std2 / (2 / (1 / M1 + 1 / M2)))))

        return {"lower_limit": M1 - M2 - z * Sm1_m2, "upper_limit": M1 - M2 + z * Sm1_m2}

    def analyze(self, kpis=None, analyze_segments=False):
        """
        Method to analyze the experiment. It returns a json object with the results
        :param   kpis: The kpis that needs to evaluate if null it evaluates all
        :type    kpis: list
        :param   analyze_segments: True to analyze also each segment
        :type    analyze_segments: bool
        :return: results as json
        :rtype:  json
        """

        result_dict = {}

        for kpi in self.kpis.get_kpis() if kpis is None else kpis:
            results = []
            kpi_eval = {
                'all':
                    {
                        'results':
                            {
                                "test": self.get_p_val(kpi=kpi),
                                "relative_conversion_uplift": self.get_relative_conversion_uplift(kpi=kpi),
                                "standard_errors": self.get_standard_errors_of_test(kpi=kpi),
                                "confidence_interval": self.get_confidence_interval_of_test(kpi=kpi)
                            }
                    }
            }
            results.append(kpi_eval)
            for segment in self.segments if analyze_segments else []:
                kpi_eval = {
                    segment:
                        {
                            'results':
                                {
                                    "test": self.get_p_val(kpi=kpi, segment=segment),
                                    "relative_conversion_uplift": self.get_relative_conversion_uplift(kpi=kpi, segment=segment),
                                    "standard_errors": self.get_standard_errors_of_test(kpi=kpi, segment=segment),
                                    "confidence_interval": self.get_confidence_interval_of_test(kpi=kpi, segment=segment)
                                }
                        }
                }
                results.append(kpi_eval)
            result_dict[kpi] = results

        return json.dumps(result_dict)
