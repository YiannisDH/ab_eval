import simplejson
import json
import logging
from ab_eval.core.experiment_components import variations, evaluation_metrics
from ab_eval.core.utils import get_test_summary, get_standard_error, get_z_val, get_standard_deviation
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
    :param   date_column: name of the column that hold the date
    :type    date_column: string
    """
    def __init__(
            self,
            data,
            kpis=evaluation_metrics(kpis=["CVR"]),
            variations=variations(),
            segments=None,
            alternative='two-sided',
            significance_level=0.05,
            date_column='date',
            *args, **kwargs):
        super(experiment, self).__init__(*args, **kwargs)
        self.data = experiment.transform_date_column(data, date_column)
        self.kpis = kpis
        self.variations = variations
        self.segments = segments
        self.alternative = alternative
        self.date_column = date_column
        if significance_level > 1:
            raise ValueError("significance_level should be >0 and <1 : {}")
        self.significance_level = significance_level

    @staticmethod
    def transform_date_column(df, date_column):
        df[date_column] = df[date_column].astype(str)
        return df

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

    def get_p_val(self, kpi='CVR', segment=None, segment_column='segment', date=None):
        """Method that calculates the p-value for a given dataset and KPI


        :param   kpi: the KPI that should be used
        :type    kpi: str
        :param   segment: the segment that should be used
        :type    segment: str
        :param   segment_column: the column name that contains the segment information
        :type    segment_column: str
        :param   variation_column: the column name that contains the variation information
        :type    variation_column
        :param   date: if date is given (format '%Y%m%d') then the check will happen up to that date
        :type    date: string ('%Y%m%d')
        :return: the p value
        :rtype:  dict

        """

        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        if date is None:
            df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())
        else:
            df_summary = get_test_summary(self.data[self.data[self.date_column] <= date], kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())

        zscore, pval = sm.stats.proportions_ztest([df_summary[kpi][self.variations.variation_label], df_summary[kpi][self.variations.control_label]],
                                                  [df_summary['total'][self.variations.variation_label],
                                                  df_summary['total'][self.variations.control_label]],
                                                  alternative=self.alternative)

        return {"z-score": zscore, 'p-value': pval}

    def get_relative_conversion_uplift(self, kpi='CVR', segment=None, segment_column='segment', date=None):
        """Method that calculates the relative conversion_uplift

        :param   kpi: the KPI that should be used
        :type    kpi: str
        :param   segment: the segment that should be used
        :type    segment: str
        :param   segment_column: the column name that contains the segment information
        :type    segment_column: str
        :param   variation_column: the column name that contains the variation information
        :type    variation_column
        :param   date: if date is given (format '%Y%m%d') then the check will happen up to that date
        :type    date: string ('%Y%m%d')
        :return: the relative conversion uplift
        :rtype:  float
        """
        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        if date is None:
            df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())
        else:
            df_summary = get_test_summary(self.data[self.data[self.date_column] <= date], kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())

        return (df_summary['rate'][self.variations.variation_label] - df_summary['rate'][self.variations.control_label]) / \
            df_summary['rate'][self.variations.control_label]

    def get_standard_errors_of_test(self, kpi='CVR', segment=None, segment_column='segment', date=None):
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
        :param   date: if date is given (format '%Y%m%d') then the check will happen up to that date
        :type    date: string ('%Y%m%d')
        :return: standard error for variation and control
        :rtype:  dict
        """
        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        if date is None:
            df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())
        else:
            df_summary = get_test_summary(self.data[self.data[self.date_column] <= date], kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())

        return {"control_standard_error": get_standard_error(df_summary['rate'][self.variations.variation_label],
                                                             df_summary['total'][self.variations.variation_label]),
                "variation_standard_error": get_standard_error(df_summary['rate'][self.variations.control_label],
                                                               df_summary['total'][self.variations.control_label])}

    def get_summary(self, kpi='CVR', segment=None, segment_column='segment', date=None):
        """Method that calculates the p-value for a given dataset and KPI


        :param   kpi: the KPI that should be used
        :type    kpi: str
        :param   segment: the segment that should be used
        :type    segment: str
        :param   segment_column: the column name that contains the segment information
        :type    segment_column: str
        :param   variation_column: the column name that contains the variation information
        :type    variation_column
        :param   date: if date is given (format '%Y%m%d') then the check will happen up to that date
        :type    date: string ('%Y%m%d')
        :return: the p value
        :rtype:  dict

        """

        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        if date is None:
            df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())
        else:
            df_summary = get_test_summary(self.data[self.data[self.date_column] <= date], kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())

        return {
            'variation':
                {
                    "label":
                        self.variations.variation_label,
                    "sessions":
                        float(df_summary['total'][self.variations.variation_label]),
                    'conversions':
                        float(df_summary[kpi][self.variations.variation_label])
                },
            'control':
                {
                    "label":
                        self.variations.control_label,
                    "sessions":
                        float(df_summary['total'][self.variations.control_label]),
                    'conversions':
                        float(df_summary[kpi][self.variations.control_label])
                }
        }

    def get_confidence_interval_of_test(self, kpi='CVR', segment=None, segment_column='segment', date=None):
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
        :param   date: if date is given (format '%Y%m%d') then the check will happen up to that date
        :type    date: string ('%Y%m%d')
        :return: confidence_interval of the test summary as a tuple
        :rtype:  json
        """

        if kpi not in self.get_expirement_kpis():
            raise ValueError("Please use a valid KPI. this can be one of the followings: {}"
                             .format_map(self.get_expirement_kpis()))

        if date is None:
            df_summary = get_test_summary(self.data, kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())
        else:
            df_summary = get_test_summary(self.data[self.data[self.date_column] <= date], kpi=kpi, segment=segment, segment_column=segment_column,
                                          variations_column=self.variations.get_column_name())

        M1 = df_summary['rate'][self.variations.variation_label]
        M2 = df_summary['rate'][self.variations.control_label]
        N1 = df_summary['total'][self.variations.variation_label]
        N2 = df_summary['total'][self.variations.control_label]
        z = get_z_val(sig_level=self.significance_level, two_tailed=True if self.alternative == 'two-sided' else False)
        std1 = get_standard_deviation(M1)
        std2 = get_standard_deviation(M2)
        std1 *= std1
        std2 *= std2
        Sm1_m2 = (np.sqrt(((N1 - 1) * std1 ^ 2 + (N2-1) * std2 ^ 2) / (N1 + N2 - 2) ) )
        SE1_2 = Sm1_m2(np.sqrt(1 / N1 + 1 / N2))
        logging.info("ppppp eeeee oooo ssss")

        return {"lower_limit": M1 - M2 - z * SE1_2, "upper_limit": M1 - M2 + z * SE1_2}

    def analyze(self, kpis=None, analyze_segments=False, date=None):
        """
        Method to analyze the experiment. It returns a json object with the results
        :param   kpis: The kpis that needs to evaluate if null it evaluates all
        :type    kpis: list
        :param   analyze_segments: True to analyze also each segment
        :type    analyze_segments: bool
        :return: results as json
        :rtype:  json
        """

        results = []
        for kpi in self.kpis.get_kpis() if kpis is None else kpis:

            kpi_eval = {
                'kpi': kpi,
                'segment': 'all',
                'summary':
                    {
                        "test": self.get_p_val(kpi=kpi, date=date),
                        "relative_conversion_uplift": self.get_relative_conversion_uplift(kpi=kpi, date=date),
                        "standard_errors": self.get_standard_errors_of_test(kpi=kpi, date=date),
                        "confidence_interval": self.get_confidence_interval_of_test(kpi=kpi, date=date),
                        "volumes": self.get_summary(kpi=kpi, date=date)
                    }
            }
            results.append(kpi_eval)
            for segment in self.segments if analyze_segments else []:
                kpi_eval = {
                    'kpi': kpi,
                    'segment': segment,
                    'summary':
                        {
                            "test": self.get_p_val(kpi=kpi, segment=segment, date=date),
                            "relative_conversion_uplift": self.get_relative_conversion_uplift(kpi=kpi, segment=segment, date=date),
                            "standard_errors": self.get_standard_errors_of_test(kpi=kpi, segment=segment, date=date),
                            "confidence_interval": self.get_confidence_interval_of_test(kpi=kpi, segment=segment, date=date),
                            "volumes": self.get_summary(kpi=kpi, segment=segment, date=date)
                        }
                }
                results.append(kpi_eval)
        return simplejson.dumps(results, ignore_nan=True)

    def analyze_historically(self, kpis=None, analyze_segments=False):
        """
        Method to analyze the experiment. It returns a json object with the results
        :param   kpis: The kpis that needs to evaluate if null it evaluates all
        :type    kpis: list
        :param   analyze_segments: True to analyze also each segment
        :type    analyze_segments: bool
        :return: results as json
        :rtype:  json
        """

        unique_dates = self.data[self.date_column].unique()

        results = []
        for kpi in self.kpis.get_kpis() if kpis is None else kpis:
            summary = {
                "test": self.get_p_val(kpi=kpi),
                "relative_conversion_uplift": self.get_relative_conversion_uplift(kpi=kpi),
                "standard_errors": self.get_standard_errors_of_test(kpi=kpi),
                "confidence_interval": self.get_confidence_interval_of_test(kpi=kpi),
                "volumes": self.get_summary(kpi=kpi)
            }
            history = []
            for date in unique_dates:
                daily_results = {
                    "date": date,
                    "test": self.get_p_val(kpi=kpi, date=date),
                    "relative_conversion_uplift": self.get_relative_conversion_uplift(kpi=kpi, date=date),
                    "standard_errors": self.get_standard_errors_of_test(kpi=kpi, date=date),
                    "confidence_interval": self.get_confidence_interval_of_test(kpi=kpi, date=date),
                    "volumes": self.get_summary(kpi=kpi, date=date)
                }
                history.append(daily_results)

            results.append({
                'kpi': kpi,
                'segment': 'all',
                'summary': summary,
                'history': history
            })

            for segment in self.segments if analyze_segments else []:
                summary = {
                    "test": self.get_p_val(kpi=kpi),
                    "relative_conversion_uplift": self.get_relative_conversion_uplift(kpi=kpi, segment=segment),
                    "standard_errors": self.get_standard_errors_of_test(kpi=kpi, segment=segment),
                    "confidence_interval": self.get_confidence_interval_of_test(kpi=kpi, segment=segment),
                    "volumes": self.get_summary(kpi=kpi, segment=segment)
                }
                history = []
                for date in unique_dates:
                    daily_results = {
                        "date": date,
                        "test": self.get_p_val(kpi=kpi, date=date, segment=segment),
                        "relative_conversion_uplift": self.get_relative_conversion_uplift(kpi=kpi, date=date, segment=segment),
                        "standard_errors": self.get_standard_errors_of_test(kpi=kpi, date=date, segment=segment),
                        "confidence_interval": self.get_confidence_interval_of_test(kpi=kpi, date=date, segment=segment),
                        "volumes": self.get_summary(kpi=kpi, date=date, segment=segment)
                    }
                    history.append(daily_results)

                results.append({
                    'kpi': kpi,
                    'segment': segment,
                    'summary': summary,
                    'history': history
                })
        return simplejson.dumps(results, ignore_nan=True)

    def is_valid(self):
        """
        Checks if the experiment is valid over time. A valid experiment over time should have for all days variation and control
        :return: true or false
        :rtype:  bool
        """

        if self.segments is None:

            if (set(self.data[self.variations.column_name]) == {self.variations.variation_label, self.variations.control_label})\
                    and (self.data[self.variations.column_name].nunique() * self.data[self.date_column].nunique() == len(self.data.index)):
                return True
        else:
            logging.info("Be aware that the validity check is not working if you have segments in your data. It always returns True.")
            return True
        return False
