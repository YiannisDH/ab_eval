import random
import string
import logging
import scipy.stats as scs
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

# import numpy as np


def generate_random_cvr_data(sample_size, p_control, p_variation, days=None, control_label='A',
                             variation_label='B'):
    """This function generates fake dataset for an ab test
    :param:   sample_size: sample size of the experament
    :type:    s_size_control: int
    :param    p_control: The conversion rate of the control group (probability to convert)
    :type     p_control: float
    :param    p_variation: The conversion rate of the variation   (probability to convert)
    :type     p_variation: float
    :param:   (optional) days: if provided, a column for 'ts' will be included to divide the data in chunks of time
            Note: overflow data will be included in an extra day
    :type:  days: integer
    :param:   (optional) control_label: The label of the control group
    :type:    control_label: str
    :param:   (optional) variation_label: The label of the variation
    :type:    variation_label: str
    :returns: df :dataframe with the generated test data
    :rtype: dataframe
    """
    logging.info('Fake test is generating with sample_size={}.'.format(sample_size))
    data = []
    # total amount of rows in the data
    group_bern = scs.bernoulli(0.5)  # used to generate equal sample sizes
    # initiate bernoulli distributions to randomly sample from
    A_bern = scs.bernoulli(p_control)
    B_bern = scs.bernoulli(p_variation)
    d1 = datetime.strptime('2018-01-01', '%Y-%m-%d')
    for idx in range(sample_size):
        row = {}
        row['segment'] = random.choice(['new', 'returning'])
        row['fullvisitorid'] = ''.join(random.choices(string.digits, k=20))
        row['visitid'] = random.randint(1400000000, 1500000000)  # randint is inclusive at both ends
        if days is not None:
            if type(days) == int:
                row['date'] = (d1 + timedelta(days=idx // (sample_size // days))).strftime('%Y-%m-%d')

            else:
                raise ValueError("Expecting integer but got {}.".format(type(days)))
        # assign group based on 50/50 probability and assign values
        group_type = lambda x: control_label if x == 0 else variation_label
        row['group'] = group_type(group_bern.rvs())
        if row['group'] == 0:
            # assign conversion based on provided parameters
            row['CVR'] = A_bern.rvs()
            row['mCVR1'] = A_bern.rvs()
        else:
            row['CVR'] = B_bern.rvs()
            row['mCVR1'] = A_bern.rvs()

        data.append(row)

    # convert data into pandas dataframe
    df = pd.DataFrame(data)
    return df


def get_segments_sample_size(df, segment=None, segment_column='segment'):
    """
    This function returns the sample size (int) of a specific segment
    :param  df: the dataframe with the test data
    :type   dataframe
    :param  segment: (optional) the name of the segment to calculate the sample size
    :type   segment: string
    :param  segment_column: (optional) the column name that contains the segment information
    :type   segment_column: string
    :return sample_size: the sample size of the specific segment
    """
    if segment:
        return df.loc[df[segment_column] == segment].shape[0]
    return df.shape[0]


def get_test_summary(df, kpi, segment=None, segment_column='segment', variations_column='group'):
    """
    :param   df: the dataframe with the test data
    :type    dataframe
    :param   kpi: column name that contains the KPI
    :type    kpi: string
    :param   segment: (optional) the name of the segment to calculate the sample size
    :type    segment: string
    :param   segment_column: (optional) the column name that contains the segment information
    :type    segment_column: string
    :param   variations_column: (optional) the column name that contains the variation information
    :type    variations_column: string
    :return: dataframe with test_sammary
    """

    if segment:
        df = df[df[segment_column] == segment]

    ab_summary = df.pivot_table(values=kpi, index=variations_column, aggfunc=np.sum)
    ab_summary['total'] = df.pivot_table(values=kpi, index=variations_column, aggfunc=lambda x: len(x))
    ab_summary['rate'] = df.pivot_table(values=kpi, index=variations_column)

    return ab_summary


def get_min_sample_size(baseline_cvr, expected_uplift, power=0.8, sig_level=0.05):
    """
    Return the minimum sample size that we need for a split test.
    :param  baseline_cvr: the probability of success for the control group (cvr of control group)]
    :type   baseline_cvr: float
    :param  expected_uplift: the expected uplift from the test (absolute value)
    :type   expected_uplift: float
    :param  power: (optional) probability of rejecting the null hypothesis when the
                    null hypothesis is false, typically 0.8
    :param  sig_level: (optional) the significance level, typically is 0.05 (or 95%)
    :type   sig_level: float
    :return: min
    """

    standard_norm = scs.norm(0, 1)
    # find Z_beta from desired power
    Z_beta = standard_norm.ppf(power)
    # find Z_alpha
    Z_alpha = standard_norm.ppf(1 - sig_level / 2)
    # average of probabilities from both groups
    pooled_prob = (baseline_cvr + baseline_cvr + expected_uplift) / 2
    min_sample_size = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2 / expected_uplift**2)

    return min_sample_size


def get_standard_error(conversion_probability, sample_size):
    """
    This method gets the conversion_probability and the sample_size and returns the standard error

    :param   conversion_probability: the conversion probability
    :type    conversion_probability: float
    :param   sample_size: the sample size
    :type    sample_size: integer
    :return: standard_error
    """
    return standard_deviation(conversion_probability) / np.sqrt(sample_size)


def z_val(sig_level=0.05, two_tailed=True):
    """
    Returns the z value for a given significance level

    :param   sig_level: the significance level
    :type    sig_level: float
    :param   two_tailed: true if the evaluation is 2 tailed
    :type    two_tailed: bool
    :return: z_val
    """
    z_dist = scs.norm()
    if two_tailed:
        sig_level = sig_level / 2
        area = 1 - sig_level
    else:
        area = 1 - sig_level

    return z_dist.ppf(area)


def standard_deviation(conversion_probability):
    """
    :param   conversion_probability: the conversion probability
    :type    conversion_probability: float
    :return: standard_deviation
    """
    return np.sqrt((conversion_probability * (1 - conversion_probability)))


def confidence_interval(sample_mean=0, sample_std=1, sample_size=1, significance_level=0.05):
    """
    Calculates and returns the confidence interval for given sample size and standard deviation
    :param   sample_mean: the mean of the sample
    :type    sample_mean: float
    :param   sample_std: the standard deviation of the sample
    :type    sample_std: float
    :param   sample_size: the sample size
    :type    sample_size: integer
    :param   significance_level: the significance level
    :type    significance_level: float
    :return: confidence_interval
    """

    return sample_mean - z_val(significance_level) * sample_std / np.sqrt(sample_size)
