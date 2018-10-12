import random
import string
import logging
import scipy.stats as scs
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# import numpy as np


def generate_random_cvr_data(sample_size, p_control, p_variation, days=None, control_label='A',
                             variation_label='B'):
    """This function generates fake dataset for an ab test
    :param: sample_size: sample size of the experament
    :type: s_size_control: int
    :param p_control: The conversion rate of the control group (probability to convert)
    :type p_control: float
    :param p_variation: The conversion rate of the variation   (probability to convert)
    :type p_variation: float
    :param: (optional) days: if provided, a column for 'ts' will be included to divide the data in chunks of time
            Note: overflow data will be included in an extra day
    :type: days: integer
    :param: (optional) control_label: The label of the control group
    :type: control_label: str
    :param: (optional) variation_label: The label of the variation
    :type: variation_label: str
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
        # assign group based on 50/50 probability and asign values
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