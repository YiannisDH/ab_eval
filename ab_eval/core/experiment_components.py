import json
import logging

logger = logging.getLogger(__name__)


class evaluation_metrics(object):
    """
    Class tha defines the KPIs that we need to use for evaluation
    :param kpis: the list of kpis
    :type kpis: list of kpis
    :param primary_KPI: the primary KPI that will be used in every evaluation
    :type primary_KPI: string
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



class variations(object):
    """
    Class that defines the variations characteristics inside the dataset
    :param column_name: the column name that contains the variation information inside the dataframe
    :type column_name: string
    :param control_label: the name of the control group that can be found inside the variations column
    :type control_label: string
    :param variation_label: the name of the variation group that can be found inside the variations column
    :type control_label: string
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


