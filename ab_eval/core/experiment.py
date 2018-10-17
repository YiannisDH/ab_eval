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
