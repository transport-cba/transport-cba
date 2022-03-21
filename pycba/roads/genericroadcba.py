from ..genericcba import GenericCBA


class GenericRoadCBA(GenericCBA):
    """
    Base class for road projects

    TODO: move functionality from pycba.roads.svk.OPIIv3p0.RoadCBA here.
    """

    def __init__(self,
                 init_year,
                 evaluation_period,
                 price_level,
                 fin_discount_rate,
                 eco_discount_rate,
                 currency,
                 verbose=False
                 ):

        super().__init__(init_year=init_year,
                         evaluation_period=evaluation_period,
                         price_level=price_level,
                         fin_discount_rate=fin_discount_rate,
                         eco_discount_rate=eco_discount_rate,
                         currency=currency,
                         verbose=verbose)

    def read_parameters(self):
        raise NotImplementedError("Function read_parameters is supposed"
                                  "to be defined in the specific "
                                  "implementation")

    def prepare_parameters(self):
        raise NotImplementedError("Function prepare_parameters is supposed"
                                  "to be defined in the specific "
                                  "implementation")

    def read_project_inputs(self):
        raise NotImplementedError("Function read_project_inputs is supposed"
                                  "to be defined in the specific "
                                  "implementation")

    def perform_economic_analysis(self):
        raise NotImplementedError("Function perform_economic_analysis is "
                                  "supposed to be defined in the specific "
                                  "implementation")

    def print_economic_indicators(self):
        super().print_economic_indicators()

    def perform_financial_analysis(self):
        raise NotImplementedError("To be added.")

    def print_financial_indicators(self):
        raise NotImplementedError("To be added.")