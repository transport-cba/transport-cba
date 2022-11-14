import pandas as pd
import numpy as np
import os

class GenericCBA(object):
    """Main module class.
    Packages all core concepts and functionality of a cost-benefit analysis.

    Usage

    GenericCBA.read_parameters() # reads parameters in the form given in a
                                 # guidebook

    # (...)
    # potential manipulation of parameters - testing, sensitivity analysis
    # (...)

    GenericCBA.prepare_parameters() # manipulates parameters internally
                                    # to a form suitable for computation
    GenericCBA.read_project_inputs()    # reads project inputs

    # (...)
    # potential manipulation of inputs, although setting up separate input
    # files is preferred
    # (...)

    GenericCBA.perform_economic_analysis()
    GenericCBA.print_economic_indicators()

    GenericCBA.perform_financial_analysis()
    GenericCBA.print_financial_indicators()
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
        """
        Input
        ------
        init_year: int
            The first year of economic analysis

        evaluation_period: int
            Duration of evaluation period in years.

        price_level: int
            The year in relation to which prices are set.

        fin_discount_rate: float
            Discount rate for financial analysis.

        eco_discount_rate: float
            Discount rate for economic analysis.

        currency: str
            Currency code.

        verbose: bool, optional
            Flag to control verbosity.

        Returns
        ------
        object
            GenericCBA object.
        """

        self.yr_i = init_year
        self.N_yr = evaluation_period
        self.yr_f = self.yr_i + self.N_yr - 1
        self.yrs = [i for i in range(self.yr_i, self.yr_i + self.N_yr)]

        self.yr_op = None
        self.N_yr_bld = None
        self.N_yr_op = None
        self.yrs_op = None

        self.yr_pl = price_level

        self.r_fin = fin_discount_rate
        self.r_eco = eco_discount_rate

        self.currency = currency

        self.verbose = verbose

        self.C_fin = None       # CAPEX financial
        self.C_eco = None       # CAPEX economic
        self.O0_fin = {}      # OPEX financial in 0th variant
        self.O0_eco = {}      # OPEX economic in 0th variant
        self.O1_fin = {}      # OPEX financial in 1st variant
        self.O1_eco = {}      # OPEX economic in 1st variant
        self.I0_fin = {}        # income financial in 0th variant
        self.I1_fin = {}        # income financial in 1st variant

        self.UC = {}            # unit costs
        self.B0 = {}            # benefits in 0th variant
        self.B1 = {}            # benefits in 1st variant
        self.NB = {}            # net benefits
        self.NC = {}            # net costs
        self.NI = {}            # net income

        # economic indicators
        self.ENPV = None
        self.ERR = None
        self.EBCR = None

        # pycba directory
        self.dirn = os.path.dirname(__file__)

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
        print("ENPV: %.2f M %s" % (self.ENPV / 1e6, self.currency.upper()))
        print("ERR : %.2f %%" % (self.ERR * 100))
        print("BCR : %.2f" % self.EBCR)

    def perform_financial_analysis(self):
        raise NotImplementedError("To be added.")

    def print_financial_indicators(self):
        raise NotImplementedError("To be added.")
