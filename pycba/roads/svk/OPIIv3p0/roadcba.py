import pandas as pd
import numpy as np
from pycba.roads import GenericRoadCBA


class RoadCBA(GenericRoadCBA):
    """
    Implementation of economic evaluation using the methods and parameters
    defined by Slovak Guidelines for CBA appraisal of transport
    projects v3.0.

    example

    from pycba.roads.svk.OPIIv3p0 import RoadCBA
    """

    def __init__(self,
                 init_year,
                 evaluation_period,
                 price_level,
                 fin_discount_rate,
                 eco_discount_rate,
                 currency,
                 verbose=False,
                 include_freight_time=False,
                 cpi_source="newest"):
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

        consider_freight_time: bool, optional
            Flag to control if freight time savings
            are included in the appraisal.               .

        Returns
        ------
        object
            pycba.roads.svk.OPIIv3p0.RoadCBA object.
        """

        super().__init__(init_year=init_year,
                         evaluation_period=evaluation_period,
                         price_level=price_level,
                         fin_discount_rate=fin_discount_rate,
                         eco_discount_rate=eco_discount_rate,
                         currency=currency,
                         verbose=verbose)

        self.include_freight_time = include_freight_time
        self.cpi_source = cpi_source

        # parameter container
        self.paramdir = self.dirn + '/parameters/svk/transport/OPIIv3p0/'
        self.paramfile = self.paramdir + \
                         "svk_road_cba_parameters_OPIIv3p0_2022.xlsx"
        self.params_raw = {}
        self.params_clean = {}

        # OPIIv3p0 properties
        self.PRICE_LEVEL_ADJUSTED = ["c_op"]
        # self.PRICE_LEVEL_ADJUSTED = ["c_op", "toll_op",
        #           "vtts", "voc", "c_fuel", "c_acc", "c_ghg", "c_em", "noise"]

    def run_cba(self):
        self.read_parameters()
        self.prepare_parameters()

    def read_parameters(self):
        self._read_raw_parameters()

    def _read_raw_parameters(self):
        """Load all parameter dataframes"""
        if self.verbose:
            print("Reading CBA parameters...")

        # economic data

        # economic prognosis

        # inflation (CPI)
        # read meta information on CPI values
        df_cpi_meta = pd.read_excel(self.paramfile,
                                    sheet_name="cpi_meta",
                                    index_col="key")

        if self.cpi_source not in df_cpi_meta.index:
            raise ValueError("{0!s} not available as an option for CPI."
                             "Use on of {1!s} instead".format(self.cpi_source,
                                                    list(df_cpi_meta.index)))
        cpi_col = df_cpi_meta.loc[self.cpi_source, 'column']

        # read CPI information from the correct source
        self.cpi = pd.read_excel(self.paramfile,
                                 sheet_name="cpi",
                                 index_col=0)
        self.cpi = self.cpi[[cpi_col]].rename(columns={cpi_col:"cpi"})

        # RoadCBA specifics
        self.params_raw["res_val"] = \
            pd.read_excel(self.paramfile,
                          sheet_name="residual_value",
                          index_col=0)

        self.params_raw["conv_fac"] = \
            pd.read_excel(self.paramfile,
                          sheet_name="conversion_factors",
                          index_col=0)

        self.params_raw["c_op"] = \
            pd.read_excel(self.paramfile,
                          sheet_name="operation_cost")

    def _clean_parameters(self):
        """
        Incorporate scale into values.
        Remove unimportant columns. Populate the df_clean dictionary
        """
        if self.verbose:
            print("Cleaning parameters...")
        for itm in self.params_raw.keys():
            if self.verbose:
                print("    Cleaning: %s" % itm)
            self.params_clean[itm] = self.params_raw[itm].copy()
            if "nb" in self.params_clean[itm].columns:
                self.params_clean[itm].drop(columns=["nb"], inplace=True)
            if "unit" in self.params_clean[itm].columns:
                self.params_clean[itm].drop(columns=["unit"], inplace=True)

    def _wrangle_cpi(self, infl=0.02, yr_min=2000, yr_max=2100):
        """
        Fill in missing values and compute cumulative inflation
        to be able to adjust the price level
        """

        if self.verbose:
            print("Wrangling CPI...")

        self.cpi = self.cpi.reindex(np.arange(yr_min, yr_max + 1))
        self.cpi["cpi"].fillna(infl, inplace=True)

        # compute cumulative CPI
        self.cpi["cpi_index"] = np.nan
        self.cpi.loc[self.yr_pl, "cpi_index"] = 1.0
        ix = self.cpi.index.get_loc(self.yr_pl)

        # backward
        for i in range(ix - 1, -1, -1):
            yi = self.cpi.index[i]
            self.cpi.loc[yi, "cpi_index"] = \
                self.cpi.loc[yi+1, "cpi_index"] \
                * (self.cpi.loc[yi, "cpi"] + 1.0)

        # forward
        for i in range(ix + 1, len(self.cpi)):
            yi = self.cpi.index[i]
            self.cpi.loc[yi, "cpi_index"] = \
                self.cpi.loc[yi-1, "cpi_index"] / (
                            self.cpi.loc[yi-1, "cpi"] + 1.0)

    def _adjust_price_level(self):
        """Unify the prices for one price level"""
        if self.verbose:
            print("Adjusting price level...")

        for c in self.PRICE_LEVEL_ADJUSTED:
            if self.verbose:
                print("    Adjusting: %s" % c)
            self.params_clean[c]["value"] = self.params_clean[c].value \
                                        * self.params_clean[c].price_level \
                                            .map(
                lambda x: self.cpi.loc[x].cpi_index)
            self.params_clean[c].drop(columns=["price_level"], inplace=True)
            # self.params_clean[c]["value"] = self.params_clean[c].value

    def _wrangle_parameters(self):
        """Wrangle parameters into form suitable for computation."""
        # TODO replace with calls to functions

        # conversion factors
        capex_conv_fac = pd.merge(
            self.params_clean['res_val'][['default_conversion_factor']],
            self.params_clean['conv_fac'], how='left',
            left_on='default_conversion_factor', right_index=True)[['value']]
        self.params_clean['capex_conv_fac'] = capex_conv_fac

    def prepare_parameters(self):
        self._clean_parameters()
        self._wrangle_cpi()
        self._adjust_price_level()
        self._wrangle_parameters()

    def read_project_inputs(self, df_capex):
        """
            Input
            ------
            df_capex: pandas DataFrame
                Dataframe of investment costs.

            Returns
            ------

        """
        if self.verbose:
            print("Reading project inputs from dataframes...")

        self.C_fin = df_capex.copy()

        # assign core variables
        #self._assign_core_variables()
        self._wrangle_inputs()

    def _wrangle_inputs(self):
        self._wrangle_capex()

    def _wrangle_capex(self):
        """Collect capex."""

        # TODO check integrity

        # replace empty cells with zeros
        self.C_fin.fillna(0, inplace=True)

        # Capital investments before the start of evaluation periods
        # are summed and assigned to the first year of the evaluation period
        # See 3.3.1 of the Guidebook
        capex_yrs = self.C_fin.columns
        if len(capex_yrs[capex_yrs < self.yr_i]) != 0:
            if self.verbose:
                print("Squeezing CAPEX into the given economic period...")
            yrs_bef = capex_yrs[capex_yrs < self.yr_i]
            yrs_aft = capex_yrs[capex_yrs >= self.yr_i]
            self.C_fin[self.yr_i] += self.C_fin[yrs_bef].sum(1)
            self.C_fin = self.C_fin[yrs_aft]

    def compute_capex(self):
        """
        Apply conversion factors to compute CAPEX for the economic analysis.
        Create matrix of CAPEX for the whole evaluation period.
        """

        if self.verbose:
            print("Computing CAPEX...")

        # add columns for all years of evaluation period
        self.C_fin = pd.DataFrame(self.C_fin, columns=self.yrs).fillna(0)

        # assign conversion factors to items of capex dataframe
        self.cf = pd.merge(self.C_fin,
                           self.params_clean['capex_conv_fac'],
                           how='left',
                           left_index=True, right_index=True)[['value']]\
                           .fillna(self.params_clean['conv_fac'].
                                   loc['aggregate', 'value'])['value']

        self.C_eco = self.C_fin.multiply(self.cf, axis='index')


    def perform_economic_analysis(self):
        raise NotImplementedError("Function perform_economic_analysis is "
                                  "supposed to be defined in the specific "
                                  "implementation")

    def print_economic_indicators(self):
        super().print_economic_indicators()

    def perform_financial_analysis(self):
        raise NotImplementedError("To be added.")

    def print_financial_indicators(self):
        super().read_parameters()