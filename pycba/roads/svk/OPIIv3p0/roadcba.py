import pandas as pd
import numpy as np
from pycba.roads import GenericRoadCBA

DAYS_YEAR = 365.0

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
                 cpi_source="newest",
                 gdp_source="newest"):
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
        self.gdp_source = gdp_source

        # parameter container
        self.paramdir = self.dirn + '/parameters/svk/transport/OPIIv3p0/'
        self.paramfile = self.paramdir + \
                         "svk_road_cba_parameters_OPIIv3p0_2022.xlsx"
        self.params_raw = {}
        self.params_clean = {}

        # OPIIv3p0 specific part
        # OPIIv3p0 parameters behaviour
        self.PRICE_LEVEL_ADJUSTED = ["c_op", "vtts", 'voc_l', 'voc_t']
        # self.PRICE_LEVEL_ADJUSTED = ["c_op", "toll_op",
        #           "vtts", "voc", "c_fuel", "c_acc", "c_ghg", "c_em", "noise"]

        # OPIIv3p0 dataframes for road CBA
        self.RP = None      # road parameters
        self.L = None
        self.L0 = None
        self.L1 = None
        self.RF = None

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

        # GDP growth
        # read meta information on GDP values
        df_gdp_meta = pd.read_excel(self.paramfile,
                                    sheet_name="gdp_growth_meta",
                                    index_col="key")

        if self.gdp_source not in df_gdp_meta.index:
            raise ValueError("{0!s} not available as an option for GDP."
                             "Use on of {1!s} instead".
                             format(self.gdp_source, list(df_gdp_meta.index)))
        gdp_col = df_gdp_meta.loc[self.gdp_source, 'column']

        # read GDP information from the correct source
        self.gdp_growth = pd.read_excel(self.paramfile,
                                        sheet_name="gdp_growth",
                                        index_col=0)
        self.gdp_growth = \
            self.gdp_growth[[gdp_col]].rename(columns={gdp_col: "gdp_growth"})

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
        self.params_raw['occ_p'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="passenger_occupancy")
        self.params_raw['r_tp'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="trip_purpose")
        self.params_raw['vtts'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="vtts")
        self.params_raw['voc_t'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="voc_t")
        self.params_raw['voc_l'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="voc_l")
        self.params_raw['r_fuel'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="fuel_ratio")
    def _clean_parameters(self):
        """
        Incorporate scale into values.
        Remove unimportant columns. Populate the params_clean dictionary
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

        self.params_clean['c_op']['lower_usage'] =\
            self.params_clean['c_op']['lower_usage'].astype(bool)
        self.params_clean['c_op'].set_index(['section_type', 'surface',
                                             'condition', 'lower_usage'])

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
                self.cpi.loc[yi-1, "cpi_index"]  \
                / (self.cpi.loc[yi-1, "cpi"] + 1.0)

    def _adjust_price_level(self):
        """Unify the prices to a single price level."""
        if self.verbose:
            print("Adjusting price level...")

        for c in self.PRICE_LEVEL_ADJUSTED:
            if self.verbose:
                print("    Adjusting: %s" % c)
            self.params_clean[c]["value"] = \
                self.params_clean[c].value * self.params_clean[c].price_level.\
                    map(lambda x: self.cpi.loc[x].cpi_index)

            if "gdp_growth_adjustment" in self.params_clean[c].columns:
                # rename to communicate new meaning of the value
                self.params_clean[c].rename(
                    columns={"price_level":"base_year"}, inplace=True)
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

        # opex
        self.params_clean['c_op'] = self.params_clean['c_op'].set_index(
            ["section_type", "surface", "condition", "lower_usage"]
        )

        self._wrangle_vtts()
        self._wrangle_fuel()
        self._wrangle_voc()

    def _wrangle_vtts(self):
        """
        Average the value of travel time savings over everything other
        than vehicle type.

        Result:
            self.params_clean['vtts'] contains unit values of vtts for
            vehicle types. Unit is eur/h/veh.
        """

        vtts = self.params_clean['r_tp'].set_index('vehicle')
        # ignore train and mass transit for RoadCBA
        vtts = vtts.drop(index=['transit', 'train'])
        vtts = pd.DataFrame(vtts.stack(0), columns=['value'])
        vtts.index.names = ['vehicle', 'purpose']

        # compute passenger purpose per vehicle
        occ_p = self.params_clean['occ_p'].set_index('vehicle')[['value']]
        vtts = vtts * occ_p
        vtts_unit = self.params_clean['vtts']

        # compute value of travel time saved per vehicle per purpose
        vtts = vtts_unit.set_index('purpose')['value'].to_frame() * vtts

        # append info on price level and gdp adjustment elasticity
        vtts = pd.merge(
            vtts.reset_index(),
            vtts_unit[['purpose', 'base_year', 'gdp_growth_adjustment']],
            on='purpose',
            how='left')

        vtts = vtts.set_index(['vehicle', 'purpose'])
        self.params_clean['vtts'] = vtts.copy()

    def _wrangle_voc(self):
        # distance-dependent
        voc_l = self.params_clean['voc_l']
        voc_l = voc_l.set_index(['vehicle', 'fuel'])
        self.params_clean['voc_l'] = voc_l.copy()

        # time-dependent
        voc_t = self.params_clean['voc_t']
        voc_t = voc_t.set_index(['vehicle', 'fuel'])
        self.params_clean['voc_t'] = voc_t.copy()

    def _wrangle_fuel(self):
        """

        Result:
            self.params_clean['r_fuel'] contains values indexed by vehicle
            and fuel type.
        """

        # set index on fuel ratio
        c = "r_fuel"
        self.params_clean[c].reset_index(inplace=True, drop=True)
        self.params_clean[c].set_index(["vehicle", "fuel"], inplace=True)


    def prepare_parameters(self):
        self._clean_parameters()
        self._wrangle_cpi()
        self._adjust_price_level()
        self._wrangle_parameters()

    def read_project_inputs(self, df_rp, df_capex,
                            df_int_0, df_int_1, df_vel_0, df_vel_1):
        """
            Input
            ------
            df_rp: pandas DataFrame
                Dataframe of road parameters.

            df_capex: pandas DataFrame
                Dataframe of investment costs.

            df_int_0: pandas DataFrame
                Dataframe of vehicle intensities (AADT) in variant 0.

            df_vel_0: pandas DataFrame
                Dataframe of vehicle velocities in variant 0.

             df_int_1: pandas DataFrame
                Dataframe of vehicle intensities (AADT) in variant 1.

            df_vel_1: pandas DataFrame
                Dataframe of vehicle velocities in variant 1.

            Returns
            ------

        """
        if self.verbose:
            print("Reading project inputs from dataframes...")

        self.RP = df_rp.copy()
        self.C_fin = df_capex.copy()

        self.I0 = df_int_0.copy()
        self.I1 = df_int_1.copy()
        self.V0 = df_vel_0.copy()
        self.V1 = df_vel_1.copy()

        # assign core variables
        #self._assign_core_variables()
        self._wrangle_inputs()

    def _wrangle_inputs(self):
        """
        Modify input matrices of intensities and velocities
        in line with the economic period and other global requirements.
        Ensure that the columns representing years are integers.
        """

        self.I0 = self._wrangle_intensity_velocity(self.I0, 'I0')
        self.I1 = self._wrangle_intensity_velocity(self.I1, 'I1')
        self.V0 = self._wrangle_intensity_velocity(self.V0, 'V0')
        self.V1 = self._wrangle_intensity_velocity(self.V1, 'V1')

        self._wrangle_capex()

    def _wrangle_intensity_velocity(self, df, name):
        """
        df: pandas DataFrame
            Dataframe of vehicle intensities or velocities.

        name: string
            Name of the dataframe to be used in a warning.
        """

        df_out = df.reset_index()\
                    .drop(columns='id_model_section')\
                    .set_index(['id_road_section', 'vehicle'])
        df_out.columns = df_out.columns.astype(int)

        if df_out.columns[-1] < self.yr_f:
            if self.verbose:
                print("Warning: "
                      + name
                      + " not forecast until the end of period,"
                        "filling with zeros.")
        df_out = df_out[self.yrs].fillna(0)

        return df_out

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
        cf = pd.merge(self.C_fin,
                      self.params_clean['capex_conv_fac'],
                      how='left',
                      left_index=True,
                      right_index=True)[['value']].fillna(
            self.params_clean['conv_fac'].loc['aggregate', 'value'])['value']

        self.C_eco = self.C_fin.multiply(cf, axis='index')

    def compute_opex(self):
        """Create a dataframe of operation costs (OPEX)."""
        if self.verbose:
            print("Computing OPEX...")

        assert bool(self.UC) == True, "Unit costs not computed."

        # maintenance of road sections
        UC = self.UC["c_op"].copy()

        # road parameters

        # road areas - variant 0

        for variant in [0, 1]:
            RA = self.RP.loc[
                    self.RP['variant']==variant,
                    ['section_type', 'surface', 'condition', 'lower_usage',
                     'length', 'width', 'area']
                ].copy()

            # fill values which were not given as a parameter
            unfilled = RA['area'].isna()
            RA.loc[unfilled, "area"] = RA.loc[unfilled, "width"]\
                                        * RA.loc[unfilled, "length"]\
                                        * 1e3
            RA = RA.groupby(
                    ['section_type', 'surface', 'condition', 'lower_usage']
                )[['area']].sum()

            # time matrix of road areas
            RA = pd.DataFrame(np.outer(RA['area'], np.ones_like(self.yrs)),
                               index=RA.index,
                               columns=self.yrs)
            # save to dataframe of financial operating costs
            if variant == 0:
                self.O0_fin = (RA * UC).dropna()
            else:
                self.O1_fin = (RA * UC).dropna()

        # toll system operating costs
        # NOT IMPLEMENTED

        # apply the aggregate conversion factor
        cf = self.params_clean['conv_fac'].loc['aggregate', 'value']

        self.O0_eco = self.O0_fin * cf
        self.O1_eco = self.O1_fin * cf
        self.NC["opex"] = self.O1_eco.sum() - self.O0_eco.sum()


    # compute areas

    # =====
    # Preparation functions
    # =====
    def _adjust_for_gdp(self, years, df_gdp_growth, df_values):
        pass

    def _create_unit_cost_matrix(self):
        """Define the unit cost time-series matrices for each benefit"""

        if self.verbose:
            print("Creating time series for benefits' unit costs...")

        for b in ["c_op", "vtts", "voc_l", 'voc_t']:
            if self.verbose:
                print("    Creating: %s" % b)
            # set up empty dataframe
            self.UC[b] = pd.DataFrame(columns=self.yrs,
                                      index=self.params_clean[b].index)

            # Adjust for gdp growth until the start of the evaluation period.
            if "base_year" in self.params_clean[b].columns:
                base_year_col = "base_year"
            else:
                base_year_col = "price_level"

            b_base_yr = self.params_clean[b][base_year_col].iloc[0]
            assert np.all(self.params_clean[b][base_year_col] == b_base_yr )
            # We assume that all unit costs for a single benefit
            # are given for the same base year.

            helper_yrs = np.arange(b_base_yr, self.yr_i+1)
            UC_helper = pd.DataFrame(columns=helper_yrs,
                                     index=self.params_clean[b].index)
            UC_helper[b_base_yr] = self.params_clean[b]['value']

            for yr in helper_yrs[1:]:
                # set to the value of the preceding year
                UC_helper[yr] = \
                    UC_helper[yr-1]
                # adjust with gdp growth if required
                if "gdp_growth_adjustment" in self.params_clean[b].columns:
                    UC_helper[yr] = UC_helper[yr] \
                                    * (1.0
                                       + self.gdp_growth.loc[yr-1, "gdp_growth"]
                                        * self.params_clean[b]
                                       ["gdp_growth_adjustment"])

            self.UC[b][self.yr_i] = UC_helper[self.yr_i]
            for yr in self.yrs[1:]:
                # set to the value of the preceding year
                self.UC[b][yr] = \
                    self.UC[b][yr-1]# * self.cpi.loc[yr, "cpi_index"]
                # adjust with gdp growth if required
                if "gdp_growth_adjustment" in self.params_clean[b].columns:
                    self.UC[b][yr] = self.UC[b][yr] \
                    * (1.0 + self.gdp_growth.loc[yr-1].gdp_growth \
                    * self.params_clean[b].gdp_growth_adjustment)

                # name the columns as 'year'
                # helps later with stacking/unstacking
                self.UC[b].columns.name = 'year'


    def _create_length_matrix(self):
        """Create the matrix of lengths with years as columns"""
        if self.verbose:
            print("Creating length matrix...")

        # prepare index of road parameters: distinguish by length and variant
        self.RP.reset_index(inplace=True)
        self.RP.set_index(['id_road_section', 'variant'], inplace=True)

        self.L = pd.DataFrame(\
            np.outer(self.RP.length, np.ones_like(self.yrs)), \
            columns=self.yrs, index=self.RP.index)

        self.L0 = self.L.loc[(slice(None), 0), :]
        self.L0 = self.L0.reset_index().drop(columns='variant')\
                      .set_index('id_road_section')

        self.L1 = self.L.loc[(slice(None), 1), :]
        self.L1 = self.L1.reset_index().drop(columns='variant') \
                      .set_index('id_road_section')

    def _compute_travel_time_matrix(self):
        """Compute travel time by road section and vehicle type"""
        if self.verbose:
            print("Creating matrices of travel times...")
        assert self.L is not None, "Compute length matrix first."

        self.T0 = self.L0 / self.V0
        self.T0 = self.T0.replace([np.inf, -np.inf], 0.0).dropna()
        self.T0 = self.T0.reset_index()\
                         .set_index(['id_road_section', 'vehicle'])

        self.T1 = self.L1 / self.V1
        self.T1 = self.T1.replace([np.inf, -np.inf], 0.0).dropna()
        self.T1 = self.T1.reset_index()\
                         .set_index(['id_road_section', 'vehicle'])

    def _create_fuel_ratio_matrix(self):
        if self.verbose:
            print("Creating matrix of fuel ratios by vehicle...")
        r_fuel = self.params_clean['r_fuel']
        self.RF = pd.DataFrame(np.outer(r_fuel['value'],
                                        np.ones_like(self.yrs)),
                               columns=self.yrs,
                               index=r_fuel.index)

    def _compute_vtts(self):
        """Mask is given by the intensities, as these are zero
        in the construction years"""

        if self.verbose:
            print("    Computing VTTS...")
        assert self.T0 is not None, "Compute travel time first."
        assert self.T1 is not None, "Compute travel time first."

        b = "vtts"
        self.B0[b] = self.UC[b] * self.T0 * self.I0 * DAYS_YEAR
        self.B1[b] = self.UC[b] * self.T1 * self.I1 * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()

    def _compute_voc(self):
        assert self.L0 is not None, "Compute length matrix first."
        assert self.L1 is not None, "Compute length matrix first."
        assert self.T0 is not None, "Compute travel time first."
        assert self.T1 is not None, "Compute travel time first."
        assert self.RF is not None, "Compute fleet composition first"

        # length-dependent part
        b = 'voc_l'
        self.B0[b] = self.UC[b] * self.RF * (self.L0 * self.I0) * DAYS_YEAR
        self.B1[b] = self.UC[b] * self.RF * (self.L1 * self.I1) * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()

        # time-dependent part
        b = 'voc_t'
        self.B0[b] = self.UC[b] * self.RF * (self.T0 * self.I0) * DAYS_YEAR
        self.B1[b] = self.UC[b] * self.RF * (self.T1 * self.I1) * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()

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
