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
        self.ACCELERATION_COLUMNS = ['exit_intravilan','roundabout_intravilan',
                    'roundabout_extravilan', 'intersection_intravilan',
                    'intersection_extravilan', 'interchange']
        self.PRICE_LEVEL_ADJUSTED = ["c_op", "vtts", 'voc_l', 'voc_t', 'c_ghg']
        # self.PRICE_LEVEL_ADJUSTED = ["c_op", "toll_op",
        #           "vtts", "voc", "c_fuel", "c_acc", "c_ghg", "c_em", "noise"]

        # OPIIv3p0 dataframes for road CBA
        self.RP = None      # road parameters

        self.L0 = None
        self.L1 = None
        self.V0 = None
        self.V1 = None
        self.T0 = None
        self.T1 = None
        self.I0 = None
        self.I1 = None

        self.RF = None  # ratio of fuel types
        self.QF0 = None  # quantity of fuel burnt on a section in variant 0
        self.QF1 = None  # quantity of fuel burnt on a section in variant 1

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
        self.params_raw['fuel_coeffs'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="fuel_consumption")
        self.params_raw['fuel_acc'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="fuel_consumption_acceleration")
        self.params_raw['fuel_rho'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="fuel_density")
        self.params_raw['c_fuel'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="fuel_cost")
        self.params_raw['r_ghg'] = \
            pd.read_excel(self.paramfile,
                      sheet_name="greenhouse_rate")
        self.params_raw['c_ghg'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="co2_cost")
        self.params_raw['r_em'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="emission_rate")
        self.params_raw['c_em'] = \
            pd.read_excel(self.paramfile,
                          sheet_name="emission_cost")
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
            if "scale" in self.params_clean[itm].columns:
                if self.verbose:
                    print("Changing scale of %s" % itm)
                self.params_clean[itm]["value"] = \
                    self.params_clean[itm].value * self.params_clean[itm].scale
                self.params_clean[itm].drop(columns=["scale"], inplace=True)

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
        self._wrangle_ghg()
        self._wrangle_emissions()

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

        c = 'fuel_rho'
        self.params_clean[c].set_index('fuel', inplace=True)

        # convert from eur/l to eur/kg
        c = 'c_fuel'
        self.params_clean[c].set_index('fuel', inplace=True)

        self.params_clean[c]["value"] = \
            self.params_clean[c]['value']\
            / self.params_clean['fuel_rho']['value']

        # convert parameters to output polyfit in kg/km
        c = "fuel_coeffs"
        self.params_clean[c] = pd.merge(self.params_clean[c]\
                                                .reset_index(drop=True),
                                        self.params_clean['fuel_rho']\
                                                .rename(columns={
                                                    'value': 'density'}),
                                        how="left",
                                        on="fuel")\
                                    .set_index(["vehicle", "fuel"])

        # multiply polynomial coefficients by density
        for col in ["a0", "a1", "a2", "a3"]:
            self.params_clean[c][col] = \
                self.params_clean[c][col] * self.params_clean[c]['density']
        self.params_clean[c].drop(columns=["density"], inplace=True)

        # multiply by density
        c = "fuel_acc"
        self.params_clean[c].set_index(['vehicle', 'fuel'], inplace=True)
        self.params_clean[c] = self.params_clean[c].stack()\
                                                   .to_frame().rename(
                                                        columns={0: 'value'})
        self.params_clean[c] = self.params_clean[c]\
                                    * self.params_clean['fuel_rho']
        self.params_clean[c] = self.params_clean[c].unstack(2)
        # simplify columns index
        self.params_clean[c].columns = self.params_clean[c]\
                                                .columns.get_level_values(1)

    def _wrangle_ghg(self):
        """
        Output:
            self.params_clean['r_ghg'] contains production of co2e in g/kg fuel
            consumed by vehicle type, fuel and gas
        """
        b = 'r_ghg'
        self.params_clean[b].set_index(['vehicle', 'fuel', 'gas'],
                                       inplace=True)
        # convert values of gas in g/kg to gCO2eq/kg by applying equivalence
        # factor
        self.params_clean[b]['value'] = self.params_clean[b]['value'] \
                                          * self.params_clean[b]['co2e_factor']
        self.params_clean[b].drop(columns=['co2e_factor'], inplace=True)

        b = 'c_ghg'
        self.params_clean[b].set_index('year', inplace=True)

    def _wrangle_emissions(self):
        """
        Output:
            self.params_clean['c_em'] contains cost of emission pollution
            in eur/kg fuel consumed by vehicle type, fuel, substance
            and environment
        """
        c = 'r_em'
        self.params_clean[c].set_index(['vehicle', 'fuel', 'substance'],
                                       inplace=True)

        #

        c = 'c_em'
        self.params_clean[c].set_index(['substance', 'environment'],
                                       inplace=True)

        # # multiply cost per kg by rate per kg fuel
        # c_em = self.params_clean[c][['value']] * self.params_clean['r_em']
        # # merge with elasticity and price level values
        # c_em = pd.merge(c_em.reset_index(),
        #                 self.params_clean[c][
        #                     ['price_level', 'gdp_growth_adjustment']],
        #                 how='left',
        #                 left_on=['substance', 'environment'],
        #                 right_index=True)
        # self.params_clean[c] = c_em.set_index(['vehicle', 'fuel',
        #                                        'substance', 'environment'])



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

        # fill 0 to acceleration coefficients
        self.RP[self.ACCELERATION_COLUMNS] = \
            self.RP[self.ACCELERATION_COLUMNS].fillna(0)

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

        for b in ["c_op", "vtts", "voc_l", 'voc_t', 'c_fuel', 'c_em']:
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
                                      * self.params_clean[b]["gdp_growth_adjustment"])

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

        # greenhouse gases have a fixed time evolution
        b = 'c_ghg'

        r_ghg = self.params_clean['r_ghg']
        c_ghg = self.params_clean['c_ghg'][['value']]

        self.UC[b] = pd.DataFrame(np.outer(r_ghg, c_ghg),
                                   index=r_ghg.index,
                                   columns=c_ghg.index)
        # restrict to years of analysis
        self.UC[b] = self.UC[b][self.yrs]

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
        if self.verbose:
            print("    Computing VTTS...")
        assert self.T0 is not None, "Compute travel time first."
        assert self.T1 is not None, "Compute travel time first."

        # Mask is given by the intensities, as these are zero
        # in the construction years
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

    def _compute_fuel_cost(self):
        if self.verbose:
            print("    Computing fuel cost...")
        assert self.QF0 is not None, \
            "Compute matrix of fuel consumption (QF0) first."
        assert self.QF1 is not None, \
            "Compute matrix of fuel consumption (QF1) first."

        b = "fuel"
        c = "c_fuel"
        self.B0[b] = (self.RF * self.UC[c]) * (self.QF0 * self.I0) * DAYS_YEAR
        self.B1[b] = (self.RF * self.UC[c]) * (self.QF1 * self.I1) * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()

    def _compute_fuel_consumption(self):
        """
        Compute the consumption in kg per vehicle by section, vehicle
        and fuel type.
        """
        if self.verbose:
            print("    Computing fuel consumption...")
        assert self.L0 is not None, "Compute length matrix first."
        assert self.L1 is not None, "Compute length matrix first."

        ###
        # velocity-dependent part
        ###

        # get a matrix of ones per vehicle, fuel type and year
        helper_ones = self.RF.copy() / self.RF.copy()

        # velocity by vehicle, fuel type, section and year
        # assumes vehicles with different fuel move at the same speed
        V0s = helper_ones * self.V0
        V0s = V0s.sort_index()

        V1s = helper_ones * self.V1
        V1s = V1s.sort_index()

        # quantity of fuel consumed per vehicle, fuel type and section
        self.QF0 = pd.DataFrame(0, columns=V0s.columns, index=V0s.index)
        self.QF1 = pd.DataFrame(0, columns=V1s.columns, index=V1s.index)

        for (veh, f), cs in self.params_clean['fuel_coeffs'].iterrows():
            # consumption-velocity curve coefficients
            c = cs.values

            # variant 0
            vs = V0s.loc[(veh, f)]
            qf = np.polynomial.polynomial.polyval(vs, c, tensor=False)
            self.QF0.loc[(veh, f)] = qf.values

            # variant 1
            vs = V1s.loc[(veh, f)]
            qf = np.polynomial.polynomial.polyval(vs, c, tensor=False)
            self.QF1.loc[(veh, f)] = qf.values

        # velocity part
        self.QFv0 = self.QF0 * self.L0
        self.QFv1 = self.QF1 * self.L1

        ##
        # acceleration-dependent part
        ##

        self.RP = self.RP.reset_index().set_index('id_road_section')

        # time matrix of acceleration ratios - variant 0, 1
        acceleration_mat0 = self.RP.loc[self.RP['variant'] == 0,
                                        self.ACCELERATION_COLUMNS]\
                                    .stack().to_frame()
        acceleration_mat1 = self.RP.loc[self.RP['variant'] == 1,
                                        self.ACCELERATION_COLUMNS] \
                                    .stack().to_frame()

        # reindex to the original columns
        self.RP = self.RP.reset_index()\
                         .set_index(['id_road_section', 'variant'])

        acceleration_mat0.columns = ['ratio']
        acceleration_mat0.index.names = ['id_road_section', 'acceleration']
        acceleration_mat1.columns = ['ratio']
        acceleration_mat1.index.names = ['id_road_section', 'acceleration']

        acceleration_mat0 = pd.DataFrame(np.outer(acceleration_mat0['ratio'],
                                        np.ones_like(self.yrs)),
                                        columns=self.yrs,
                                        index=acceleration_mat0.index)

        acceleration_mat1 = pd.DataFrame(np.outer(acceleration_mat1['ratio'],
                                         np.ones_like(self.yrs)),
                                         columns=self.yrs,
                                         index=acceleration_mat1.index)

        # time-matrix of fuel consumption
        fuel_acc_mat = self.params_clean['fuel_acc'].stack().to_frame()
        fuel_acc_mat.columns = ['value']
        fuel_acc_mat.index.names = ['vehicle', 'fuel', 'acceleration']

        fuel_acc_mat = pd.DataFrame(np.outer(fuel_acc_mat['value'],
                                             np.ones_like(self.yrs)),
                                    columns=self.yrs,
                                    index=fuel_acc_mat.index)

        # ones in the index and columns structure of intensity dataframes
        ones0 = self.I0/self.I0
        ones1 = self.I1/self.I1

        QFa0 = ((helper_ones * ones0) * acceleration_mat0 * fuel_acc_mat)
        QFa1 = ((helper_ones * ones1) * acceleration_mat1 * fuel_acc_mat)

        # acceleration dependent part
        self.QFa0 = QFa0.reset_index()\
                        .groupby(['vehicle', 'fuel', 'id_road_section'])[self.yrs]\
                        .sum()
        self.QFa1 = QFa1.reset_index() \
                        .groupby(['vehicle', 'fuel', 'id_road_section'])[self.yrs]\
                        .sum()

        self.QF0 = self.QFv0 + self.QFa0
        self.QF1 = self.QFv1 + self.QFa1

    def _compute_greenhouse(self):
        if self.verbose:
            print("    Computing greenhouse gases...")
        assert self.RF is not None, "Compute matrix of fuel ratios (RF) first."
        assert self.QF0 is not None, "Compute matrix of fuel consumption (QF0) first."
        assert self.QF1 is not None, "Compute matrix of fuel consumption (QF1) first."

        b = "ghg"
        self.B0[b] = (self.UC['c_ghg'] * self.RF)  \
                     * (self.QF0 * self.I0) * DAYS_YEAR
        self.B1[b] = (self.UC['c_ghg']  * self.RF) \
                     * (self.QF1 * self.I1) * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()

    def _compute_emissions(self):
        if self.verbose:
            print("    Computing emissions...")
        assert self.RF is not None, "Compute matrix of fuel ratios (RF) first."
        assert self.QF0 is not None, "Compute matrix of fuel consumption (QF0) first."
        assert self.QF1 is not None, "Compute matrix of fuel consumption (QF1) first."

        # merge environment variable onto intensity dataframe
        I0_env = pd.merge(self.I0.reset_index(),
                          self.RP.loc[slice(None, 0),
                                      ['environment']].reset_index(),
                          how='left',
                          on='id_road_section')
        I0_env.set_index(['id_road_section', 'vehicle','environment'],
                         inplace=True)

        I1_env = pd.merge(self.I1.reset_index(),
                          self.RP.loc[slice(None, 1),
                                      ['environment']].reset_index(),
                          how='left',
                          on='id_road_section')
        I1_env.set_index(['id_road_section', 'vehicle', 'environment'],
                         inplace=True)

        # compute unit cost of emissions per kg fuel for every year
        uc_em = pd.DataFrame(self.UC['c_em'].stack(),
                             columns=['value']) * self.params_clean['r_em']
        uc_em = uc_em.unstack('year')
        uc_em.columns = uc_em.columns.get_level_values(1)

        b = "em"
        lvl_order = ['id_road_section', 'vehicle', 'fuel',
                     'substance', 'environment']
        self.B0[b] = I0_env * self.RF * self.QF0 * uc_em * DAYS_YEAR
        self.B0[b] = self.B0[b].reorder_levels(lvl_order).sort_index()
        self.B1[b] = I1_env * self.RF * self.QF1 * uc_em * DAYS_YEAR
        self.B1[b] = self.B1[b].reorder_levels(lvl_order).sort_index()
        
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
