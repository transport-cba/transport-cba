import pandas as pd
import numpy as np
from numpy.matlib import repmat
import numpy_financial as npf
import time
from .param_container import ParamContainer


VEHICLE_TYPES = ["car", "lgv", "hgv", "bus"]
ENVIRONMENTS = ["intravilan", "extravilan"]
DAYS_YEAR = 365.0

INPUT_SHEETS = ['road_params', 'capex', 'intensities_0', 'intensities_1', 'velocities_0', 'velocities_1']

IDX_CAPEX = ["land", "pavements", "bridges", "tunnels", "buildings",
    "slope_stabilisation", "retaining_walls", "noise_barriers",
    "safety_features", "supervision", "planning_design"]

COLS_ROAD_PARAMS = ["name", "variant_0", "variant_1", "length",
    "length_bridges", "length_tunnels", "category", "lanes",
    "environment", "width", "layout", "toll_sections"]

IDX_NAME_VEL = ["id_section", "vehicle"]


class RoadCBA(ParamContainer):
    """The object to perform CBA computation for roads"""
    def __init__(self,
            init_year,
            country,
            year_price_level=None,
            period=30,
            fin_discount_factor=0.04,
            eco_discount_factor=0.05,
            currency="EUR",
            verbose=False
        ):
        """
        Inputs
        ------
        - init_year : int, initial year of construction
        - country : str, country code
        - year_price_level : int, year of economic analysis, by default same as init_year
        - period : int, number of years for the economic analysis, default 30
        - fin_discount_factor : float, discount factor for financial analysis
        - eco_discount_factor : float, discount factor for economic analysis
        """
        self.yr_init = init_year
        self.N_yr = period
        self.yr_end = self.yr_init + self.N_yr - 1
        self.yr_price_level = year_price_level
        if self.yr_price_level is None:
            self.yr_price_level = self.yr_init
        if self.yr_init != self.yr_price_level:
            print("Warning: start year not same as price level.")
        self.N_yr_build = None
        self.N_yr_op = None
        self.yr_op = None
        self.yrs = np.arange(self.yr_init, self.yr_init + self.N_yr)

        self.r_fin = fin_discount_factor
        self.r_eco = eco_discount_factor

        self.country = country
        self.currency = currency

        self.veh_types = VEHICLE_TYPES
        self.envs = ENVIRONMENTS

        self.verbose = verbose

        # define empty frames
        self.RP = None # road parameters
        self.C_fin = None # financial capex
        self.C_eco = None # economic capex
        self.O0_fin = None # financial opex in variant 0
        self.O0_eco = None # economic opex in variant 0
        self.O1_fin = None
        self.O1_eco = None

        self.V0 = None
        self.V1 = None
        self.L = None
        self.T0 = None
        self.T1 = None
        self.I0 = None
        self.I1 = None

        self.RF = None      # ratio of fuel types
        self.QF0 = None     # quantity of fuel burnt on a section in variant 0
        self.QF1 = None     # quantity of fuel burnt on a section in variant 1

        self.UC = {}        # unit costs for various components
        self.B0 = {}        # benefits in 0th variant
        self.B1 = {}        # benefits in 1st variant
        self.NB = {}        # net benefits
        self.NC = {}        # net costs

        # initialise resulting variables
        self.economic_indicators = None

        # initialise parameter container
        super().__init__(self.country, self.yr_price_level, verbose=self.verbose)

        # intro information
        print("Computing cost benefit analysis...")
        print(f"Initial year: {self.yr_init}")
        print(f"Price level: {self.yr_price_level}")

    # =====
    # Initialisation functions
    # =====
    def prepare_parameters(self, source=None):
        """Read in CBA parameters and adjust them for furhter use.
        Another source than the built-in one can be chosen."""
        if source is None:
            super().read_raw_params()
        else:
            raise NotImplementedError()

        super().adjust_cpi()
        super().adjust_gdp_growth()
        super().adjust_greenhouse_cost()
        super().clean_params()
        super().adjust_price_level()
        super().wrangle_params()

    
    def replace_parameter(self, param):
        """Replace a specific parameter frame"""
        raise NotImplementedError()


    def _assign_core_variables(self):
        """After reading project inputs define years of operation
        and various arrays of sections"""
        if self.C_fin is not None:
            self.yr_op = int(self.C_fin.columns[-1]) + 1
            self.N_yr_build = len(self.C_fin.columns)
            self.N_yr_op = self.N_yr - self.N_yr_build
            self.yrs_op = np.arange(self.yr_init + self.N_yr_build, self.yr_init + self.N_yr)

            self.secs = self.RP.index
            self.secs_0 = self.RP[self.RP.variant_0 == 1].index
            self.secs_1 = self.RP[self.RP.variant_1 == 1].index
            self.secs_old = \
                self.RP[(self.RP.variant_0 == 1) & (self.RP.variant_1 == 1)].index
            self.secs_repl = \
                self.RP[(self.RP.variant_0 == 1) & (self.RP.variant_1 == 0)].index
            self.secs_new = \
                self.RP[(self.RP.variant_0 == 0) & (self.RP.variant_1 == 1)].index


    def _wrangle_inputs(self):
        """Modify input matrices of intensities and velocities
        in line with the economic period and other global requirements.
        Ensure that the columns representing years are integers."""
        self.I0.columns = self.I0.columns.astype(int)
        self.I1.columns = self.I1.columns.astype(int)
        self.V0.columns = self.V0.columns.astype(int)
        self.V1.columns = self.V1.columns.astype(int)

        # remove unused rows in 0th variant
        self.I0 = self.I0.loc[self.secs_0]
        self.V0 = self.V0.loc[self.secs_0]

        if self.I0.columns[-1] < self.yr_end and self.verbose:
            print("warning: I0 not forecast until the end of period, fill by zeros")
        # self.I0 = self.I0[self.yrs].fillna(0)

        if self.I1.columns[-1] < self.yr_end and self.verbose:
            print("warning: I1 not forecast until the end of period, fill by zeros")
        # self.I1 = self.I1[self.yrs].fillna(0)

        if self.V0.columns[-1] < self.yr_end and self.verbose:
            print("warning: V0 not forecast until the end of period, fill by zeros")
        # self.V0 = self.V0[self.yrs].fillna(0)

        if self.V1.columns[-1] < self.yr_end and self.verbose:
            print("warning: V0 not forecast until the end of period, fill by zeros")
        # self.V1 = self.V1[self.yrs].fillna(0)


    def read_project_inputs(self,
            df_road_params,
            df_capex,
            df_int_0,
            df_int_1,
            df_vel_0,
            df_vel_1
        ):
        """Read input dataframes.

        Inputs
        ------
        - road parameters : pd.dataframe
        - capital investment (CAPEX) : pd.dataframe
        - intensities in variant 0 : pd.dataframe
        - intensities in variant 1 : pd.dataframe
        - velocities in variant 0 : pd.dataframe
        - velocities in variant 1 : pd.dataframe
        """
        if self.verbose:
            print("Reading project inputs from dataframes...")
        self.RP = df_road_params

        self.C_fin = df_capex
        self.C_fin.columns = self.C_fin.columns.astype(int)
        self.C_fin = self.C_fin[sorted(self.C_fin.columns)]

        self.I0 = df_int_0
        self.I0.columns = self.I0.columns.astype(int)
        self.I0 = self.I0[sorted(self.I0.columns)]
       
        self.I1 = df_int_1
        self.I1.columns = self.I1.columns.astype(int)
        self.I1 = self.I1[sorted(self.I1.columns)]

        self.V0 = df_vel_0
        self.V0.columns = self.V0.columns.astype(int)
        self.V0 = self.V0[sorted(self.V0.columns)]
        
        self.V1 = df_vel_1
        self.V1.columns = self.V1.columns.astype(int)
        self.V1 = self.V1[sorted(self.V1.columns)]

        # assign core variables
        self._assign_core_variables()
        self._wrangle_inputs()

    
    def read_project_inputs_excel(self, file_xls):
        if not file_xls.split(".")[-1] in ["xls", "xlsx"]:
            raise ValueError(f"invalid file extension, expected xls or xlsx")
        
        if self.verbose:
            print("Reading project inputs from %s..." % file_xls)

        xls = pd.ExcelFile(file_xls)
        
        if set(xls.sheet_names) != set(INPUT_SHEETS):
            raise ValueError(f"wrong sheet names, submitted: {xls.sheet_names}, required: {INPUT_SHEETS}")

        # read the sheets
        self.RP = xls.parse("road_params", index_col=0)
        self.C_fin = xls.parse("capex").reset_index(drop=True)
        if "category" not in self.C_fin.columns:
            print("warning: category not in index of CAPEX")
            self.C_fin.set_index(['item'], inplace=True)
        else:
            self.C_fin.set_index(['item', 'category'], inplace=True)

        self.I0 = xls.parse("intensities_0").reset_index(drop=True)
        self.I0.set_index(["id_section", "vehicle"], inplace=True)
        self.I0 = self.I0[sorted(self.I0.columns)]

        self.I1 = xls.parse("intensities_1").reset_index(drop=True)
        self.I1.set_index(["id_section", "vehicle"], inplace=True)
        self.I1 = self.I1[sorted(self.I1.columns)]

        self.V0 = xls.parse("velocities_0").reset_index(drop=True)
        self.V0.set_index(["id_section", "vehicle"], inplace=True)
        self.V0 = self.V0[sorted(self.V0.columns)]

        self.V1 = xls.parse("velocities_1").reset_index(drop=True)
        self.V1.set_index(["id_section", "vehicle"], inplace=True)
        self.V1 = self.V1[sorted(self.V1.columns)]

        self._assign_core_variables()
        self._wrangle_inputs()


    def _verify_input_integrity(self):
        """Perform various checks on the quality of the inputs"""
        assert self.I0 is not None, "I0 table undefined"
        assert self.I0.index.names == IDX_NAME_VEL, f"incorrect indices of I0 table, expect {IDX_NAME_VEL}"

        assert self.I1 is not None, "I1 table undefined"
        assert self.I1.index.names == IDX_NAME_VEL, f"incorrect indices of I1 table, expect {IDX_NAME_VEL}"

        assert self.V0 is not None, "V0 table undefined"
        assert self.V0.index.names == IDX_NAME_VEL, f"incorrect indices of V0 table, expect {IDX_NAME_VEL}"

        assert self.V1 is not None, "V1 table undefined"
        assert self.V1.index.names == IDX_NAME_VEL, f"incorrect indices of V1 table, expect {IDX_NAME_VEL}"

        # consistency of columns
        if not (set(self.I0.columns) == set(self.I1.columns) == set(self.V0.columns) == set(self.V1.columns)):
            raise ValueError(f"year ranges in intensity and velocity tables are not same")
        
        for dff in [self.I0, self.I1, self.V0, self.V1]:
            check_year_order(dff, self.yr_init, self.yr_end, self.N_yr)
            check_table_content(dff)

        # road parameters
        assert self.RP is not None, "road parameters table undefined"
        if set(self.RP.columns) != set(COLS_ROAD_PARAMS):
            raise ValueError((
                f"Incorrect columns of road parameters table, "
                f"submitted: {self.RP.columns}, required: {COLS_ROAD_PARAMS}"))

        # capex table
        assert self.C_fin is not None, "CAPEX table undefined"
        # FIX, define required capex index correctly
        # if not set(self.C_fin.index) == set(IDX_CAPEX):
        #     raise ValueError((
        #         f"incorrect index of CAPEX, "
        #         f"submitted: {self.C_fin.index}, required: {IDX_CAPEX}"))

    def _verify_param_integrity(self):
        """Verify that all the relevant parameter frames contain 
        correct columns and indices"""
        # TODO: check if years agree
        pass


    # =====
    # Computing CAPEX, OPEX and residual value
    # =====
    def _wrangle_capex(self):
        """Removing columns and squeezing investment expenses
        in years before the start into the first year
        of the economic period"""

        if "category" in self.C_fin.columns:
            self.C_fin.drop(columns="category", inplace=True)
        if "category" in self.C_fin.index.names:
            self.C_fin = self.C_fin.reset_index("category")\
                .drop(columns="category")
        if "total" in self.C_fin.columns:
            self.C_fin.drop(columns="total", inplace=True)
        self.C_fin.columns = self.C_fin.columns.astype(int)

        self.C_fin.fillna(0, inplace=True)

        # collect investment before the first year
        capex_yrs = self.C_fin.columns
        if len(capex_yrs[capex_yrs < self.yr_init]) > 0:
            if self.verbose:
                print(f"Squeezing CAPEX {capex_yrs} into the given economic period starting with {self.yr_init}...")
            if self.yr_init not in capex_yrs:
                self.C_fin[self.yr_init] = 0.0
            yrs_bef = capex_yrs[capex_yrs < self.yr_init]
            yrs_aft = capex_yrs[capex_yrs >= self.yr_init]
            self.C_fin[self.yr_init] += self.C_fin[yrs_bef].sum(1)
            self.C_fin = self.C_fin[yrs_aft]


    def compute_capex(self):
        """Apply conversion factors to compute
        CAPEX for the economic analysis."""
        if self.verbose:
            print("Computing CAPEX...")

        self._wrangle_capex()

        # reindex columns
        self.C_fin = pd.DataFrame(self.C_fin, columns=self.yrs).fillna(0)
        self.C_fin_tot = pd.DataFrame(self.C_fin.sum(1), columns=["value"])

        # apply conversion factors to get economic CAPEX
        self.cf = self.C_fin.copy()\
            .merge(self.df_clean["conv_fac"][["aggregate"]], \
            how="left", on="item")\
            .fillna(self.df_clean["conv_fac"]\
            .loc["construction", "aggregate"])["aggregate"]
        self.C_eco = self.C_fin.multiply(self.cf, axis=0)
        self.C_eco_tot = pd.DataFrame(self.C_eco.sum(1), columns=["value"])

        self.NC["capex"] = self.C_eco.sum()


    def compute_residual_value(self):
        """Create a dataframe of residual values by each element"""
        if self.verbose:
            print("Computing residual value...")

        RV = self.df_clean["res_val"].copy()
        RV.replacement_cost_ratio.fillna(1.0, inplace=True)
        RV["op_period"] = self.N_yr_op
        RV["replace"] = np.where(RV.lifetime <= RV.op_period, 1, 0)
        RV["rem_ratio"] = np.where(
            RV.lifetime <= RV.op_period,
            (2*RV.lifetime - RV.op_period) / RV.lifetime,
            (RV.lifetime - RV.op_period) / RV.lifetime
        ).round(2)
        RV.rem_ratio.fillna(1.0, inplace=True) # fill land

        # financial
        self.RV_fin = RV.merge(self.C_fin_tot, how="left", on="item").fillna(0)
        self.RV_fin["res_value"] = np.where(self.RV_fin.replace == 0,
            self.RV_fin.value * self.RV_fin.rem_ratio,
            self.RV_fin.value * self.RV_fin.rem_ratio * self.RV_fin.replacement_cost_ratio)

        # economic
        self.RV_eco = RV.merge(self.C_eco_tot, how="left", on="item")
        self.RV_eco["res_value"] = np.where(
            self.RV_eco.replace == 0,
            self.RV_eco.value * self.RV_eco.rem_ratio,
            self.RV_eco.value * self.RV_eco.rem_ratio * self.RV_eco.replacement_cost_ratio)

        self.NB["res_val"] = pd.Series(
            np.array([0] * (self.N_yr-1) + [1]) * self.RV_eco.res_value.sum(),
            index=self.yrs
        )


    def compute_opex(self):
        """Create a dataframe of operation costs (OPEX)."""
        if self.verbose:
            print("Computing OPEX...")

        assert len(self.UC.keys()) != 0, "before computing OPEX, create unit costs first"

        UC = self.UC["c_op"].copy()
        lvl_order = ["id_section", "operation_type", "item"]

        # create area matrix
        def define_area(x):
            return x if x in ["tunnels", "bridges"] else "pavements"

        UC = UC.reset_index(["item"])
        UC["area_type"] = UC.item.map(lambda x: define_area(x))
        UC = UC.reset_index().set_index(
            ["category", "operation_type", "area_type", "item"]).sort_index()
        
        # variant 0
        RA0 = self.RP.loc[self.secs_0, 
            ["category", "length", "length_bridges", "length_tunnels", "width"]].copy()
        RA0["pavements"] = RA0.width * RA0.length * 1e3
        RA0["bridges"] = RA0.width * RA0.length_bridges * 1e3
        RA0["tunnels"] = RA0.width * RA0.length_tunnels * 1e3
        RA0 = RA0.drop(
            columns=["length", "length_bridges", "length_tunnels", "width"])
        RA0 = RA0.reset_index().melt(id_vars=["id_section", "category"],
            var_name="area_type", value_name="value")
        RA0 = RA0.groupby(["id_section", "category", "area_type"])\
            [["value"]].sum()
        
        # time matrix of road areas
        RA0 = pd.DataFrame(np.outer(RA0.value, np.ones_like(self.yrs)), \
            index=RA0.index, columns=self.yrs)
        
        # summary
        assert hasattr(self, "mask0"), "Create OPEX mask first."
        self.O0_fin = (RA0 * (UC * self.mask0)).dropna().droplevel(
            ["category", "area_type"]).reorder_levels(lvl_order).sort_index()
        
        # variant 1
        O1_old = self.O0_fin.loc[self.secs_old].copy()
        O1_repl = self.O0_fin.loc[self.secs_repl].copy()
        if not O1_repl.empty:
            O1_repl[self.yrs_op] = 0.0
        
        RA1 = self.RP.loc[self.secs_new,
            ["category", "length", "length_bridges", "length_tunnels", "width"]].copy()
        RA1["pavements"] = RA1.width * RA1.length * 1e3
        RA1["bridges"] = RA1.width * RA1.length_bridges * 1e3
        RA1["tunnels"] = RA1.width * RA1.length_tunnels * 1e3
        RA1 = RA1.drop(
            columns=["length", "length_bridges", "length_tunnels", "width"])
        RA1 = RA1.reset_index().melt(id_vars=["id_section", "category"],
            var_name="area_type", value_name="value")
        RA1 = RA1.groupby(["id_section", "category", "area_type"])[["value"]].sum()
        
        # time matrix of road areas
        RA1 = pd.DataFrame(np.outer(RA1.value, np.ones_like(self.yrs)),
            index=RA1.index, columns=self.yrs)
        
        # summary
        O1_new = (RA1 * (UC * self.mask1)).dropna().droplevel(
            ["category", "area_type"]).reorder_levels(lvl_order).sort_index()
        self.O1_fin = pd.concat([O1_old, O1_repl, O1_new]).sort_index()
        
        # economic values
        c = "conv_fac"
        cf = self.df_clean[c]\
            .loc[self.df_clean[c].expense_type == "operation", "aggregate"]
        cf.index.name = "operation_type"
        cf = pd.DataFrame(np.outer(cf, np.ones_like(self.yrs)),
            columns=self.yrs, index=cf.index)
        
        self.O0_eco = self.O0_fin * cf
        self.O1_eco = self.O1_fin * cf
        self.NC["opex"] = self.O1_eco.sum() - self.O0_eco.sum()


    def _compute_toll(self):
        raise NotImplementedError()


    # =====
    # Preparation functions
    # =====
    def _create_unit_cost_matrix(self):
        """Define the unit cost (UC) matrices for each benefit"""
        if self.verbose:
            print("Creating time matrices for benefits...")

        for b in ["c_op", "toll_op", \
            "vtts", "voc", "c_fuel", "c_acc", "c_em", "noise"]:
            if self.verbose:
                print("    Creating: %s" % b)
            self.UC[b] = \
                pd.DataFrame(columns=self.yrs, index=self.df_clean[b].index)
            self.UC[b][self.yr_init] = self.df_clean[b].value
            for yr in self.yrs[1:]:
                self.UC[b][yr] = \
                    self.UC[b][self.yr_init]# * self.cpi.loc[yr, "cpi_index"]
                if "gdp_growth_adjustment" in self.df_clean[b].columns:
                    self.UC[b][yr] = self.UC[b][yr] \
                    * (1.0 + self.gdp_growth.loc[yr].gdp_growth \
                    * self.df_clean[b].gdp_growth_adjustment)

            if b in ["noise"]:
                self.UC[b] = self.UC[b].sort_index().round(5)
            else:
                self.UC[b] = self.UC[b].sort_index().round(2)

        # greenhouse unit cost computed separately due to its structure
        b = "c_ghg"
        self.UC[b] = pd.DataFrame(self.df_clean[b].loc[self.yr_init:self.yr_end, "value"])
        self.UC[b].columns = ["co2eq"] 
        self.UC[b] = self.UC[b].T

    
    def _create_opex_cost_mask(self):
        """Compose a time matrix of zeros and ones indicating 
        if maintanance has to be performed in a given year."""
        lvl_order = ["category", "operation_type", "item"]
        # variant 0
        mask0 = pd.DataFrame(0, \
            index=self.df_clean["c_op"].index, columns=self.yrs)

        for itm in mask0.index:
            p = self.df_clean["c_op"].loc[itm, "periodicity"].astype(int)
            if p == 1:
                mask0.loc[itm] = 1
            else:
                v = np.zeros(mask0.shape[1]).astype(int)
                for i, _ in enumerate(v):
                    if (i+1) % p == 0:
                        v[i] = 1
                mask0.loc[itm] = v
        
        self.mask0 = mask0.reorder_levels(lvl_order).sort_index()
        
        # variant 1
        mask1 = pd.DataFrame(0, index=self.df_clean["c_op"].index, columns=self.yrs_op)
        
        for itm in mask1.index:
            p = self.df_clean["c_op"].loc[itm, "periodicity"].astype(int)
            if p == 1:
                mask1.loc[itm] = 1
            else:
                v = np.zeros(mask1.shape[1]).astype(int)
                for i, _ in enumerate(v):
                    if (i+1) % p == 0:
                        v[i] = 1
                mask1.loc[itm] = v
        
        mask1 = pd.DataFrame(mask1, columns=self.yrs).fillna(0).astype(int)
        self.mask1 = mask1.reorder_levels(lvl_order).sort_index()


    def _create_length_matrix(self):
        """Create the matrix of lengs with years as columns"""
        if self.verbose:
            print("Creating length matrix...")
        self.L = pd.DataFrame(
            np.outer(self.RP.length, np.ones_like(self.yrs)),
            columns=self.yrs, index=self.RP.index
        )


    def _compute_travel_time_matrix(self):
        """Compute travel time by road section and vehicle type"""
        if self.verbose:
            print("Creating matrices of travel times...")
        assert self.L is not None, "Compute length matrix first."

        self.T0 = (self.L / self.V0).replace([np.inf, -np.inf], 0.0)
        self.T1 = (self.L / self.V1).replace([np.inf, -np.inf], 0.0)


    def _create_fuel_ratio_matrix(self):
        if self.verbose:
            print("Creating matrix of fuel ratios by vehicle...")
        rfuel = self.df_clean["r_fuel"].ratio.sort_index()
        self.RF = pd.DataFrame(repmat(rfuel, self.N_yr, 1).T, \
            columns=self.yrs, index=rfuel.index)


    # =====
    # Functions to compute economic benefits and costs
    # =====
    def economic_analysis(self, param_source=None):
        """Wrapping method for the overall computation
        of costs, benefits and overall indicators (ENPV, ERR, BCR)."""
        ti = time.time()
        self.prepare_parameters(source=param_source)
        self.compute_costs_benefits()
        self.compute_economic_indicators()

        self.compute_time = time.time() - ti
        print(f"Computation time: {self.compute_time} s.")
        return self.economic_indicators


    def compute_costs_benefits(self):
        """Compute financial and economic costs and benefits"""
        self._verify_input_integrity()
        self._verify_param_integrity()

        # unit costs
        if self.verbose:
            print("Preparing unit values...")
        self._create_unit_cost_matrix()
        self._create_opex_cost_mask()

        # costs
        if self.verbose:
            print("Computing costs...")
        self.compute_capex()
        self.compute_opex()
        self.compute_residual_value()

        # benefits
        if self.verbose:
            print("Computing benefits...")

        self._create_length_matrix()
        self._compute_travel_time_matrix()
        
        self._compute_vtts()
        self._compute_voc()
        self._compute_accidents()
        self._create_fuel_ratio_matrix()
        self._compute_fuel_consumption()
        self._compute_fuel_cost()
        self._compute_greenhouse()
        self._compute_emissions()
        self._compute_noise()


    def compute_economic_indicators(self):
        """Perform economic analysis"""
        assert self.NB is not None, "Compute economic benefits first."

        if self.verbose:
            print("\nComputing ENPV, ERR, BCR...")
        self.df_eco = pd.DataFrame(self.NB).T
        self.df_eco = pd.concat(
            [-pd.DataFrame(self.NC).T, pd.DataFrame(self.NB).T],
            keys=["cost", "benefit"], names=["type", "item"]
        ).round(2)

        self.df_eco = self.df_eco.fillna(0.0) # remove nans

        self.df_enpv = pd.DataFrame(
            self.df_eco.apply(
                lambda x: npf.npv(self.r_eco, x), axis=1
            ).round(2), columns=["value"]
        )

        # compute economic indicators
        self.ENPV = npf.npv(self.r_eco, self.df_eco.sum())
        self.ERR = npf.irr(self.df_eco.sum())
        self.EBCR = (self.df_enpv.loc["benefit"].sum() / -self.df_enpv.loc["cost"].sum()).value
        
        # format economic indicators
        vals = [self.ENPV / 1e6, self.ERR * 100.0, self.EBCR]
        vals = [np.round(val, 3) for val in vals]
        self.economic_indicators = pd.DataFrame({
            "Quantity": ["ENPV", "ERR", "BCR"],
            "Unit": ["M "+self.currency.upper(), "%", ""],
            "Value": vals,
        })


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
        assert self.L is not None, "Compute length matrix first."
        if self.verbose:
            print("    Computing VOC...")
        assert self.L is not None, "Compute length matrix first."

        b = "voc"
        dum = pd.DataFrame(1, index=pd.MultiIndex.from_product(\
            [self.L.index, self.UC[b].index]), columns=self.yrs)
        dum.index.names = ["id_section", "vehicle"]
        self.B0[b] = ((self.UC[b] * dum) * self.L) * self.I0 * DAYS_YEAR
        self.B1[b] = ((self.UC[b] * dum) * self.L) * self.I1 * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()


    def _compute_accidents(self):
        assert self.L is not None, "Compute length matrix first."
        if self.verbose:
            print("    Computing accidents...")
        assert self.L is not None, "Compute length matrix first."

        b = "accidents"
        scale = 1e-8
        LL = self.L.merge(\
            self.RP[["lanes", "environment", "category", "layout"]], \
            how="left", on="id_section").reset_index()\
            .set_index(["id_section", "lanes", "environment", "category", \
            "layout"]).reorder_levels(\
                ["id_section", "category", "lanes", "layout", "environment"])
        LL.columns = LL.columns.astype(int)

        UCA = (LL * self.UC["c_acc"] * scale)\
            .droplevel(["layout", "lanes", "category", "environment"])\
            .dropna(subset=[self.yr_init]).sort_index()

        self.B0[b] = UCA * self.I0 * DAYS_YEAR
        self.B1[b] = UCA * self.I1 * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()


    def _compute_fuel_consumption(self):
        """Compute the consumption by section, vehicle and fuel type"""
        if self.verbose:
            print("    Computing fuel consumption...")
        assert self.L is not None, "Compute length matrix first."

        # polynomial coefficients and consumption function
        def vel2cons(coeffs, v):
            """Convert velocity in km/h to fuel consumption in
            kg/km via a polynomial"""
            return np.polyval(coeffs[::-1], v)

        # length matrix with appropriate division of fuel/vehicle types
        dum = pd.DataFrame(1, index=pd.MultiIndex.from_product(\
            [self.secs, self.veh_types]), columns=self.yrs)
        dum.index.names = ["id_section", "vehicle"]
        L = self.L * dum
        
        ind = self.df_clean["r_fuel"].reset_index()[["vehicle", "fuel"]]
        L = L.reset_index().merge(ind, how="left", on="vehicle")\
            .set_index(["id_section", "vehicle", "fuel"])
        L = L.sort_index()

        # quantity of fuel, variant 0
        self.QF0 = pd.DataFrame(columns=self.yrs, index=L.loc[self.secs_0].index)
        for ind, _ in self.QF0.iterrows():
            ids, veh, f = ind
            self.QF0.loc[(ids, veh, f)] = self.V0.loc[(ids, veh)]\
                .map(lambda v: vel2cons(self.df_clean["fuel_coeffs"].loc[(veh, f)], v)) * L.loc[ind]

        # quantity of fuel, variant 1
        self.QF1 = pd.DataFrame(columns=self.yrs, index=L.loc[self.secs_1].index)
        for ind, _ in self.QF1.iterrows():
            ids, veh, f = ind
            self.QF1.loc[(ids, veh, f)] = self.V1.loc[(ids, veh)]\
                .map(lambda v: vel2cons(\
                self.df_clean["fuel_coeffs"].loc[(veh, f)], v)) * L.loc[ind]


    def _compute_fuel_cost(self):
        if self.verbose:
            print("    Computing fuel cost...")
        assert self.RF is not None, "Compute matrix of fuel ratios (RF) first."
        assert self.QF0 is not None, "Compute matrix of fuel consumption (QF0) first."
        assert self.QF1 is not None, "Compute matrix of fuel consumption (QF1) first."

        b = "fuel"
        c = "c_fuel"
        self.B0[b] = (self.UC[c] * self.RF) * (self.QF0 * self.I0) * DAYS_YEAR
        self.B1[b] = (self.UC[c] * self.RF) * (self.QF1 * self.I1) * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()


    def _compute_greenhouse(self):
        if self.verbose:
            print("    Computing greenhouse gases...")
        assert self.RF is not None, "Compute matrix of fuel ratios (RF) first."
        assert self.QF0 is not None, "Compute matrix of fuel consumption (QF0) first."
        assert self.QF1 is not None, "Compute matrix of fuel consumption (QF1) first."
        b = "ghg"

        # UCG: unit cost of greenhouse gases in EUR/kg(fuel)
        UCG = pd.DataFrame(\
            np.outer(self.df_clean["r_ghg"].values, self.UC["c_ghg"].values),\
            index=self.df_clean["r_ghg"].index, columns=self.yrs)
        
        self.B0[b] = (UCG * self.RF) * (self.QF0 * self.I0) * DAYS_YEAR
        self.B1[b] = (UCG * self.RF) * (self.QF1 * self.I1) * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()


    def _compute_emissions(self):
        if self.verbose:
            print("    Computing emissions...")
        assert self.RF is not None, "Compute matrix of fuel ratios (RF) first."
        assert self.QF0 is not None, "Compute matrix of fuel consumption (QF0) first."
        assert self.QF1 is not None, "Compute matrix of fuel consumption (QF1) first."
        
        b = "emissions"
        RE = pd.DataFrame(repmat(self.df_clean["r_em"].value, self.N_yr, 1).T, 
                  columns=self.yrs, index=self.df_clean["r_em"].index)

        # UCE: unit cost of emissions in EUR/kg(fuel)
        UCE = RE * self.UC["c_em"]
        UCE = UCE.groupby(["fuel", "vehicle", "environment"]).sum()
        UCE = UCE.reset_index()\
            .set_index(["environment", "vehicle", "fuel"]).sort_index()

        # add section ID
        UCE = UCE.reset_index().merge(self.RP.environment.reset_index(), \
            how="left", on="environment").set_index(\
            ["id_section", "environment", "vehicle", "fuel"]).sort_index()
        
        lvl_order = ["id_section", "vehicle", "fuel", "environment"]
        self.B0[b] = (UCE * self.RF).reorder_levels(lvl_order).sort_index() * (self.QF0 * self.I0) * DAYS_YEAR
        self.B1[b] = (UCE * self.RF).reorder_levels(lvl_order).sort_index() * (self.QF1 * self.I1) * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()


    def _compute_noise(self):
        if self.verbose:
            print("    Computing noise...")
        assert self.L is not None, "Compute length matrix first."

        b = "noise"
        L = self.L.reset_index().merge(self.RP.environment.reset_index())
        L = L.set_index(["id_section", "environment"])
        
        # CN: cost of noise in EUR
        CN = (self.UC[b] * L).reorder_levels(["id_section", "environment", "vehicle"]).sort_index()

        self.B0[b] = CN * self.I0 * DAYS_YEAR
        self.B1[b] = CN * self.I1 * DAYS_YEAR
        self.NB[b] = self.B0[b].sum() - self.B1[b].sum()


    """
    Saving data
    """
    def save_results_to_excel(self, fname_res="cba_results.xlsx"):
        with pd.ExcelWriter(fname_res) as writer:
            for benefit in ["vtts", "voc", "fuel", "accidents", "ghg", "emissions", "noise"]:
                self.B0[benefit].to_excel(writer, sheet_name=f"{benefit}_0")
                self.B1[benefit].to_excel(writer, sheet_name=f"{benefit}_1")
            self.economic_indicators.to_excel(writer, sheet_name="indicators")
        
        print(f"CBA output saved to {fname_res}")     


    def save_inputs_to_excel(self, fname_out="cba_inputs.xlsx"):
        """save the input sheets to excel"""
        with pd.ExcelWriter(fname_out) as writer:
            self.RP.to_excel(writer, sheet_name="road_params")
            self.C_fin.to_excel(writer, sheet_name="capex")
            self.I0.to_excel(writer, sheet_name="intensities_0")
            self.I1.to_excel(writer, sheet_name="intensities_1")
            self.V0.to_excel(writer, sheet_name="velocities_0")
            self.V1.to_excel(writer, sheet_name="velocities_1")
        print(f"CBA inputs saved to {fname_out}")
            

    # =====
    # Financial analysis
    # =====
    def financial_analysis(self):
        """Perform financial analysis"""
        raise NotImplementedError()
    

def check_year_order(dff, year_start, year_end, n_year):
    pass

def check_table_content(dff):
    """Check if values are not missing and are all numeric"""
    if dff.isna().sum().sum() != 0:
        raise ValueError(f"nan values found one of intensity/velocity tables")
    
    if not dff.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all():
        raise ValueError(f"non-numeric values in one of intensity/velocity tables")