import pandas as pd
import numpy as np
from numpy.matlib import repmat
from .data_container import DataContainer


VEHICLE_TYPES = ["car", "lgv", "hgv", "bus"]
DAYS_YEAR = 365


class RoadCBA(DataContainer):
    def __init__(self,
                 init_year,
                 price_level,
                 country,
                 period=30,
                 fin_discount_factor=0.04,
                 eco_discount_factor=0.05,
                 currency="eur",
                 verbose=False
                 ):
        """
        Input
        -----
        - init_year: initial year of construction and economic analysis
        - price_level: 
        - country: country code
        - period: number of years for the economic analysis
        - fin_discount_factor: discount factor for financial analysis
        - eco_discount_factor: discount factor for economic analysis

        """
        self.yr_i = init_year
        self.N_yr = period
        self.yr_f = self.yr_i + self.N_yr - 1
        self.pl = price_level
        if self.yr_i != self.pl:
            print("Warning: start year not same as price level.")
        self.N_yr_bld = None
        self.N_yr_op = None
        self.yr_op = None
        self.yrs = np.arange(self.yr_i, self.yr_i + self.N_yr)

        self.r_fin = fin_discount_factor
        self.r_eco = eco_discount_factor

        self.country = country
        self.currency = currency

        self.veh_types = VEHICLE_TYPES

        # define empty frames
        self.R = None
        self.C_fin = None
        self.C_eco = None
        self.O_fin = None
        self.O_eco = None
        self.I0 = None
        self.I1 = None
        self.V0 = None
        self.V1 = None
        self.T0 = None
        self.T1 = None

        self.UC = {}
        self.B0 = {}
        self.B1 = {}
        self.NB = {}

        super().__init__(self.country, self.yr_i)


    def prepare_parameters(self, verbose=False):
        """Read in and manipulate all the CBA parameters"""
        super().read_data(verbose=verbose)
        super().adjust_cpi(yr_max=self.yr_f, verbose=verbose)
        super().clean_data(verbose=verbose)
        super().adjust_price_level(verbose=verbose)
        super().wrangle_data(verbose=verbose)


    def _assign_remaining_years(self):
        if self.C_fin is not None:
            self.yr_op = int(self.C_fin.columns[-1]) + 1
            self.N_yr_bld = len(self.C_fin.columns) - 1 # 1st col is total capex
            self.N_yr_op = self.N_yr - self.N_yr_bld
            self.yrs_op = \
                np.arange(self.yr_i + self.N_yr_bld, self.yr_i + self.N_yr)


    def read_project_inputs(self,
                            df_road_params,
                            df_capex,
                            df_int_0,
                            df_int_1,
                            verbose=False
                            ):
        """Read the dataframes
        * road parameters
        * capital investment (CAPEX)
        * intensities in 0th variant
        * intensities in 1st variant
        """
        if verbose:
            print("Reading project inputs...")
        self.R = df_road_params
        self.C_fin = df_capex
        self.I0 = df_int_0
        self.I1 = df_int_1
        # TODO: VERIFY INTEGRITY AND CONSISTENCY OF INPUTS

        # assign core variables
        self._assign_remaining_years()
        self.secs_0 = self.R[self.R.variant == 0].index
        self.secs_1 = self.R.index

    
    def read_project_inputs_csv(self,
                      file_road_params,
                      file_capex,
                      file_int_0,
                      file_int_1,
                      verbose=False
                      ):
        # TODO: VERIFY EXTENSIONS
        if verbose:
            print("Reading project inputs from csv...")
        self.R = pd.read_csv(file_road_params, index_col=0)
        self.C_fin = pd.read_csv(file_capex, index_col=0)
        self.I0 = pd.read_csv(file_int_0).reset_index()
        self_I0.set_index(["id_section", "vehicle"], inplace=True)
        self.I1 = pd.read_csv(file_int_1).reset_index()
        self.I1.reset_index().set_index(["id_section", "vehicle"], inplace=True)
        # TODO: VERIFY INTEGRITY AND CONSISTENCY OF INPUTS

        # assign core variables
        self._assign_remaining_years()
        self.secs_0 = self.R[self.R.variant == 0].index
        self.secs_1 = self.R.index


    def read_project_inputs_excel(self, file_xls, verbose=False):
        # TODO: VERIFY EXTENSION
        if verbose:
            print("Reading project inputs from xls/xlsx...")
        xls = pd.ExcelFile(file_xls)
        self.R = xls.parse("road_params", index_col=0)
        self.C_fin = xls.parse("capex_fin", index_col=0)
        self.I0 = xls.parse("intensities_0").reset_index()
        self.I0.set_index(["id_section", "vehicle"], inplace=True)
        self.I1 = xls.parse("intensities_1").reset_index()
        self.I1.set_index(["id_section", "vehicle"], inplace=True)
        # TODO: VERIFY INTEGRITY AND CONSISTENCY OF INPUTS

        # assign core variables
        self._assign_remaining_years()
        self.secs_0 = self.R[self.R.variant == 0].index
        self.secs_1 = self.R.index


    def read_intensities(self, df_int_0, df_int_1):
        pass


    def read_intensities_csv(self, csv_int_0, csv_int_1):
        pass

    
    def read_intensities_excel(self, xls_int):
        pass


    def fill_intensities(self):
        pass


    def read_velocities(self,
                        df_vel_0,
                        df_vel_1,
                        verbose=False
                        ):
        if verbose:
            print("Loading velocitiies...")
        self.V0 = df_vel_0
        self.V1 = df_vel_1


    def read_velocities_csv(self,
                            csv_vel_0, 
                            csv_vel_1,
                            verbose=False
                            ):
        """Read the dataframe of velocities ordered by project section
        and vehicle category"""
        if csv_vel_0[-3:] != "csv" or csv_vel_1[-3:] != "csv":
            print("One of files does not have required extension: csv.")

        if verbose:
            print("Reading velocities from csv...")
        self.V0 = pd.read_csv(csv_vel_0).reset_index()
        self.V0.set_index(["id_section", "vehicle"], inplace=True)
        self.V1 = pd.read_csv(csv_vel_1).reset_index()
        self.V1.set_index(["id_section", "vehicle"], inplace=True)


    def read_velocities_excel(self, xls_vel, verbose=False):
        """Read the dataframe of velocities from one excel file."""
        if not xls_vel.split(".")[-1] in ["xls", "xlsx"]:
            print("File does not have the required extension: xls, xlsx.")

        if verbose:
            print("Reading velocities xls/xlsx...")
        self.V0 = pd.read_excel(xls_vel, sheet_name="velocities_0")
        self.V0.set_index(["id_section", "vehicle"], inplace=True)
        self.V1 = pd.read_excel(xls_vel, sheet_name="velocities_1")
        self.V1.set_index(["id_section", "vehicle"], inplace=True)


    def fill_velocities(self):
        """Create the velocity matrices according to pre-defined rules"""
        # TODO: WRITE THE FILLING FUNCTIONS
        if self.V0 or self.V1:
            print("Warning: velocities already defined, overwriting.")
        self.V0 = pd.DataFrame(columns=self.I0.columns, index=self.I0.index)
        self.V1 = pd.DataFrame(columns=self.I1.columns, index=self.I1.index)

    
    def compute_opex(self):
        """Create a dataframe of operation costs (OPEX).
        Only new road sections are considered."""
        c = "c_op"
        self._create_time_opex_mat()  # defined UC[c]
        self._create_time_opex_mask() # defined O_mask

        # compute pavement area
        self.R["area"] = self.R.length * 1e3 * self.R.width # CHECK UNITS
        
        self.UO = self.UC[c] * self.O_mask
        
        ids_new = self.R[self.R.variant == 1].index.values
        dfs = {}
        for rid in ids_new:
            dfs[rid] = \
                pd.DataFrame(columns=self.yrs_op, index=["routine", "periodic"])
            dfs[rid].index.name = "operation_type"
            rcat = self.R.loc[rid, "category"]
            tmp = self.UO.reset_index(["operation_type", "category"])
            tmp = tmp[tmp.category == rcat]

            dfs[rid].loc["routine"] = tmp[tmp.operation_type == "routine"]\
                .drop(columns=["operation_type", "category"]).sum(axis=0) *\
                self.R.loc[rid, "area"]

            dfs[rid].loc["periodic"] = tmp[tmp.operation_type == "periodic"]\
                .drop(columns=["operation_type", "category"]).sum(axis=0) *\
                self.R.loc[rid, "area"]

        self.O_fin = pd.concat(dfs.values(), keys=dfs.keys())
        self.O_fin.index.set_names(["id_section", "operation_type"], inplace=True)

    
    def _create_time_opex_mat(self, verbose=True):
        if verbose:
            print("Creating time matrix for OPEX...")
        c = "c_op"

        self.UC[c] = \
            pd.DataFrame(columns=self.yrs, index=self.df_clean[c].index)

        self.UC[c][self.yr_i] = self.df_clean[c].value
        # FIX ERROR IN CPI INDEXING
        for yr in self.yrs[1:]:
            self.UC[c][yr] = \
                self.UC[c][self.yr_i] * self.cpi.loc[yr, "cpi_index"]
        
        self.UC[c] = self.UC[c].round(2)
        self.UC[c] = self.UC[c][self.yrs_op] # choose operation years


    def _create_time_opex_mask(self):
        """Compose a time matrix of zeros and ones indicating if the
        maintanance has to be performed in a given year."""
        c = "c_op"
        self.O_mask = \
            pd.DataFrame(0, index=self.df_clean[c].index, columns=self.yrs_op)
        
        for itm in self.O_mask.index:
            p = self.df_clean[c].loc[itm, "periodicity"].astype(int)
            if p == 1:
                self.O_mask.loc[itm] = 1
            else:
                v = np.zeros_like(self.yrs_op).astype(int)
                for i, _ in enumerate(v):
                    if (i+1) % p == 0:
                        v[i] = 1
                self.O_mask.loc[itm] = v


    def compute_residual_value(self):
        """Create a dataframe of residual values by each element"""
        pass


    def economic_analysis(self):
        """Perform economic analysis"""
        pass


    def _create_time_benefit_mat(self, verbose=False):
        """Define the time-cost matrices for each benefit"""
        if verbose:
            print("Creating time matrices for benefits...")

        for b in ["vtts", "voc", "c_acc", "c_em", "noise"]:
            if verbose:
                print("Creating: %s" % b)
            self.UC[b] = \
                pd.DataFrame(columns=self.yrs, index=self.df_clean[b].index)
            self.UC[b][self.yr_i] = self.df_clean[b].value
            for yr in self.yrs[1:]:
                self.UC[b][yr] = \
                    self.UC[b][self.yr_i] * self.cpi.loc[yr, "cpi_index"]
                if "gdp_growth_adjustment" in self.df_clean[b].columns:
                    self.UC[b][yr] = self.UC[b][yr] \
                    * (1.0 + self.gdp_growth.loc[yr].gdp_growth \
                    * self.df_clean[b].gdp_growth_adjustment)

            self.UC[b] = self.UC[b].round(2)

        b = "gg"
        # TODO: CREATE GREENHOUSE TIME MATRIX


    def _compute_travel_time_matrix(self):
        """Compute travel time by road section and vehicle type"""
        # 0th variant
        tmp = {}
        for ii in self.secs_0:
            tmp[ii] = self.R.loc[ii, "length"] / self.V0.loc[ii]
            
        self.T0 = pd.concat(tmp.values(), keys=tmp.keys())
        self.T0.sort_index(inplace=True)
        self.T0.index.names = self.V0.index.names

        # 1st variant
        tmp = {}
        for ii in self.secs_1:
            tmp[ii] = self.R.loc[ii, "length"] / self.V1.loc[ii]
        
        self.T1 = pd.concat(tmp.values(), keys=tmp.keys())
        self.T1.sort_index(inplace=True)
        self.T1.index.names = self.V0.index.names


    def _compute_vtts(self):
        """Mask is given by the intensities, as these are zero
        in the construction years"""
        b = "vtts"
        # adjust VTTS unit cost matrix
        # TODO: UNIFY UC0 AND UC1
        UC0 = pd.DataFrame(repmat(self.UC[b], len(self.secs_0), 1), \
            columns=self.V0.columns, index=self.V0.index)
        UC1 = pd.DataFrame(repmat(self.UC[b], len(self.secs_1), 1), \
            columns=self.V1.columns, index=self.V1.index)

        # matrix of benefits
        self.B0[b] = self.T0 * self.I0 * UC0 * DAYS_YEAR
        self.B1[b] = self.T1 * self.I1 * UC1 * DAYS_YEAR
        self.NB[b] = self.B0[b].sum(0) - self.B1[b].sum(0)


    def _compute_voc(self):
        b = "voc"
        # create length matrix
        L = pd.DataFrame(repmat(self.R.loc[self.secs_1].length, \
            len(self.yrs), len(self.veh_types)).T,
            index=self.V1.swaplevel().sort_index().index, \
            columns=self.V1.columns)
        L = L.swaplevel().sort_index()

        # create unit cost matrix
        UC0 = pd.DataFrame(repmat(self.UC[b], len(self.secs_0), 1), \
            columns=self.V0.columns, index=self.V0.index)
        UC1 = pd.DataFrame(repmat(self.UC[b], len(self.secs_1), 1), \
            columns=self.V1.columns, index=self.V1.index)

        self.B0[b] = UC0 * self.I0 * L.loc[self.secs_0] * DAYS_YEAR
        self.B1[b] = UC1 * self.I1 * L.loc[self.secs_1] * DAYS_YEAR
        self.NB[b] = self.B1[b].sum(0) - self.B0[b].sum(0)


    def _compute_fuel(self):
        pass


    def _compute_accidents(self):
        pass
    

    def _compute_greenhouse(self):
        pass


    def _compute_exhalates(self):
        pass


    def _compute_noise(self):
        pass


    def financial_analysis(self):
        """Perform financial analysis"""
        pass


    def _compute_econ_capex(self):
        """Apply conversion factors to compute
        CAPEX for the economic analysis."""
        pass







