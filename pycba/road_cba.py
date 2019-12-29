import pandas as pd
import numpy as np
from .data_container import DataContainer


veh_cats = ["car", "lgv", "hgv", "bus"]


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

        self.TM = {}

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
        # VERIFY INTEGRITY AND CONSISTENCY OF INPUTS

        self._assign_remaining_years()

    
    def read_project_inputs_csv(self,
                      file_road_params,
                      file_capex,
                      file_int_0,
                      file_int_1,
                      verbose=False
                      ):
        # VERIFY EXTENSIONS
        if verbose:
            print("Reading project inputs from csv...")
        self.R = pd.read_csv(file_road_params, index_col=0)
        self.C_fin = pd.read_csv(file_capex, index_col=0)
        self.I0 = pd.read_csv(file_int_0)
        self.I0.set_index(["id_section", "vehicle"], inplace=True)
        self.I1 = pd.read_csv(file_int_1)
        self.I1.set_index(["id_section", "vehicle"], inplace=True)
        # VERIFY INTEGRITY AND CONSISTENCY OF INPUTS

        self._assign_remaining_years()


    def read_project_inputs_excel(self, file_xls, verbose=False):
        # VERIFY EXTENSION
        if verbose:
            print("Reading project inputs from xls/xlsx...")
        xls = pd.ExcelFile(file_xls)
        self.R = xls.parse("road_params", index_col=0)
        self.C_fin = xls.parse("capex_fin", index_col=0)
        self.I0 = xls.parse("intensities_0")
        self.I0.set_index(["id_section", "vehicle"], inplace=True)
        self.I1 = xls.parse("intensities_1")
        self.I1.set_index(["id_section", "vehicle"], inplace=True)
        # VERIFY INTEGRITY AND CONSISTENCY OF INPUTS

        self._assign_remaining_years()


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
        self.V0 = pd.read_csv(csv_vel_0)
        self.V0.set_index(["id_section", "vehicle"], inplace=True)
        self.V1 = pd.read_csv(csv_vel_1)
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
        if self.V0 or self.V1:
            print("Warning: velocities already defined, overwriting.")
        self.V0 = pd.DataFrame(columns=self.I0.columns, index=self.I0.index)
        self.V1 = pd.DataFrame(columns=self.I1.columns, index=self.I1.index)
        # WRITE THE FILLING FUNCTION

    
    def compute_opex(self):
        """Create a dataframe of operation costs (OPEX).
        Only new road sections are considered."""
        c = "c_op"
        self._create_time_opex_mat()  # defined TM[c]
        self._create_time_opex_mask() # defined O_mask

        # compute pavement area
        self.R["area"] = self.R.length * 1e3 * self.R.width # CHECK UNITS
        
        self.UO = self.TM[c] * self.O_mask
        
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

        self.TM[c] = \
            pd.DataFrame(columns=self.yrs_op, index=self.df_clean[c].index)

        self.TM[c][self.yr_op] = self.df_clean[c].value
        for yr in self.yrs_op[1:]:
            self.TM[c][yr] = \
                self.TM[c][self.yr_op] * self.cpi.loc[yr, "cpi_index"]
        
        self.TM[c] = self.TM[c].round(2)

#        # financial
#        self.TM_fin[c] = \
#            pd.DataFrame(columns=self.yrs_op, index=self.df_clean[c].index)
#
#        self.TM_fin[c][self.yr_op] = self.df_clean[c].value_fin
#        for yr in self.yrs_op[1:]:
#            self.TM_fin[c][yr] = \
#                self.TM_fin[c][self.yr_op] * self.cpi.loc[yr].cpi_index
#
#        # economic
#        self.TM_eco[c] = \
#            pd.DataFrame(columns=self.yrs_op, index=self.df_clean[c].index)
#
#        self.TM_eco[c][self.yr_op] = self.df_clean[c].value_eco
#        for yr in self.yrs_op[1:]:
#            self.TM_eco[c][yr] = \
#                self.TM_eco[c][self.yr_op] * self.cpi.loc[yr].cpi_index
#
#        self.TM_fin[c] = self.TM_fin[c].round(2)
#        self.TM_eco[c] = self.TM_eco[c].round(2)
#        # ALTERNATIVE: can do by np.outer


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


    def _create_time_benefit_mat(self, verbose=True):
        """Define the time-cost matrices for each benefit"""
        if verbose:
            print("Creating time matrices for benefits...")

        for b in ["vtts", "voc", "c_acc", "c_em", "noise"]:
            if verbose:
                print("Creating: %s" % b)
            self.TM[b] = \
                pd.DataFrame(columns=self.yrs_op, index=self.df_clean[b].index)
            self.TM[b][y_start] = self.df_clean[b].value
            for yr in self.yrs_op[1:]:
                self.TM[b][yr] = \
                    self.TM[b][self.yrs_op] * self.cpi.loc[yr, "cpi_index"]
                if "gdp_growth_adjustment" in self.df_clean[b].columns:
                    self.TM[b][yr] = self.TM[b][yr] \
                    * (1.0 + self.gdp_growth.loc[yr].gdp_growth \
                    * self.df_clean[b].gdp_growth_adjustment)

        b = "gg"
        # CREATE GREENHOUSE TIME MATRIX


    def _compute_vtts(self):
        pass


    def _compute_voc(self):
        pass


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







