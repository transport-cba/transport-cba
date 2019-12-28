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
        self.pl = price_level
        if self.yr_i != self.pl:
            print("Warning: start year not same as price level.")
        self.yr_const = None
        self.yr_op = None
        self.period = period
        self.yr_f = self.yr_i + self.period

        self.r_fin = fin_discount_factor
        self.r_eco = eco_discount_factor

        self.country = country
        self.currency = currency

        # define empty frames
        self.R = None
        self.C_inv = None
        self.C_eco = None
        self.O_inv = None
        self.O_eco = None
        self.I0 = None
        self.I1 = None
        self.V0 = None
        self.V1 = None

        # inherit
        super().__init__(self.country, self.yr_i)


    def prepare_parameters(self, verbose=False):
        """Read in and manipulate all the CBA parameters"""
        super().read_data(verbose=verbose)
        super().adjust_cpi(verbose=verbose)
        super().clean_data(verbose=verbose)
        super().adjust_price_level(verbose=verbose)
        super().wrangle_data(verbose=verbose)


    def read_project_inputs(self,
                            df_road_params,
                            df_capex,
                            df_int_0,
                            df_int_1,
                            verbose=False
                            ):
        if verbose:
            print("Reading project inputs...")
        """Read the dataframes
        * road parameters
        * capital investment (CAPEX)
        * intensities in 0th variant
        * intensities in 1st variant
        """
        self.R = df_road_params
        self.C_fin = df_capex
        self.I0 = df_int_0
        self.I1 = df_int_1
        # VERIFY INTEGRITY AND CONSISTENCY OF INPUTS

    
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


    def read_project_inputs_excel(self, file_xls, verbose=False):
        # VERIFY EXTENSION
        if verbose:
            print("Reading project inputs from xls/xlsx...")
        xls = pd.ExcelFile(file_xls)
        self.R = xls.parse("road_params", index_col=0)
        self.C_inv = xls.parse("capex_fin", index_col=0)
        self.I0 = xls.parse("intensities_0")
        self.I0.set_index(["id_section", "vehicle"], inplace=True)
        self.I1 = xls.parse("intensities_1")
        self.I1.set_index(["id_section", "vehicle"], inplace=True)
        # VERIFY INTEGRITY AND CONSISTENCY OF INPUTS


    def read_intensities(self, df_int_0, df_int_1):
        pass


    def read_intensities_csv(self, csv_int_0, csv_int_1):
        pass

    
    def read_intensities_excel(self, xls_int):
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
        if self.V0 or self.V1:
            print("Warning: velocities already defined, overwriting.")
        self.V0 = pd.DataFrame(columns=self.I0.columns, index=self.I0.index)
        self.V1 = pd.DataFrame(columns=self.I1.columns, index=self.I1.index)
        # WRITE THE FILLING FUNCTION

    
    def compute_opex(self):
        """Create a dataframe of operation costs (OPEX)"""
        pass


    def compute_residual_value(self):
        """Create a dataframe of residual values by each element"""
        pass


    def economic_analysis(self):
        """Perform economic analysis"""
        pass


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







