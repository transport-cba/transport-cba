import pandas as pd
import numpy as np
import yaml


class RoadCBA():
    vcat_cols = ["PV", "LGV", "HGV", "Bus"]

    def __init__(self, input_file):
        """Read in the input file in Yaml and save the variables"""
        data = yaml.loads(input_file)
        pass

    def read_project(self, df_prj):
        """Read the dataframe of each section of the road project"""
        pass

    def read_investment(self, df_fin):
        """Read financial input for the project, determining
        the composition of objects (road, tunnel)
        with the investment plan by years"""
        pass

    def read_velocities(self, df_vel):
        """Read the dataframe of velocities ordered by project section
        and vehicle category"""
        pass

    def read_intensities(self, csv_int):
        """Read the dataframe of intensities ordered by project section 
        and vehicle category"""
        df_int = read_csv(csv_int, index=road_sec)
        if df_int.columns != vcat_cols:
            raise ValueError("FILL")
            # raise error

    def fin_analysis(self):
        """Perform financial analysis"""
        pass

    def econ_analysis(self):
        """Perform economic analysis"""
        pass

    def _compute_vtts(self):
        pass

    def _compute_voc(self):
        pass

    def _compute_fuel(self):
        pass

    def _compute_acc(self):
        pass
    
    def _compute_greenhouse(self):
        pass

    def _compute_exhalates(self):
        pass

    def _compute_noise(self):
        pass


        







