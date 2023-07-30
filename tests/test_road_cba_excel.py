#!/usr/bin/env python
import transport_cba
from transport_cba import RoadCBA
from transport_cba.sample_projects import load_sample_bypass
import os

import numpy as np
import pandas as pd

print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"transport_cba version: {transport_cba.__version__}")
print(f"Source for testing: {transport_cba.__file__}")

def test_whole_cba_from_excel():
    DIRN_BASE = os.path.dirname(os.path.realpath(__file__))
    
    input_data = f"{DIRN_BASE}/../use_cases/cba_inputs_i63_kutniky.xlsx"
    input_data = f"{DIRN_BASE}/../use_cases/cba_inputs_r2_soroska.xlsx"
    input_data = f"{DIRN_BASE}/../use_cases/cba_inputs_d1_hp_ll_ds.xlsx" # start with year 2016
    
    cba = RoadCBA(2019, "svk", verbose=True)
    cba.read_project_inputs_excel(input_data)
    res = cba.economic_analysis()
    print(res)
    print("Net benefits:")
    print(pd.DataFrame(cba.df_eco).T)

    assert res["Value"].isna().sum() == 0
