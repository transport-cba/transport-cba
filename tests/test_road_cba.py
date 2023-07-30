#!/usr/bin/env python3
import transport_cba
from transport_cba import RoadCBA
from transport_cba.sample_projects import load_sample_bypass

import numpy as np
import pandas as pd

print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"transport_cba version: {transport_cba.__version__}")
print(f"Source for testing: {transport_cba.__file__}")


def test_whole_cba_pipeline():
    """Test whole pipeline starting from loaded bypass"""
    b = load_sample_bypass()
    
    cba = RoadCBA(2020, "svk", verbose=True)
    cba.read_project_inputs(*b.values())
    
    # compute cba
    res = cba.economic_analysis()
    print(res)
    
    # save outputs
    cba.save_results_to_excel()

    assert res.shape == (3, 3) and res["Value"].isna().sum() == 0

def test_real_cba_excels():
    pass

def test_input_saving():
    b = load_sample_bypass()
    
    cba = RoadCBA(2020, "svk", verbose=True)
    cba.read_project_inputs(*b.values())
    
    # test saving inputs
    cba.save_inputs_to_excel()
    assert True

def test_sheet_names():
    pass

def test_consistency_year_ranges():
    pass

def test_road_param_cols():
    pass

def test_capex_cols():
    pass
