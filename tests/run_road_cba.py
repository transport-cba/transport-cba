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

b = load_sample_bypass()

cba = RoadCBA(2020, "svk", verbose=True)
cba.read_project_inputs(*b.values())

# test saving inputs
cba.save_inputs_to_excel()

# compute cba
res = cba.economic_analysis()
print(res)
res.to_csv("economic_result.csv", index=False)
