#!/usr/bin/env python3
import pycba
from pycba import RoadCBA
from pycba.sample_projects import load_sample_bypass

import numpy as np
import pandas as pd
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")

print(f"Source for testing: {pycba.__file__}")

b = load_sample_bypass()

cba = RoadCBA(2020, "svk", verbose=True)
cba.read_project_inputs(*b.values())

# test saving inputs
cba.save_inputs_to_excel()

# compute cba
res = cba.economic_analysis()
print(res)

# save outputs
cba.save_results_to_excel()
