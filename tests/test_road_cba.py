#!/usr/bin/env python
from pycba import RoadCBA
from pycba.sample_projects import load_sample_bypass

import numpy as np
import pandas as pd
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")

b = load_sample_bypass()

cba = RoadCBA(2020, 2020, "svk", verbose=True)
cba.read_project_inputs(*b.values())
res = cba.economic_analysis()
print(res)
