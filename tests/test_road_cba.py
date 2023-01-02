#!/usr/bin/env python
from pycba import RoadCBA
from pycba.sample_projects import load_sample_bypass

b = load_sample_bypass()

cba = RoadCBA(2020, 2020, "svk", verbose=True)
cba.read_project_inputs(*b.values())
cba.economic_analysis()
