#!/usr/bin/env python
"""
Read in the excel and transform each sheet into a csv
in the production folder.

18/12/19
"""
import pandas as pd

country_code = "svk/"
dirn = "../files/" + country_code

xls = pd.ExcelFile(dirn + "svk_cba_road_params_201912.xlsx")
print("Sheet names:", xls.sheet_names)

for name in xls.sheet_names:
    df = xls.parse(name)
    print("Saving %s..." % name)
    df.to_csv(dirn + name + ".csv", index=False)
