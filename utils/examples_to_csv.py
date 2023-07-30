#!/usr/bin/env python
"""
Read in the excel and transform each sheet into a csv
in the production folder.

18/12/19
"""
import pandas as pd


dirn = "../transport_cba/examples/"

fname = dirn + "example_bypass.xlsx"
xls = pd.ExcelFile(fname)
print("File name: %s" % fname)
print("Sheet names:", xls.sheet_names)


for name in xls.sheet_names:
    df = xls.parse(name)
    print("Saving %s..." % name)
    df.to_csv(dirn + "bypass_" + name + ".csv", index=False)
