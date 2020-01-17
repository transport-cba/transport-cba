# pycba
A Python module for cost-benefit analysis of infrastructure projects


This module provides a consistent way to evaluate economic efficiency
of a road section with well-defined inputs and parameters.
It offers a significantly wider options for analysis of alternatives
than Excel.


## Dependencies
Numpy, Pandas


## Project inputs
Inputs contain the following:
* capital expenditures (CAPEX)
* old and new parameters of road sections (length, width, number of lanes etc)
* intensities in variant 0 and 1
* velocities in variant 0 and 1

Several options to load inputs:
* directly as a Pandas dataframe
* from csv's
* from an Excel file


