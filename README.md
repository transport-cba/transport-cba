# pycba
A Python module for cost-benefit analysis of infrastructure projects


This module provides a consistent way to evaluate economic efficiency
of a road project with well-defined inputs and parameters.
It offers a significantly wider options for analysis of alternatives
than Excel.


## Inputs
Required project inputs:
* capital expenditures (CAPEX) with pre-defined items
* parameters of old and new road sections (length, width, number of lanes etc)
* intensities in variant 0 and 1 (without and with the project)
* velocities in variant 0 and 1

Options to load project inputs:
* separately as Pandas dataframes
* from an Excel file with sheet names:
  `road_params, capex, intensities_0, intensities_1, velocities_0, velocities_1`


## Outputs
* Dataframe of costs and benefits
* Economic indicators:
  - net present value (ENPV)
  - internal rate of return (ERR)
  - benefit to cost ratio (BCR)


## Example
Values might differ slightly.

```python
>>> from pycba import RoadCBA
>>> from pycba.sample_projects import load_sample_bypass

>>> b = load_sample_bypass()

>>> cba = RoadCBA(2020, 2020, "svk")
>>> cba.read_project_inputs(b["RP"], b["C_fin"], b["I0"], b["I1"], b["V0"], b["V1"])
>>> cba.economic_analysis()
>>> res = rcba.economic_indicators()
>>> res
```
|    |     Value | Unit   |
|---:|----------:|:-------|
|  0 | 3.33583   | M EUR  |
|  1 | 0.0562048 | %      |
|  2 | 1.07576   | nan    |



