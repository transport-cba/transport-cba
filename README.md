# transport-cba

A Python module for cost-benefit analysis of infrastructure projects.

Provides a consistent way to evaluate economic efficiency
of road projects with well-defined inputs and parameters.

Main benefits compared to traiditional Excel-based approach:
- several orders of magnitude faster and cheaper
- wider options for analysis of alternative scenarios
- significantly lower margin for error


## Installation
From pip:
```
pip install transport-cba
```
Or directly from git:
```
pip install git+https://github.com/transport-cba/transport-cba.git
```

## Inputs
Load project inputs as an Excel file with following sheet names:
  `road_params, capex, intensities_0, intensities_1, velocities_0, velocities_1`

Meaning of required inputs:
* capital expenditures (CAPEX) with pre-defined items
* parameters of road sections (length, width, number of lanes etc)
* vehicle intensities in variant 0 and 1 (without and with the project) by road segment
* vehicle velocities in variant 0 and 1 by segment

For illustration, please download the sample input (see below).


## Outputs
* Dataframe of costs and benefits
* Economic indicators:
  - economic net present value (ENPV)
  - economic internal rate of return (ERR)
  - benefit to cost ratio (BCR)
  - dataframes with breakdown of relevant benefits by years


## Example
NB: Values might differ slightly.

```python
>>> from transport_cba import RoadCBA
>>> from transport_cba.sample_projects import load_sample_bypass

>>> bypass = load_sample_bypass()

>>> cba = RoadCBA(2020, "svk")
>>> cba.read_project_inputs(
...     bypass["road_params"],
...     bypass["capex"],
...     bypass["intensities_0"],
...     bypass["intensities_1"],
...     bypass["velocities_0"],
...     bypass["velocities_1"]
... )
>>> cba.economic_analysis()
>>> cba.economic_indicators
```
|    | Quantity   | Unit   |   Value |
|---:|:-----------|:-------|--------:|
|  0 | ENPV       | M EUR  |   3.336 |
|  1 | ERR        | %      |   5.62  |
|  2 | BCR        |        |   1.076 |
```
