# pycba
A Python module for cost-benefit analysis of infrastructure projects

In development.

This module provides a consistent way to evaluate economic efficiency
of a road project with well-defined inputs and parameters.
It offers a significantly wider options for analysis of alternatives
than Excel.

## Dependencies
numpy, pandas, numpy_financial

## Contributors
[Peter Vanya](https://github.com/petervanya), Inovec Technology

## Inputs
Required project inputs:
* capital expenditures (CAPEX) with pre-defined items
* parameters of old and new road sections (length, width, number of lanes etc)
  and toll sections
* intensities in variant 0 and 1 (without and with the project)
* velocities in variant 0 and 1
* expected accident rate in variant 0 and 1

Options to load project inputs:
* separately as Pandas dataframes

## Outputs
* Dataframe of costs and benefits
* Economic indicators:
  - net present value (ENPV)
  - internal rate of return (ERR)
  - benefit to cost ratio (BCR)

## Example

```python
from pycba.roads.svk.OPIIv3p0 import RoadCBA
from pycba.sample_projects import load_lietavska_lucka

dict_LL = load_lietavska_lucka()

cba_LL = RoadCBA(2022, 30, 2021, 0.04, 0.05, 'eur', verbose=False)
cba_LL.read_project_inputs(df_rp=dict_LL['RP'], 
                           df_capex=dict_LL['C_fin'],
                           df_int_0=dict_LL['I0'], 
                           df_int_1=dict_LL['I1'],
                           df_vel_0=dict_LL['V0'], 
                           df_vel_1=dict_LL['V1'])
cba_LL.read_custom_accident_rates(dict_LL['acc'])
cba_LL.read_toll_section_types(dict_LL['TP'])

cba_LL.read_parameters()

cba_LL.perform_economic_analysis()

cba_LL.print_economic_indicators()
```

Output (values might differ):
```
ENPV: 55.69 M EUR
ERR : 12.27 %
BCR : 2.21
```



