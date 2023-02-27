from .roadcba import RoadCBA

import pandas as pd
import numpy as np

class UpdatedFuelCurveRoadCBA(RoadCBA):
    """
    Implementation of economic evaluation using the methods and parameters
    defined by Slovak Guidelines for CBA appraisal of transport
    projects v3.0 except fuel consumption.

    Fuel consumption assumptions updated to lagged WebTAG curves.
    """
    def __init__(self,
                 init_year,
                 evaluation_period,
                 price_level,
                 fin_discount_rate,
                 eco_discount_rate,
                 currency,
                 verbose=False,
                 include_freight_time=False,
                 cpi_source="newest",
                 gdp_source="newest"):
        """
        Input
        ------
        init_year: int
            The first year of economic analysis

        evaluation_period: int
            Duration of evaluation period in years.

        price_level: int
            The year in relation to which prices are set.

        fin_discount_rate: float
            Discount rate for financial analysis.

        eco_discount_rate: float
            Discount rate for economic analysis.

        currency: str
            Currency code.

        verbose: bool, optional
            Flag to control verbosity.

        consider_freight_time: bool, optional
            Flag to control if freight time savings
            are included in the appraisal.               .

        Returns
        ------
        object
            pycba.roads.svk.OPIIv3p0.RoadCBA object.
        """

        super().__init__(init_year=init_year,
                         evaluation_period=evaluation_period,
                         price_level=price_level,
                         fin_discount_rate=fin_discount_rate,
                         eco_discount_rate=eco_discount_rate,
                         currency=currency,
                         verbose=verbose,
                         include_freight_time=include_freight_time,
                         cpi_source=cpi_source,
                         gdp_source=gdp_source)

        # update parameter file to match updated fuel curves file
        self.paramfile = self.paramdir + \
                         "svk_road_cba_parameters_OPIIv3p0_2022_updated_fuel_curves.xlsx"

    def _wrangle_fuel(self):
        """
        Result:
            self.params_clean['r_fuel'] contains values indexed by vehicle
            and fuel type.
        """

        # set index on fuel ratio
        c = "r_fuel"
        self.params_clean[c].reset_index(inplace=True, drop=True)
        self.params_clean[c].set_index(["vehicle", "fuel"], inplace=True)

        c = 'fuel_rho'
        self.params_clean[c].set_index('fuel', inplace=True)

        # convert from eur/l to eur/kg
        c = 'c_fuel'
        self.params_clean[c].set_index('fuel', inplace=True)

        self.params_clean[c]["value"] = \
            self.params_clean[c]['value']\
            / self.params_clean['fuel_rho']['value']

        # convert parameters to output polyfit in kg/km
        c = "fuel_coeffs"
        self.params_clean[c] = pd.merge(self.params_clean[c]\
                                                .reset_index(drop=True),
                                        self.params_clean['fuel_rho']\
                                                .rename(columns={
                                                    'value': 'density'}),
                                        how="left",
                                        on="fuel")\
                                    .set_index(["vehicle", "fuel"])

        # multiply polynomial coefficients by density
        # change w.r.t. RoadCBA
        for col in ["a-1", "a0", "a1", "a2"]:
            self.params_clean[c][col] = \
                self.params_clean[c][col] * self.params_clean[c]['density']
        self.params_clean[c].drop(columns=["density"], inplace=True)

        # multiply by density
        c = "fuel_acc"
        self.params_clean[c].set_index(['vehicle', 'fuel'], inplace=True)
        self.params_clean[c] = self.params_clean[c].stack()\
                                                   .to_frame().rename(
                                                        columns={0: 'value'})
        self.params_clean[c] = self.params_clean[c]\
                                    * self.params_clean['fuel_rho']
        self.params_clean[c] = self.params_clean[c].unstack(2)
        # simplify columns index
        self.params_clean[c].columns = self.params_clean[c]\
                                                .columns.get_level_values(1)

    def _compute_fuel_consumption(self):
        """
        Compute the consumption in kg per vehicle by section, vehicle
        and fuel type.
        """
        if self.verbose:
            print("    Computing fuel consumption...")
        assert self.L0 is not None, "Compute length matrix first."
        assert self.L1 is not None, "Compute length matrix first."

        ###
        # velocity-dependent part
        ###

        # get a matrix of ones per vehicle, fuel type and year
        helper_ones = self.RF.copy() / self.RF.copy()

        # velocity by vehicle, fuel type, section and year
        # assumes vehicles with different fuel move at the same speed
        V0s = helper_ones * self.V0
        V0s = V0s.sort_index()

        V1s = helper_ones * self.V1
        V1s = V1s.sort_index()

        # quantity of fuel consumed per vehicle, fuel type and section
        self.QF0 = pd.DataFrame(0, columns=V0s.columns, index=V0s.index)
        self.QF1 = pd.DataFrame(0, columns=V1s.columns, index=V1s.index)

        # update w.r.t. RoadCBA
        for (veh, f), cs in self.params_clean['fuel_coeffs'].iterrows():
            # consumption-velocity curve coefficients
            c = cs.values

            # variant 0
            vs = V0s.loc[(veh, f)]
            # update w.r.t. RoadCBA
            qf = np.polynomial.polynomial.polyval(vs, c, tensor=False)/vs
            self.QF0.loc[(veh, f)] = qf.values

            # variant 1
            vs = V1s.loc[(veh, f)]
            # update w.r.t. RoadCBA
            qf = np.polynomial.polynomial.polyval(vs, c, tensor=False)/vs
            self.QF1.loc[(veh, f)] = qf.values

        # update w.r.t. RoadCBA
        # fix division by zero: zero velocity -> zero consumption
        self.QF0 = self.QF0.replace(np.inf, 0)
        self.QF1 = self.QF1.replace(np.inf, 0)

        # velocity part
        self.QFv0 = self.QF0 * self.L0
        self.QFv1 = self.QF1 * self.L1

        ##
        # acceleration-dependent part
        ##

        # self.RP = self.RP.reset_index().set_index('id_road_section')

        # time matrix of acceleration ratios - variant 0, 1
        acceleration_mat0 = self.RP.loc[self.RP['variant'] == 0,
                                        self.ACCELERATION_COLUMNS]\
                                    .stack().to_frame()
        acceleration_mat1 = self.RP.loc[self.RP['variant'] == 1,
                                        self.ACCELERATION_COLUMNS] \
                                    .stack().to_frame()

        # # reindex to the original columns
        # self.RP = self.RP.reset_index()\
        #                  .set_index(['id_road_section', 'variant'])

        acceleration_mat0.columns = ['ratio']
        acceleration_mat0.index.names = ['id_road_section', 'acceleration']
        acceleration_mat1.columns = ['ratio']
        acceleration_mat1.index.names = ['id_road_section', 'acceleration']

        acceleration_mat0 = pd.DataFrame(np.outer(acceleration_mat0['ratio'],
                                        np.ones_like(self.yrs)),
                                        columns=self.yrs,
                                        index=acceleration_mat0.index)

        acceleration_mat1 = pd.DataFrame(np.outer(acceleration_mat1['ratio'],
                                         np.ones_like(self.yrs)),
                                         columns=self.yrs,
                                         index=acceleration_mat1.index)

        # time-matrix of fuel consumption
        fuel_acc_mat = self.params_clean['fuel_acc'].stack().to_frame()
        fuel_acc_mat.columns = ['value']
        fuel_acc_mat.index.names = ['vehicle', 'fuel', 'acceleration']

        fuel_acc_mat = pd.DataFrame(np.outer(fuel_acc_mat['value'],
                                             np.ones_like(self.yrs)),
                                    columns=self.yrs,
                                    index=fuel_acc_mat.index)

        # ones in the index and columns structure of intensity dataframes
        ones0 = self.I0/self.I0
        ones1 = self.I1/self.I1

        QFa0 = ((helper_ones * ones0) * acceleration_mat0 * fuel_acc_mat)
        QFa1 = ((helper_ones * ones1) * acceleration_mat1 * fuel_acc_mat)

        # acceleration dependent part
        self.QFa0 = QFa0.reset_index()\
                        .groupby(['vehicle', 'fuel', 'id_road_section'])[self.yrs]\
                        .sum()
        self.QFa1 = QFa1.reset_index() \
                        .groupby(['vehicle', 'fuel', 'id_road_section'])[self.yrs]\
                        .sum()

        self.QF0 = self.QFv0 + self.QFa0
        self.QF1 = self.QFv1 + self.QFa1