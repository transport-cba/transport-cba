import pandas as pd
from numpy import arange, ones_like, where


available_countries = ["svk"]


class DataContainer(object):
    def __init__(self, country, price_level):
        """Read in all the CBA values necessary for economic analysis
        for a given country"""
        if country not in available_countries:
            raise ValueError("Data for country '%s' not available." % country)
        
        self.country = country
        self.dirn = "../files/%s/" % self.country
        self.pl = price_level
        
        self.df_raw = {}
        self.df_clean = {}
        self.TM_fin = {}
        self.TM_eco = {}


#    def __dir__(self):
#        return [attr for attr in dir(self) if not attr[:2] == "__"]


    def read_data(self, verbose=False):
        """Read in all the relevant data"""
        if verbose:
            print("Reading CBA data...")

        # macro data
        self.gdp_growth = pd.read_csv(self.dirn + "gdp_growth.csv", \
            index_col="year")
        self.cpi = pd.read_csv(self.dirn + "cpi.csv", index_col="year")

        # financial data
        self.df_raw["c_op"] = \
            pd.read_csv(self.dirn + "operation_cost.csv", index_col=0)
        self.df_raw["res_val"] = \
            pd.read_csv(self.dirn + "residual_value.csv", index_col=0)
        self.df_raw["c_fuel"] =\
            pd.read_csv(self.dirn + "fuel_cost.csv", index_col=0)

        # physical data
        self.fuel_rho = \
            pd.read_csv(self.dirn + "fuel_density.csv", index_col="fuel")

        # economic data
        self.df_raw["conv_fac"] =\
            pd.read_csv(self.dirn + "conversion_factors.csv", index_col=0)
        self.df_raw["occ_p"] =\
            pd.read_csv(self.dirn + "passenger_occupancy.csv", index_col=0)
        self.df_raw["occ_f"] =\
            pd.read_csv(self.dirn + "freight_occupancy.csv", index_col=0)
        self.df_raw["r_tp"] =\
            pd.read_csv(self.dirn + "trip_purpose.csv", index_col=0)
        self.df_raw["vtts"] =\
            pd.read_csv(self.dirn + "vtts.csv", index_col=0)
        self.df_raw["voc"] =\
            pd.read_csv(self.dirn + "voc.csv", index_col=0)
        self.df_raw["r_fuel"] =\
            pd.read_csv(self.dirn + "fuel_consumption.csv", index_col=0)
        self.df_raw["r_acc"] =\
            pd.read_csv(self.dirn + "accident_rate.csv", index_col=0)
        self.df_raw["c_acc"] =\
            pd.read_csv(self.dirn + "accident_cost.csv", index_col=0)
        self.df_raw["r_gg"] =\
            pd.read_csv(self.dirn + "greenhouse_rate.csv", index_col=0)
        self.df_raw["c_gg"] =\
            pd.read_csv(self.dirn + "greenhouse_cost.csv", index_col=0)
        self.df_raw["r_em"] =\
            pd.read_csv(self.dirn + "emission_rate.csv", index_col=0)
        self.df_raw["c_em"] =\
            pd.read_csv(self.dirn + "emission_cost.csv", index_col=0)
        self.df_raw["noise"] =\
            pd.read_csv(self.dirn + "noise.csv", index_col=0)


    def adjust_cpi(self, infl=0.02, yr_min=2000, yr_max=2050, verbose=False):
#    def adjust_cpi(self, infl=0.02, N_bw=20, N_fw=30, verbose=False):
        """Fill in mising values and compute cumulative inflation 
        to be able to adjust the price level"""
        if verbose:
            print("Adjusting CPI...")

        self.cpi = self.cpi.reindex(arange(yr_min, yr_max+1))
        self.cpi["cpi"].fillna(infl, inplace=True)

        # compute cumulative CPI
        self.cpi["cpi_index"] = ""
        self.cpi["cpi_index"] = \
            pd.to_numeric(self.cpi.cpi_index, errors="coerce")
        self.cpi.loc[self.pl, "cpi_index"] = 1.0
        ix = self.cpi.index.get_loc(self.pl)

        # backward
        for i in range(ix-1, -1, -1):
            self.cpi.iloc[i]["cpi_index"] = \
                self.cpi.iloc[i+1].cpi_index * (self.cpi.iloc[i].cpi + 1.0)
        
        # forward
        for i in range(ix+1, len(self.cpi)):
            self.cpi.iloc[i]["cpi_index"] = \
                self.cpi.iloc[i-1].cpi_index * (self.cpi.iloc[i-1].cpi + 1.0)


    def clean_data(self, verbose=False):
        """Remove unimportant columns and populate the df_clean dictionary"""
        for itm in self.df_raw.keys():
            if verbose:
                print("Cleaning: %s" % itm)
            self.df_clean[itm] = self.df_raw[itm].copy()
            if "nb" in self.df_clean[itm].columns:
                self.df_clean[itm].drop(columns=["nb"], inplace=True)
            if "unit" in self.df_clean[itm].columns:
                self.df_clean[itm].drop(columns=["unit"], inplace=True)
        
        for c in ["c_op", "vtts", "voc", "c_acc", "c_gg", "c_em", "noise"]:
            if "scale" in self.df_clean[c].columns:
                 self.df_clean[c]["value"] =\
                    self.df_clean[c].value * self.df_clean[c].scale
                 self.df_clean[c].drop(columns=["scale"], inplace=True)


    def adjust_price_level(self, verbose=False):
        """Unify the prices for one price level"""
        if verbose:
            print("Adjusting price level...")
        for c in ["c_op", "vtts", "voc", "c_fuel", "c_acc", "c_gg", "c_em",\
            "noise"]:
            if verbose:
                print("Adjusting: %s" % c)
            self.df_clean[c]["value"] = self.df_clean[c].value \
                * self.df_clean[c].price_level\
                .map(lambda x: self.cpi.loc[x].cpi_index)
            self.df_clean[c].drop(columns=["price_level"], inplace=True)
            self.df_clean[c]["value"] = self.df_clean[c].value.round(2)


    def wrangle_data(self, *args, **kwargs):
        self._wrangle_opex(*args, **kwargs)
        self._wrangle_vtts(*args, **kwargs)
        self._wrangle_fuel(*args, **kwargs)
        self._wrangle_accidents(*args, **kwargs)
        self._wrangle_greenhouse(*args, **kwargs)
        self._wrangle_emissions(*args, **kwargs)
        self._wrangle_noise(*args, **kwargs)


    def _wrangle_opex(self, verbose=False):
        """Set up index"""
        c = "c_op"
        self.df_clean[c].reset_index(inplace=True)
        self.df_clean[c].set_index(\
            ["item","operation_type","category"], inplace=True)

#        """Set index and add conversion factors"""
#        c = "c_op"
#        self.df_clean[c]["conv_fac"] = where(
#            self.df_clean[c].operation_type == "periodic", 
#            self.df_clean["conv_fac"].loc["periodic"]["aggregate"],
#            self.df_clean["conv_fac"].loc["routine"]["aggregate"])
#        self.df_clean[c]["value_eco"] = \
#            (self.df_clean[c].value * self.df_clean[c].conv_fac).round(2)
#
#        self.df_clean[c].rename(columns={"value": "value_fin"}, inplace=True)
#        
#        self.df_clean[c].reset_index(inplace=True)
#        self.df_clean[c].set_index(\
#            ["item","operation_type","category"], inplace=True)
#        self.df_clean[c].drop(columns="conv_fac", inplace=True)


    def _wrangle_vtts(self, verbose=False):
        """Average the value of the travel time saved"""
        if "distance" in self.df_clean["vtts"].columns:
            if verbose:
                print("Contracting distance.")
            gr = self.df_clean["vtts"]\
                .groupby(by=["vehicle","substance","purpose",\
                "gdp_growth_adjustment"])
            vtts = gr["value"].mean()
            vtts = vtts.reset_index()
        else:
            vtts = self.df_clean["vtts"].copy()

        # add trip purpose and merge
        r_tp = self.df_clean["r_tp"].reset_index().melt(id_vars="vehicle", \
            var_name="purpose", value_name="purpose_ratio")
        vtts = pd.merge(vtts, r_tp, how="left", on=["vehicle","purpose"])

        # add passenger occupancy
        self.df_clean["occ_p"]["substance"] = "passengers"
        self.df_clean["occ_p"].reset_index(inplace=True)
        
        vtts = pd.merge(vtts, \
            self.df_clean["occ_p"][["vehicle","value","substance"]],
            how="left", on=["vehicle","substance"], suffixes=("", "_occ"))

        # add freight occupancy
        self.df_clean["occ_f"]["substance"] = "freight"
        self.df_clean["occ_f"].reset_index(inplace=True)
        vtts = pd.merge(vtts, 
            self.df_clean["occ_f"][["vehicle","value","substance"]],
            how="left", on=["vehicle","substance"], suffixes=("", "_freight"))

        vtts["value_occ"] = vtts.value_occ.fillna(vtts.value_freight)
        vtts.drop(columns=["value_freight"], inplace=True)

        vtts.rename(columns={"value": "value_subst"}, inplace=True)
        vtts["value"] = vtts.value_subst * vtts.value_occ
        vtts.drop(columns=["value_subst","value_occ"], inplace=True)

        # contract by substance
        vtts = vtts.groupby(by=["vehicle","purpose",\
            "gdp_growth_adjustment","purpose_ratio"])\
            ["value"].sum().reset_index()

        # unify gdp growth adjustment by trip purpose ratio
        vtts["gdp_ga2"] = vtts.purpose_ratio * vtts.gdp_growth_adjustment
        vtts["value2"] = vtts.purpose_ratio * vtts.value

        vtts = vtts.groupby(["vehicle"])["gdp_ga2","value2"].sum()
        vtts.columns = ["gdp_growth_adjustment","value"]
        vtts["value"] = vtts.value.round(2)

        self.df_clean["vtts"] = vtts.copy()


    def _wrangle_fuel(self, verbose=True):
        """Convert units from eur/l to eur/kg"""
        # convert to kg/km and add conversion factors
        c = "c_fuel"
        self.df_clean[c]["value"] = \
            self.df_clean[c].value / self.fuel_rho.value
        self.df_clean[c] *= self.df_clean["conv_fac"].loc["factor", "fuel"]

        c = "r_fuel"
        self.df_clean[c] = pd.merge(self.df_clean[c].reset_index(), 
            self.fuel_rho.drop(columns=["unit"]), how="left", on="fuel")\
            .set_index(["vehicle","fuel"])
        
        # multiply polynomial coefficients by density
        for itm in ["a0", "a1", "a2", "a3"]:
            self.df_clean[c][itm] = \
                self.df_clean[c][itm] * self.df_clean[c].value
        self.df_clean[c].drop(columns=["value"], inplace=True)


    def _wrangle_accidents(self, verbose=False):
        """Unify the two datasets storing values for accidents"""
        self.df_clean["c_acc"]["value"] = self.df_clean["c_acc"].value *\
            self.df_clean["c_acc"].correct_unreported *\
            self.df_clean["c_acc"].correct_pass_per_acc
        self.df_clean["c_acc"].drop(columns=["correct_unreported",\
            "correct_pass_per_acc"], inplace=True)
        
        self.df_clean["r_acc"]["value"] = \
            self.df_clean["r_acc"]\
            [["fatal","severe_injury","light_injury","damage"]]\
            .values @ self.df_clean["c_acc"].value.values

        self.df_clean["r_acc"]["gdp_growth_adjustment"] = \
            self.df_clean["c_acc"].gdp_growth_adjustment.values[0]
        
        # copy to the cost dataframe
        self.df_clean["c_acc"] = self.df_clean["r_acc"]\
            [["lanes","label","environment","value","gdp_growth_adjustment"]].copy()

        self.df_clean["c_acc"].reset_index(inplace=True)
        self.df_clean["c_acc"].set_index(\
            ["road_type","lanes","label","environment"], inplace=True)


    def _wrangle_greenhouse(self, verbose=False):
        b = "r_gg"
        self.df_clean[b]["value"] = \
            self.df_clean[b].value * self.df_clean[b].factor

        gr = self.df_clean[b].groupby(["vehicle", "fuel"])["value"].sum()
        self.df_clean[b] = pd.DataFrame(gr).round(0)


    def _wrangle_emissions(self, verbose=False):
        b = "c_em"
        self.df_clean[b].reset_index(inplace=True)
        self.df_clean[b].set_index(["polluant", "environment"], \
            inplace=True)
        self.df_clean[b] = self.df_clean[b].sort_index()

        b = "r_em"
        self.df_clean[b].reset_index(inplace=True)
        self.df_clean[b].set_index(["polluant", "vehicle", "fuel"], \
            inplace=True)
        self.df_clean[b] = self.df_clean[b].sort_index()


    def _wrangle_noise(self, verbose=False):
        c = "noise"
        self.df_clean[c] = self.df_clean[c]\
            [self.df_clean[c].traffic_type == "thin"]
        self.df_clean[c].drop(columns=["traffic_type"], inplace=True)

        self.df_clean[c]["value2"] = self.df_clean[c].value\
            * self.df_clean[c].ratio
        gr = self.df_clean[c]\
            .groupby(["vehicle","environment","gdp_growth_adjustment"])

        self.df_clean[c] = gr["value2"].sum()
        self.df_clean[c] = self.df_clean[c].reset_index()
        self.df_clean[c].rename(columns={"value2": "value"}, inplace=True)
        self.df_clean[c].set_index(["vehicle","environment"], inplace=True)







