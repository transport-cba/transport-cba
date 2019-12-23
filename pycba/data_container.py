import pandas as pd
from numpy import arange, ones_like


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
        self.read_data()


    def __str__(self):
        """Print the list of stored dataframes"""
        return "\n".join([s for s in self.__dir__() if s[:2] == "df"])


    def read_data(self):
        """Read in all the relevant data"""
        # macro data
        self.gdp_growth = pd.read_csv(self.dirn + "gdp_growth.csv", \
            index_col="year")
        self.cpi = pd.read_csv(self.dirn + "cpi.csv", index_col="year")
        #self.fuel_rho = pd.read_csv(self.dirn + "fuel_density.csv", index_col="fuel")

        # financial data
        self.df_raw["op_cost"] =\
            pd.read_csv(self.dirn + "operation_cost.csv", index_col=0)
        self.df_raw["res_val"] =\
            pd.read_csv(self.dirn + "residual_value.csv", index_col=0)

        # economic data
        self.df_raw["conv_fac"] =\
            pd.read_csv(self.dirn + "conversion_factors.csv", index_col=0)
        self.df_raw["p_occ"] = pd.read_csv(self.dirn + "passenger_occupancy.csv", index_col=0)
        self.df_raw["f_occ"] = pd.read_csv(self.dirn + "freight_occupancy.csv", index_col=0)
        self.df_raw["tp"] = pd.read_csv(self.dirn + "trip_purpose.csv", index_col=0)
        self.df_raw["vtts"] = pd.read_csv(self.dirn + "vtts.csv", index_col=0)
        self.df_raw["voc"] = pd.read_csv(self.dirn + "voc.csv", index_col=0)
        self.df_raw["r_fuel"] = pd.read_csv(self.dirn + "fuel_consumption.csv", index_col=0)
        self.df_raw["c_fuel"] = pd.read_csv(self.dirn + "fuel_cost.csv", index_col=0)
        self.df_raw["r_acc"] = pd.read_csv(self.dirn + "accident_rate.csv", index_col=0)
        self.df_raw["c_acc"] = pd.read_csv(self.dirn + "accident_cost.csv", index_col=0)
        self.df_raw["r_gg"] = pd.read_csv(self.dirn + "greenhouse_rate.csv", index_col=0)
        self.df_raw["c_gg"] = pd.read_csv(self.dirn + "greenhouse_cost.csv", index_col=0)
        self.df_raw["r_em"] = pd.read_csv(self.dirn + "emission_rate.csv", index_col=0)
        self.df_raw["c_em"] = pd.read_csv(self.dirn + "emission_cost.csv", index_col=0)
        self.df_raw["noise"] = pd.read_csv(self.dirn + "noise.csv", index_col=0)
        

    def adjust_cpi(self, infl=0.02, N_bw=20, N_fw=30, verbose=False):
        """Fill in mising values and compute cumulative inflation 
        to be able to adjust the price level"""
        min_yr = self.cpi.index.min()
        max_yr = self.cpi.index.max()
        if verbose:
            print("Values between years %i and %i" % (min_yr, max_yr))
        v_bw = arange(min_yr - N_bw, min_yr)
        v_fw = arange(max_yr + 1, max_yr + N_fw + 1)

        self.cpi = pd.concat([
            pd.DataFrame({"cpi": infl*ones_like(v_bw), "year": v_bw}),
            self.cpi.reset_index(),
            pd.DataFrame({"cpi": infl*ones_like(v_fw), "year": v_fw}),
            ], sort=True)
        self.cpi.set_index("year", inplace=True)

        self.cpi["cpi2"] = self.cpi.cpi + 1.0

        # compute cumulative CPI
        self.cpi["cpi_index"] = ""
        self.cpi["cpi_index"] = \
            pd.to_numeric(self.cpi.cpi_index, errors="coerce")
        self.cpi.loc[self.pl] = 1.0
        ix = self.cpi.index.get_loc(self.pl)

        # backward
        for i in range(ix-1, -1, -1):
            self.cpi.iloc[i]["cpi_index"] = \
                self.cpi.iloc[i+1].cpi_index * self.cpi.iloc[i].cpi2
        
        # forward
        for i in range(ix+1, len(self.cpi)):
            self.cpi.iloc[i]["cpi_index"] = \
                self.cpi.iloc[i-1].cpi_index * self.cpi.iloc[i].cpi2

        self.cpi.drop(columns=["cpi2"], inplace=True)

        if verbose:
            print(self.cpi.head(10))


    def clean_data(self, verbose=False):
        """Remove unimportant columns and populate the df_clean dictionary"""
        for itm in self.df_raw.keys():
            if verbose:
                print(itm)
            self.df_clean[itm] = self.df_raw[itm].copy()
            if "nb" in self.df_clean[itm].columns:
                self.df_clean[itm].drop(columns=["nb"], inplace=True)
            if "unit" in self.df_clean[itm].columns:
                self.df_clean[itm].drop(columns=["unit"], inplace=True)
        
        for c in ["op_cost", "vtts", "voc", "c_acc", "c_gg", "c_em", "noise"]:
            if "scale" in self.df_clean[c].columns:
                 self.df_clean[c]["value"] =\
                    self.df_clean[c].value * self.df_clean[c].scale
                 self.df_clean[c].drop(columns=["scale"], inplace=True)

#        # different accident rate dataframe
#        c = "r_acc"
#        for col in ["fatal","severe_injury","light_injury","damage"]:
#            self.df_clean[c][col] = self.df_clean[c].col * self.df_clean[c].scale
#            self.df_clean[c].drop(columns=["scale"], inplace=True)


    def adjust_price_level(self, verbose=False):
        """Unify the prices for one price level"""
        for c in ["op_cost", "vtts", "voc", "c_fuel", "c_acc", "c_gg", "c_em", "noise"]:
            if verbose:
                print("Adjusting: %s" % c)
            self.df_clean[c]["value"] = self.df_clean[c].value \
                * self.df_clean[c].price_level\
                .map(lambda x: self.cpi.loc[x].cpi_index)
            self.df_clean[c].drop(columns=["price_level"], inplace=True)
            self.df_clean[c]["value"] = self.df_clean[c].value.round(3)


    def wrangle_data(self):
        self._wrangle_vtts()
        self._wrangle_accidents()
        self._wrangle_noise()


    def _wrangle_vtts(self, verbose=False):
        """Average the value of the travel time saved"""
        
        if "distance" in self.df_clean["vtts"].columns:
            if verbose:
                print("Contracting distance.")
            gr = self.df_clean["vtts"].groupby(by=["vehicle","substance","purpose",\
                "gdp_growth_adjustment"])
            vtts = gr["value"].mean()
            vtts = vtts.reset_index()
        else:
            vtts = self.df_clean["vtts"].copy()

        # add trip purpose and merge
        tp = self.df_clean["tp"].reset_index().melt(id_vars="vehicle", \
            var_name="purpose", value_name="purpose_ratio")
        vtts = pd.merge(vtts, tp, how="left", on=["vehicle","purpose"])

        # add passenger occupancy
        self.df_clean["p_occ"]["substance"] = "passengers"
        self.df_clean["p_occ"].reset_index(inplace=True)
        
        vtts = pd.merge(vtts, self.df_clean["p_occ"][["vehicle","value","substance"]],
             how="left", on=["vehicle","substance"], suffixes=("", "_occ"))

        # add freight occupancy
        self.df_clean["f_occ"]["substance"] = "freight"
        self.df_clean["f_occ"].reset_index(inplace=True)
        vtts = pd.merge(vtts, self.df_clean["f_occ"][["vehicle","value","substance"]],
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

        # unify gdp growth adjustment
        vtts["gdp_ga2"] = vtts.purpose_ratio * vtts.gdp_growth_adjustment
        vtts["value2"] = vtts.purpose_ratio * vtts.value

        vtts = vtts.groupby(["vehicle"])["gdp_ga2","value2"].sum()
        vtts.columns = ["gdp_growth_adjustment","value"]
        vtts["value"] = vtts.value.round(2)

        self.df_clean["vtts"] = vtts.copy()


    def _wrangle_accidents(self, verbose=False):
        self.df_clean["c_acc"]["value"] = self.df_clean["c_acc"].value *\
            self.df_clean["c_acc"].correct_unreported *\
            self.df_clean["c_acc"].correct_pass_per_acc
        self.df_clean["c_acc"].drop(columns=["correct_unreported",\
            "correct_pass_per_acc"], inplace=True)
        
        # ADD CONTRACTION WITH ACCIDENT RATE

    def _wrangle_noise(self, verbose=False):
        c = "noise"
        self.df_clean[c] = self.df_clean[c]\
            [self.df_clean[c].traffic_type == "thin"]
        self.df_clean[c].drop(columns=["traffic_type"], inplace=True)

        self.df_clean[c]["value2"] = self.df_clean[c].value\
            * self.df_clean[c].ratio
        gr = self.df_clean[c].groupby(["vehicle","area","gdp_growth_adjustment"])

        self.df_clean[c] = gr["value2"].sum()
        self.df_clean[c] = self.df_clean[c].reset_index()
        self.df_clean[c].rename(columns={"value2": "value"}, inplace=True)
        self.df_clean[c].set_index("vehicle", inplace=True)


    def create_time_benefit_matrices(self):
        """Define the time-cost matrices for each benefit"""
        for b in ["vtts", "voc", "acc", "em", "noise"]:
            # define the matrix for the years
            # populate it with GDP-adjusted growth and inflation
            # treat greenhouse differently due to varying cost
            pass

    
    def create_time_op_matrices(self):
        # create the matrix for the cost of road operation with time
        pass






