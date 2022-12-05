import pandas as pd
import os

def load_lietavska_lucka():
    """
    Read the prepared Excel file and wrangle the dataframes
    into the format needed by pycba.

    The example is based on update of fesibility study for
    the highway feeder near Lietavska Lucka.

    The original CBA was done by the Ministry of Transport of SR:
    https://www.opii.gov.sk/strategicke-dokumenty/studie-realizovatelnosti/studia-uskutocnitelnosti-i-64-topolcany-zilina-aktualizacia
    """

    out_dict = {}
    input_file = dirn = os.path.dirname(__file__) \
                        + '/examples/lietavska_lucka/inputs.xlsx'

    # read the sheets
    out_dict['C_fin'] = pd.read_excel(input_file,
                                      sheet_name='capex',
                                      index_col=0)
    out_dict['RP'] = pd.read_excel(input_file,
                                   sheet_name='road_parameters',
                                   index_col=0)

    out_dict['RP']['lower_usage'] = \
        out_dict['RP']['lower_usage'].astype(bool)

    out_dict['acc'] = pd.read_excel(input_file,
                                    sheet_name='custom_accident_rates')

    out_dict['TP'] = pd.read_excel(input_file,
                                   sheet_name='toll_parameters')

    # read intensity and velocity
    # intensity and velcoity are in model link form to make the
    # comparison with original CBA easier.
    # They must be manipulated into form suitable for pycba.
    df_int_0 = pd.read_excel(input_file,
                             index_col=0, sheet_name="intensities_0")
    df_int_1 = pd.read_excel(input_file,
                             index_col=0, sheet_name="intensities_1")
    df_vel_0 = pd.read_excel(input_file,
                             index_col=0, sheet_name="velocities_0")
    df_vel_1 = pd.read_excel(input_file,
                             index_col=0, sheet_name="velocities_1")

    rp_links = out_dict['RP'].reset_index()[['id_road_section',
                                             'id_model_section',
                                             'variant']]
    rp_links0 = rp_links[rp_links['variant'] == 0].drop(columns='variant')
    rp_links1 = rp_links[rp_links['variant'] == 1].drop(columns='variant')

    # wrangle intensities - merge onto road parameter structure
    out_dict['I0'] = pd.merge(rp_links0,
                              df_int_0.reset_index(drop=True),
                              how='inner').set_index('id_road_section')

    out_dict['I1'] = pd.merge(rp_links1,
                              df_int_1.reset_index(drop=True),
                              how='inner').set_index('id_road_section')

    out_dict['V0'] = pd.merge(rp_links0,
                              df_vel_0.reset_index(drop=True),
                              how='inner').set_index('id_road_section')

    out_dict['V1'] = pd.merge(rp_links1,
                              df_vel_1.reset_index(drop=True),
                              how='inner').set_index('id_road_section')

    return out_dict