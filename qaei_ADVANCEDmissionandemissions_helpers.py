# qaei_ADVANCEDmissionandemissions_helpers.py
# QAEI coursework - ADVANCED method routines for Set 1 (Emissions).
# Used only for the Results_Advanced outputs; the basic methods live in qaei_missionandemissions_helpers.py.
# Import into main.py with: from qaei_ADVANCEDmissionandemissions_helpers import nvPM_EI_MEEM
#
# Contents so far:
#   nvPM_EI_MEEM : cruise nvPM EIs (mass, number) using MEEM (Ahrens et al., 2022), 4-point interpolation

import warnings
import numpy as np
import os
import pandas as pd

# Part 1: Using the Poll-Schumann performance model to get a better estimate of the mission performance (specifically the fuel burn for a more accurate estimate to help build up the 3D 1degx1degx100m grid

# Poll-Schumann aircraft parameters for the Airbus A321neo (ICAO type A21N).
# Source: pycontrails ps-aircraft-params-20250328.csv (A21N row) which is located in (on my computer): C:\Users\Brendan\miniconda3\envs\contrails\Lib\site-packages\pycontrails\models\ps_model\static. This is hard coded for the sake of the coursework but I will add an option that can just read from the correct data file appropriately such that any aircraft can be selected!
A21N_PS = {"ICAO": "A21N", "Manufacturer": "Airbus", "Type": "A321-NEO", "Year_of_first_flight": 2016, "n_engine": 2, "winglets": "yes", "WV": 52, "MTOM_kg": 93500.0, "MLM_kg": 77809.0, "MZFM_kg": 73982.0, "OEM_i_kg": 48482.0, "MPM_i_kg": 25500.0, "MZFM_MTOM": 0.780748663, "OEM_i_MTOM": 0.513368984, "MPM_i_MTOM": 0.267379679, "Sref_m2": 122.4, "span_m": 35.27, "bf_m": 3.95, "delta_2": 0.025084952, "cos_sweep": 0.906307767, "AR": 10.16317729, "psi_0": 8.080988052, "Xo": 0.973612104, "wing_constant": 0.776932809, "j_2": 0.873049689, "j_1": 0.077052255, "CL_do": 0.626442739, "etaL_D_do": 6.502543201, "nominal_F00_ISA_kn": 255.7834783, "mf_max_T_O_SLS_kg_s": 1.771291826, "mf_idle_SLS_kg_s": 0.179714609, "M_des": 0.7527, "CT_des": 0.032774858, "eta_1": 0.379556514, "eta_2": 0.38526913, "Mec": 0.787080092, "Tec": 6.202228431, "FL_max": 391.0104987, "MMO": 0.82, "pi_max_pa": 20882.78822, "pinf_co_pa": 37618.41516, "nominal_opr": 34.13434783, "nominal_bpr": 11.63652174, "nominal_fpr": 1.65}

def load_ps_params(icao, csv_path=None):
    """Return the Poll-Schumann parameters for aircraft type `icao` as a dict.

    Reads the parameter table shipped with pycontrails unless a custom csv_path is provided (added this to test some values but is not necessarily needed for the coursework)
    
    Raises ValueError if the aircraft type is not in the table in pycontrails (or the one specified by user)
    
        Important note to Dr. Eastham, this reader may not work if the date for the aircraft parameters located in pycontrails changes (at the time of writing it was ps-aircraft-params-20250328.csv, however this can easily be changed by going into this code and copying the name of the csv ported with pycontrails and putting it below. Hopefully this should not be a problem at the time of marking."""
    
    if csv_path is None:
        import pycontrails                                      # only needed to locate the table
        csv_path = os.path.join(os.path.dirname(pycontrails.__file__),
                                "models", "ps_model", "static", "ps-aircraft-params-20250328.csv")
    table = pd.read_csv(csv_path, index_col="ICAO")
    if icao not in table.index:
        raise ValueError(f"Aircraft type '{icao}' not in Poll-Schumann table ({csv_path}).")
    params = table.loc[icao].to_dict()
    params["ICAO"] = icao
    return params

def reynolds_number(S, M, T, p):
    """Return Reynolds number (Re) based on wing reference length sqrt(S) (Poll-Schumann Part 1).
    S = wing reference area [m^2], M = Mach number, T = static temperature [K], p = static pressure [Pa]."""
    gamma = 1.4                                         # ratio of specific heats for air [-]
    R = 287.05                                          # specific gas constant for air [J/(kg K)]
    mu = 1.458e-6 * T**1.5 / (T + 110.4)                # dynamic viscosity, Sutherland's law [Pa s]
    return np.sqrt(S) * M * (p / mu) * np.sqrt(gamma / (R * T))   # = rho * V * sqrt(S) / mu [-]

def lift_coefficient(m, M, p, S):
    """Return lift coefficient C_L in SLF assuming lift = weight
    m = aircraft mass [kg], M = Mach number, p = static pressure [Pa], S = wing reference area [m^2]."""
    gamma = 1.4                                         # ratio of specific heats for air [-]
    g = 9.80665                                         # gravitational acceleration [m/s^2]
    q = 0.5 * gamma * p * M**2                          # dynamic pressure, 0.5*rho*V^2 = 0.5*gamma*p*M^2 [Pa]
    return m * g / (q * S)                              # [-]
    













# Part 3:

def nvPM_EI_MEEM(ei_mat, M, T, p, OPR):
    """Return cruise nvPM EIs (mass [mg/kg], number [#/kg]) using MEEM (Ahrens et al., 2022).
    ei_mat = [[LTO EI mass [mg/kg]], [LTO EI number [#/kg]]], order [T/O, C/O, APP, IDLE].
    LTO EIs must be the system-loss-corrected values from the ICAO EEDB (MEEM Step 0)."""
    gamma, eta_comp, Pi_alt, EF = 1.4, 0.88, 0.95, 1.1          # cruise values (Table 3, Eq. 14)
    k = (gamma - 1) / gamma
    EI_mass_LTO, EI_num_LTO = (np.asarray(x, dtype=float) for x in ei_mat)
    F_LTO = np.array([1.00, 0.85, 0.30, 0.07])                  # LTO thrust settings F/F00 [-]

    # Step 1: combustor inlet conditions at altitude
    T_t = T * (1 + 0.2 * M**2)                                  # total temperature [K]
    p_t = p * (1 + 0.2 * M**2)**3.5                             # total pressure [Pa]
    P3_alt = p_t * (1 + Pi_alt * (OPR - 1))                     # Eq. 7 [Pa]
    T3_alt = T_t * (1 + ((P3_alt / p_t)**k - 1) / eta_comp)     # Eq. 8 [K]

    # Step 2: ground reference with the same T3
    P3_GR = 101325.0 * (1 + eta_comp * (T3_alt / 288.15 - 1))**(1 / k)   # Eq. 11 [Pa]
    F_GR = (P3_GR / 101325.0 - 1) / (OPR - 1)                   # Eq. 12, F_GR/F00 [-]
    if np.any((F_GR < F_LTO.min()) | (F_GR > F_LTO.max())):
        warnings.warn(f"MEEM: F_GR/F00 = {F_GR} outside LTO range 0.07-1.00; EIs held at end values.")

    # Step 3: linear interpolation of ground EIs vs thrust (4-point method)
    i = np.argsort(F_LTO)
    EI_mass_GR = np.interp(F_GR, F_LTO[i], EI_mass_LTO[i])      # [mg/kg]
    EI_num_GR = np.interp(F_GR, F_LTO[i], EI_num_LTO[i])        # [#/kg]

    # Step 4: ground to altitude (Eq. 15, 16) - same factor for number keeps the ground N/M ratio
    corr = (P3_alt / P3_GR)**1.35 * EF**2.5                     # simplified DLR correction [-]
    EI_mass = EI_mass_GR * corr                                 # [mg/kg]
    EI_num = EI_num_GR * corr                                   # [#/kg] (avoids divide-by-zero if EI_mass_GR = 0)
    return EI_mass, EI_num
