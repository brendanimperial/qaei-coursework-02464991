# qaei_ADVANCEDmissionandemissions_helpers.py
# QAEI coursework - ADVANCED method routines for Set 1 (Emissions).
# Used only for the Results_Advanced outputs; the basic methods live in qaei_missionandemissions_helpers.py.
# Import into main.py with: from qaei_ADVANCEDmissionandemissions_helpers import nvPM_EI_MEEM
#
# Contents so far:
#   nvPM_EI_MEEM : cruise nvPM EIs (mass, number) using MEEM (Ahrens et al., 2022), 4-point interpolation

import warnings
import numpy as np


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
