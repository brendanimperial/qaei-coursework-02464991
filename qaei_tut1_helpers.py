# qaei_tut1_helpers.py
# QAEI coursework - all Tutorial 1 routines and helpers in one module.
# Import into main.py with: from qaei_tools import ...

import numpy as np

# Conversion helpers and isa

def dms_to_deg(deg, minutes, seconds, hemisphere):
    """Convert degrees/minutes/seconds + hemisphere ('N','S','E','W') to signed decimal degrees."""
    value = deg + minutes / 60 + seconds / 3600     # [deg]
    if hemisphere in ("S", "W"):                    # South and West are negative
        value = -value
    return value


def isa(h):
    """Return ISA static temperature [K] and pressure [Pa] at geopotential altitude h [m] (valid 0-20 km)."""
    T0 = 288.15                                                 # sea-level temperature [K]
    p0 = 101325.0                                               # sea-level pressure [Pa]
    L = 0.0065                                                  # troposphere lapse rate [K/m]
    R = 287.05                                                  # specific gas constant for air [J/(kg K)]
    g = 9.80665                                                 # gravitational acceleration [m/s^2]
    h_trop = 11000.0                                            # tropopause altitude [m]

    if h <= h_trop:                                             # troposphere: linear temperature drop
        T = T0 - L * h                                          # [K]
        p = p0 * (T / T0) ** (g / (R * L))                      # [Pa]
    else:                                                       # lower stratosphere (11-20 km): isothermal
        T_trop = T0 - L * h_trop                                # 216.65 K
        p_trop = p0 * (T_trop / T0) ** (g / (R * L))            # ~22,632 [Pa]
        T = T_trop                                              # [K]
        p = p_trop * np.exp(-g * (h - h_trop) / (R * T_trop))   # [Pa]
    return T, p


def calc_mass_end(MTOW, fuel_capacity, seats, load_factor, m_pax=100.0):
    """Return estimated aircraft mass at end of cruise [kg] (OEW + carried payload). It assumes that the avg pax + their cargo is 100 kg."""
    m_payload_max = seats * m_pax                  # finding mass of max payload (i.e. all pax and luggage loaded) [kg]
    m_fuel_max = fuel_capacity                     # finding mass of fuel if tanks completely filled [kg]
    m_OEW = MTOW - m_payload_max - m_fuel_max      # finding operating empty weight of aircraft [kg]
    mass_end = m_OEW + (m_pax*seats*load_factor)   # therefore finding the (approximate) mass at end of cruise [kg]
    return mass_end

# Tutorial 1: mission characterisation

def calc_LTO(ei_vec, fuel_flow_vec):
    """Return LTO fuel burn and emissions for chosen species (e.g. CO2)."""
    dt = np.array([0.7, 2.2, 4.0, 26.0]) * 60       # defining the ICAO standard times for [T/O, C/O, APPROACH, IDLE/TAXI] [s]
    fuel_mode = fuel_flow_vec * dt                  # defining fuel burned in each mode [kg]
    fuel_lto = np.sum(fuel_mode)                    # calculating the total LTO fuel burn [kg]
    E_lto = np.sum(fuel_mode * ei_vec)              # calculating the total emissions sumi=1-4 mdotfuel * eilto * dtlto [EI mass unit, e.g. g for NOx, mg/# for nvPM]
    return fuel_lto, E_lto


def cruise_mission(origin, destination, mach, h_cruise):
    """Return total cruise distance [m] and flight time [s] between two (lat, lon) tuples [deg], at cruise Mach and altitude h_cruise [m]."""
    R_E = 6371e3                                    # mean Earth radius [m]
    gamma = 1.4                                     # ratio of specific heats for air [-]
    R_u = 8.314                                     # universal gas constant [J/(mol K)]
    M_air = 0.02897                                 # molar mass of air [kg/mol]
    T_cruise, _ = isa(h_cruise)                     # ISA static temperature at cruise altitude [K]

    lat_o, lon_o = np.radians(origin)               # unpacking origin (lat, lon) and converting [deg] -> [rad]
    lat_d, lon_d = np.radians(destination)          # unpacking destination (lat, lon) and converting [deg] -> [rad]

    flight_dist_total = R_E * np.arccos(np.cos(lat_o) * np.cos(lat_d) * np.cos(lon_o - lon_d)
                                        + np.sin(lat_o) * np.sin(lat_d))   # great-circle distance (spherical law of cosines) [m]

    a = np.sqrt(gamma * R_u * T_cruise / M_air)     # speed of sound at cruise [m/s]
    V = mach * a                                    # cruise speed [m/s]
    flight_time = flight_dist_total / V             # cruise flight time [s]
    return flight_dist_total, flight_time


def cruise_fuel(L_over_D, distance, mass_end):
    """Return total cruise fuel burn [kg] using the Breguet range equation, m_fuel = m_end * (exp(C*d) - 1)."""
    g = 9.81                                        # gravitational acceleration [m/s^2]
    LHV = 44e6                                      # lower heating value of kerosene (Jet A-1) [J/kg]
    eta_o = 0.3                                     # propulsive efficiency (assumed for PW1000)
    C = g / (L_over_D * LHV * eta_o)                # Breguet constant [1/m]
    m_fuel = mass_end * (np.exp(C * distance) - 1)  # total cruise fuel burn [kg]
    return m_fuel

# Tutorial 1: emissions

def calc_fuel_emissions(fuel_mass, FSC=600e-6, eps_S=0.02):
    """Return CO2, H2O, SO2 and H2SO4 emitted [kg] for a given fuel burn [kg].
    Fuel-based EIs (Jet A-1) from Tutorial 1 / lecture slides; scale directly with fuel burned."""
    EI_CO2 = 3.16                                   # [kg CO2 / kg fuel]
    EI_H2O = 1.24                                   # [kg H2O / kg fuel]
    M_S, M_SO2, M_H2SO4 = 32.0, 64.0, 98.0          # molar masses [g/mol]
    EI_SO2 = FSC * (M_SO2 / M_S) * (1 - eps_S)      # [kg SO2 / kg fuel]   (~1.18 g/kg)
    EI_H2SO4 = FSC * (M_H2SO4 / M_S) * eps_S        # [kg H2SO4 / kg fuel] (~37 mg/kg)
    m_CO2 = EI_CO2 * fuel_mass                      # [kg]
    m_H2O = EI_H2O * fuel_mass                      # [kg]
    m_SO2 = EI_SO2 * fuel_mass                      # [kg]
    m_H2SO4 = EI_H2SO4 * fuel_mass                  # [kg]
    return m_CO2, m_H2O, m_SO2, m_H2SO4


def NOx_EI(ei_vec, M, T, p, ff):
    """Return cruise EI_NOx [g/kg] using BFFM2. ei_vec = [[LTO fuel flows [kg/s]], [LTO EI_NOx [g/kg]]]."""
    theta = T / 288.15
    delta = p / 101325.0
    ff_LTO, EI_LTO = ei_vec

    ff_SL = ff * theta**3.8 / delta * np.exp(0.2 * M**2)       # sea-level equivalent fuel flow [kg/s]

    i = np.argsort(ff_LTO)                                      # interp needs increasing x
    EI_SL = 10**np.interp(np.log10(ff_SL), np.log10(ff_LTO[i]), np.log10(EI_LTO[i]))   # log-log interpolation [g/kg]

    H = -19.0 * (0.0 - 0.0063)                                  # humidity term, q = 0
    return EI_SL * (delta**1.02 / theta**3.3)**0.5 * np.exp(H)  # cruise EI_NOx [g/kg]


def nvPM_EI(ei_mat, M, T, p, OPR):
    """Return cruise nvPM EIs (mass [mg/kg], number [#/kg]) using MEEM (Ahrens et al., 2023).
    ei_mat = [[LTO EI mass [mg/kg]], [LTO EI number [#/kg]]], order [T/O, C/O, APP, IDLE]."""
    gamma, eta_comp, Pi_alt, EF = 1.4, 0.88, 0.95, 1.1          # cruise values (Table 3, Eq. 14)
    k = (gamma - 1) / gamma
    EI_mass_LTO, EI_num_LTO = ei_mat
    F_LTO = np.array([1.00, 0.85, 0.30, 0.07])                  # LTO thrust settings F/F00 [-]

    # Step 1: combustor inlet conditions at altitude
    T_t = T * (1 + 0.2 * M**2)                                  # total temperature [K]
    p_t = p * (1 + 0.2 * M**2)**3.5                             # total pressure [Pa]
    P3_alt = p_t * (1 + Pi_alt * (OPR - 1))                     # Eq. 7 [Pa]
    T3_alt = T_t * (1 + ((P3_alt / p_t)**k - 1) / eta_comp)     # Eq. 8 [K]

    # Step 2: ground reference with the same T3
    P3_GR = 101325.0 * (1 + eta_comp * (T3_alt / 288.15 - 1))**(1 / k)   # Eq. 11 [Pa]
    F_GR = (P3_GR / 101325.0 - 1) / (OPR - 1)                   # Eq. 12, F_GR/F00 [-]

    # Step 3: linear interpolation of ground EIs vs thrust (4-point method)
    i = np.argsort(F_LTO)
    EI_mass_GR = np.interp(F_GR, F_LTO[i], EI_mass_LTO[i])      # [mg/kg]
    EI_num_GR = np.interp(F_GR, F_LTO[i], EI_num_LTO[i])        # [#/kg]

    # Step 4: ground to altitude (Eq. 15, 16)
    EI_mass = EI_mass_GR * (P3_alt / P3_GR)**1.35 * EF**2.5     # [mg/kg]
    EI_num = EI_mass * EI_num_GR / EI_mass_GR                   # [#/kg]
    return EI_mass, EI_num
