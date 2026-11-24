# main.py
# QAEI coursework - main script
# Chosen Scenario: Legacy airline, Airbus A321neo, 2 x Pratt & Whitney PW1100G, LHR -> BOS

# Importing required packages
import numpy as np
from qaei_missionandemissions_helpers import (dms_to_deg, isa, calc_LTO, cruise_mission, calc_mass_end,
                        cruise_fuel, calc_fuel_emissions, NOx_EI_BFFM2, nvPM_EI_BFFM2)
from qaei_ADVANCEDmissionandemissions_helpers import (nvPM_EI_MEEM, load_ps_params,
                        reynolds_number, lift_coefficient)

# ---------------- Scenario ----------------
airline = "Legacy"
data_name = "ATLANTIC"        # name used in the QAEI dataset files
load_factor = 0.80
flights_per_day = 2           # one round trip per day
days_per_year = 365

# ---------------- Aircraft: Airbus A321neo ----------------
aircraft = "Airbus A321neo"
MTOW = 97.0e3                 # maximum takeoff weight [kg]
fuel_capacity = 26.3e3        # [kg]
mach_cruise = 0.78
range_max = 7400.0                  # [km]
h_ceiling_ft = 40000.0              # cruise ceiling [ft]
h_cruise_ft = 36000.0               # chosen cruise altitude [ft]
h_cruise_m = h_cruise_ft * 0.3048     # chosen avg cruise altitude [m]
T_cruise, p_cruise = isa(h_cruise_m)     # ISA static temperature [K] and pressure [Pa] at cruise
seats = 206                   # 16J + 190Y (useful for defining ASK later for calculating cost impact)
L_over_D = 20.0               # estimated cruise L/D
wingspan = 35.80              # [m]

# ---------------- Engine: Pratt & Whitney PW1100G ----------------
engine = "Pratt & Whitney PW1130G-JM (TALON X, Block-D)"
n_engines = 2
EEDB_UID = "01P22PW165"
bypass_ratio = 11.6
OPR = 38.1
rated_thrust = 147.3          # F00 [kN]

# LTO data per engine, order: [Takeoff, Climb-out, Approach, Idle]
fuel_flow_LTO = np.array([1.023, 0.839, 0.279, 0.099])        # [kg/s]
EI_NOx_LTO = np.array([28.88, 22.01, 10.79, 5.74])            # [g/kg]
EI_nvPM_mass_LTO = np.array([37.7, 31.4, 0.4, 2.7])           # [mg/kg]
EI_nvPM_num_LTO = np.array([3.12e14, 4.01e14, 4.07e13, 3.25e14])  # [#/kg]

# ---------------- Planned flight: LHR -> BOS (using values from data cards on tutorial 1) ----------------
origin = "LHR"                                          # London Heathrow
origin_lat = dms_to_deg(51, 28, 39, "N")                # 51 28'39"N  [deg]
origin_lon = dms_to_deg(0, 27, 41, "W")                 # 000 27'41"W [deg]

destination = "BOS"                                     # Boston Logan
destination_lat = dms_to_deg(42, 21, 47, "N")           # 42 21'47"N  [deg]
destination_lon = dms_to_deg(71, 0, 23, "W")            # 071 00'23"W [deg]

origin_coords = (origin_lat, origin_lon)                # (lat, lon) tuple for cruise_mission [deg]
destination_coords = (destination_lat, destination_lon)

# ---------------- Quick check ----------------
print("Scenario:", airline, "-", aircraft, "with", n_engines, "x", engine)
print("Route:", origin, (round(origin_lat, 4), round(origin_lon, 4)),
      "->", destination, (round(destination_lat, 4), round(destination_lon, 4)))

# ---------------- Tutorial 1 ------------------

# Characterising the mission: LTO
fuel_LTO, NOx_LTO = calc_LTO(EI_NOx_LTO, fuel_flow_LTO)   # LTO fuel & NOx emissions (g) per engine: [kg], [g]
print("LTO fuel (per engine):", fuel_LTO, "kg")
print("LTO NOx (per engine):", NOx_LTO, "g")
_, nvPM_mass_LTO = calc_LTO(EI_nvPM_mass_LTO, fuel_flow_LTO)   # LTO fuel & nvPM emissions (mg) per engine: [kg], [mg]
print("LTO nvPM (by mass)  (per engine):", nvPM_mass_LTO, "mg")
_, nvPM_num_LTO = calc_LTO(EI_nvPM_num_LTO, fuel_flow_LTO)   # LTO fuel & nvPM emissions (#) per engine: [kg], [#]
print("LTO nvPM (by number)  (per engine):", nvPM_num_LTO, "#")
print("Therefore for both engines on the aircraft the following fuel burn and emissions are observed: ")
fuel_LTO_tot = fuel_LTO * n_engines
print("LTO Fuel Burn (mass) (total):", fuel_LTO_tot, "kg")
NOx_LTO_tot = NOx_LTO * n_engines
print("LTO NOx (mass) (total):", NOx_LTO_tot, "g")
nvPM_mass_LTO_tot = nvPM_mass_LTO * n_engines
print("LTO nvPM (mass) (total):", nvPM_mass_LTO_tot, "mg")
nvPM_num_LTO_tot = nvPM_num_LTO * n_engines
print("LTO nvPM (number) (total):", nvPM_num_LTO_tot, "#")

# Characterising the mission: time and distance
flight_dist_total, flight_time = cruise_mission(origin_coords, destination_coords, mach_cruise, h_cruise_m)
print("Cruise distance:", flight_dist_total / 1e3, "km")
print("Cruise time:", flight_time / 3600, "h")

# Calculating mean cruise fuel flow
mass_end = calc_mass_end(MTOW, fuel_capacity, seats, load_factor)                 
fuel_cruise = cruise_fuel(L_over_D, flight_dist_total, mass_end)                 
print("Mass at end of cruise:", mass_end, "kg")
print("Cruise fuel (total):", fuel_cruise, "kg")
mean_fuel_flowrate_cruise_per_engine = (fuel_cruise / flight_time) / n_engines
print("Mean fuel mass flow rate (per engine):", mean_fuel_flowrate_cruise_per_engine, "kg/s")

# Cruise emissions: CO2, H2O, SOx, etc. 
CO2_cruise, H2O_cruise, SO2_cruise, H2SO4_cruise = calc_fuel_emissions(fuel_cruise)   # whole aircraft [kg]
print("Cruise CO2 (total):", CO2_cruise, "kg")
print("Cruise H2O (total):", H2O_cruise, "kg")
print("Cruise SO2 (total):", SO2_cruise, "kg")
print("Cruise H2SO4 (total):", H2SO4_cruise, "kg")

# Cruise emissions: NOx (BFFM2)
EI_NOx_cruise = NOx_EI_BFFM2(np.array([fuel_flow_LTO, EI_NOx_LTO]), mach_cruise, T_cruise, p_cruise, mean_fuel_flowrate_cruise_per_engine)       # [g/kg]
NOx_cruise = EI_NOx_cruise * fuel_cruise / 1000                    # whole aircraft [kg]
print("Cruise EI NOx:", EI_NOx_cruise, "g/kg")
print("Cruise NOx (total):", NOx_cruise, "kg")

# Cruise emissions: nvPM (BFFM2)
EI_nvPM_mass_cruise, EI_nvPM_num_cruise = nvPM_EI_BFFM2(
    np.array([fuel_flow_LTO, EI_nvPM_mass_LTO, EI_nvPM_num_LTO]), mach_cruise, T_cruise, p_cruise, mean_fuel_flowrate_cruise_per_engine) #[mg/kg], [#/kg]
nvPM_mass_cruise = EI_nvPM_mass_cruise * fuel_cruise / 1e6       # whole aircraft [kg]
nvPM_num_cruise = EI_nvPM_num_cruise * fuel_cruise               # whole aircraft [#]
print("Cruise EI nvPM mass:", EI_nvPM_mass_cruise, "mg/kg")
print("Cruise EI nvPM number:", EI_nvPM_num_cruise, "#/kg")
print("Cruise nvPM mass (total):", nvPM_mass_cruise, "kg")
print("Cruise nvPM number (total):", nvPM_num_cruise, "#")

# ------------------ ADVANCED METHOOOOOD! ------------------------

# Poll-Schumann fuel burn
ps = load_ps_params("A21N")                                     # PS aircraft parameters
mass_end_PS = ps["OEM_i_kg"] + seats * 100.0 * load_factor      # end-of-cruise mass, PS-table empty mass [kg]

Re = reynolds_number(ps["Sref_m2"], mach_cruise, T_cruise, p_cruise)
CL = lift_coefficient(mass_end_PS, mach_cruise, p_cruise, ps["Sref_m2"])
print("PS Reynolds number:", Re)
print("PS lift coefficient (end of cruise):", CL)