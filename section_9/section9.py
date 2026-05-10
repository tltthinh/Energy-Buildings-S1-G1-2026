#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import dm4bem

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['CMU Serif']

# Inputs
controller = True
indoor_air_capacity = True
glass_capacity = True

date_start = '2025-07-01 12:00'
date_end = '2025-07-31 12:00'

#date_start = '2025-01-01 12:00'
#date_end = '2025-01-31 12:00'


# Model
# =====
TC = dm4bem.file2TC('TCgen.csv', name='', auto_number=False)

# by default TC['G']['q13'] = 0, i.e. Kp -> 0, no controller (free-floating)
if controller:
    TC['G']['q13'] = 1e4       # Kp, controller gain
if not indoor_air_capacity:
    TC['C']['θ5'] = 0          # indoor air heat capacity
if not glass_capacity:
    TC['C']['θ7'] = 0          # glass (window) heat capacity

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

# Eigenvalues analysis
λ = np.linalg.eig(As)[0]
dt_max = 2 * min(-1. / λ)
dt = dm4bem.round_time(dt_max)

# Inputs
# ======
file_weather = 'FRA_AR_Grenoble.Alpes.Isere.AP.074860_TMYx.2011-2025.epw'
[data, meta] = dm4bem.read_epw(file_weather, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data

weather.index = weather.index.map(lambda t: t.replace(year=2025))
weather = weather.loc[date_start:date_end]

# Temperature sources
To = weather['temp_air']

Ti_day, Ti_night = 20, 16
Ti = pd.Series(
    [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
    index=To.index)

# Total solar irradiance (for plotting Etot)
surface_orientation = {'slope': 90, 'azimuth': 0, 'latitude': 45}
albedo = 0.2
rad_surf = dm4bem.sol_rad_tilt_surf(weather, surface_orientation, albedo)
Etot = rad_surf.sum(axis=1)

# Input data set
# TCgen.csv flow sources are numeric constants embedded in the f row.
# Pass them as constant columns with numeric keys matching us values.
input_data_set = pd.DataFrame({
    'To':   To,
    'Ti':   Ti,
    667.5:  667.5,    # constant heat source at θ0, W
    54.0:   54.0,     # constant heat source at θ4, W
    273.6:  273.6,    # constant heat source at θ7, W
    135.0:  135.0,    # constant heat source at θ8, W
    'Etot': Etot      # for plotting only
})

# Simulation
# ==========
input_data_set = input_data_set.resample(
    str(dt) + 'S').interpolate(method='linear')

u = dm4bem.inputs_in_time(us, input_data_set)

θ0 = 20
θ_exp = pd.DataFrame(index=u.index)
θ_exp[As.columns] = θ0

I = np.eye(As.shape[0])

for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k + 1]

y = (Cs @ θ_exp.T + Ds @ u.T).T

Kp = TC['G']['q13']
S = 3 * 3
q_HVAC = Kp * (u['q13'] - y['θ5']) / S  # W/m²

# Plots
data = pd.DataFrame({'To':     input_data_set['To'],
                     'θi':     y['θ5'],
                     'Etot':   input_data_set['Etot'],
                     'q_HVAC': q_HVAC})

fig, axs = plt.subplots(2, 1)
data[['To', 'θi']].plot(ax=axs[0], xticks=[], ylabel='Temperature, $θ$ / °C')
axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'], loc='upper left')
data[['Etot', 'q_HVAC']].plot(ax=axs[1], ylabel='Heat rate, $q$ / (W·m⁻²)')
axs[1].set(xlabel='Time')
axs[1].legend(['$E_{total}$', '$q_{HVAC}$'], loc='upper left')
axs[0].set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {dt_max:.0f} s')

plt.show()

# Outputs
dm4bem.print_rounded_time("Time step:", dt)
print(f"Mean outdoor temperature: {data['To'].mean():.1f} °C")
print(f"Min. indoor temperature: {data['θi'].min():.1f} °C")
print(f"Max. indoor temperature: {data['θi'].max():.1f} °C")

max_load = data['q_HVAC'].max()
max_load_index = data['q_HVAC'].idxmax()
Q_heat = q_HVAC[q_HVAC > 0].sum() * dt / 3.6e6
Q_cool = q_HVAC[q_HVAC < 0].sum() * dt / 3.6e6

print(f"Max. load: {max_load:.1f} W at {max_load_index}")
print(f"Energy consumption for heating: {Q_heat:.1f} kWh")
print(f"Energy consumption for cooling: {-Q_cool:.1f} kWh")