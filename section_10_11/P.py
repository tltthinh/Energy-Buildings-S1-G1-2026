import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem


def control_for(controller, period, dt=30, nonlinear_controller=True):
    # =========================================================
    # Obtain state-space representation
    # =========================================================

    # Disassembled thermal circuits
    folder_path = 'bldg2'
    TCd = dm4bem.bldg2TCd(folder_path, TC_auto_number=True)

    # Set HVAC controller gain (G=0 in TC3.csv placeholder)
    #TCd['c3']['G']['c3_q0'] = 1e3       # Kp [W/K]
    # TCd['c2']['C']['c2_θ0'] = 0       # zero indoor air capacity
    # TCd['c0']['C']['c0_θ0'] = 0       # zero glass capacity

    # Assembled thermal circuit
    ass_lists  = pd.read_csv('bldg2/assembly_lists.csv')
    ass_matrix = dm4bem.assemble_lists2matrix(ass_lists)
    TC         = dm4bem.assemble_TCd_matrix(TCd, ass_matrix)

    # State-space
    [As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

    # =========================================================
    # Eigenvalue analysis — maximum stable time step
    # =========================================================
    λ      = np.linalg.eig(As)[0]
    dt_max = 2 * min(-1. / λ)
    dt_max = dm4bem.round_time(dt_max)

    # =========================================================
    # Simulation with weather data
    # =========================================================
    start_date = '2025-' + period[0]
    end_date   = '2025-' + period[1]

    # Weather
    filename = './weather_data/FRA_AR_Grenoble.Alpes.Isere.AP.074860_TMYx.2011-2025.epw'
    [data, meta] = dm4bem.read_epw(filename, coerce_year=None)
    weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
    del data
    weather.index = weather.index.map(lambda t: t.replace(year=2025))
    weather = weather.loc[start_date:end_date]

    # Temperature sources
    To = weather['temp_air']

    Ti_day, Ti_night = 20, 16
    Ti_sp = pd.Series(
        [Ti_day if 6 <= hour <= 22 else Ti_night
         for hour in To.index.hour],
        index=To.index)

    # Flow-rate sources — solar irradiance on south vertical wall
    wall_out = pd.read_csv('bldg2/walls_out.csv')
    w0 = wall_out[wall_out['ID'] == 'w0']

    surface_orientation = {'slope':   w0['β'].values[0],
                           'azimuth': w0['γ'].values[0],
                           'latitude': 45}

    rad_surf = dm4bem.sol_rad_tilt_surf(
        weather, surface_orientation, w0['albedo'].values[0])

    Etot = rad_surf.sum(axis=1)

    # Window / door optical properties
    α_gSW = 0.38    # short-wave absorptivity: reflective blue glass
    τ_gSW = 0.30    # short-wave transmittance: reflective blue glass
    S_g   = 3.6     # m2, window glass area
    S_b   = 2.7     # m2, door area

    # Flow-rate sources
    Φo = w0['α0'].values[0] * w0['Area'].values[0] * Etot   # solar on outdoor wall
    Φi = τ_gSW * w0['α0'].values[0] * S_g * Etot            # solar through glass → indoor air
    Φa = α_gSW * S_g * Etot                                  # solar absorbed by glass
    Φd = w0['α0'].values[0] * S_b * Etot                     # solar on door

    # Auxiliary (internal) heat sources
    Qa = pd.Series(0, index=To.index)

    # Input data set
    input_data_set = pd.DataFrame({'To':    To,
                                   'Ti_sp': Ti_sp,
                                   'Φo':    Φo,
                                   'Φi':    Φi,
                                   'Qa':    Qa,
                                   'Φa':    Φa,
                                   'Φd':    Φd,
                                   'Etot':  Etot})

    # =========================================================
    # Time integration
    # =========================================================
    # Resample hourly data to time step dt
    input_data_set = input_data_set.resample(
        str(dt) + 'S').interpolate(method='linear')

    # Get input vector in time from input_data_set
    u = dm4bem.inputs_in_time(us, input_data_set)

    # Initial conditions
    θ0    = 20.0
    θ_exp = pd.DataFrame(index=u.index)
    θ_exp[As.columns] = θ0

    # Euler explicit time integration
    I = np.eye(As.shape[0])
    for k in range(1, u.shape[0] - 1):
        if nonlinear_controller:
            exec(controller)
        θ_exp.iloc[k + 1] = (I + dt * As) @ θ_exp.iloc[k] \
                             + dt * Bs @ u.iloc[k]

    # Outputs
    y = (Cs @ θ_exp.T + Ds @ u.T).T

    Kp = TC['G']['c3_q0']   # W/K, controller gain
    S  = 30                  # m², room floor area

    # q_HVAC [W/m²]
    if nonlinear_controller:
        q_HVAC = u['c2_θ0']
    else:
        q_HVAC = Kp * (u['c3_q0'] - y['c2_θ0']) / S

    # Plot
    data = pd.DataFrame({'To':     input_data_set['To'],
                         'θi':     y['c2_θ0'],
                         'Etot':   input_data_set['Etot'],
                         'q_HVAC': q_HVAC})

    fig, axs = plt.subplots(2, 1, sharex=True)
    data[['To', 'θi']].plot(ax=axs[0],
                             xticks=[],
                             ylabel='Temperature, $θ$ / °C')
    axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'], loc='upper right')
    axs[0].grid(True)

    data[['Etot', 'q_HVAC']].plot(ax=axs[1],
                                   ylabel='Heat rate, $q$ / (W·m⁻²)')
    axs[1].set(xlabel='Time')
    axs[1].legend(['$E_{total}$', '$q_{HVAC}$'], loc='upper right')
    axs[1].grid(True)
    print(TC['C']['c2_θ0'])
    plt.show()



# =============================================================
# Heating
# =============================================================
heating = """
Tisp = 20       # indoor setpoint temperature, °C
Kpp  = 1e3      # controller gain

if Tisp < θ_exp.iloc[k - 1]['c2_θ0']:
    u.iloc[k]['c2_θ0'] = 0
else:
    u.iloc[k]['c2_θ0'] = Kpp * (Tisp - θ_exp.iloc[k - 1]['c2_θ0'])
"""
start_date = '01-01 12:00:00'
end_date   = '01-31 12:00:00'
period     = [start_date, end_date]
control_for(heating, period)


# =============================================================
# Cooling
# =============================================================
cooling = """
Tisp = 20       # indoor setpoint temperature, °C
Δθ   = 5        # temperature deadband, °C
Kpp  = 1e2      # controller gain

if θ_exp.iloc[k - 1]['c2_θ0'] < Tisp + Δθ:
    u.iloc[k]['c2_θ0'] = 0
else:
    u.iloc[k]['c2_θ0'] = Kpp * (Tisp - θ_exp.iloc[k - 1]['c2_θ0'])
"""
period = ['07-01 12:00:00', '07-31 12:00:00']
control_for(cooling, period)


# =============================================================
# Heating and cooling with deadband
# =============================================================
heat_cool = """
Tisp = 20       # indoor setpoint temperature, °C
Δθ   = 5        # temperature deadband, °C
Kpp  = 3e2      # controller gain

if Tisp < θ_exp.iloc[k - 1]['c2_θ0'] < Tisp + Δθ:
    u.iloc[k]['c2_θ0'] = 0
else:
    u.iloc[k]['c2_θ0'] = Kpp * (Tisp - θ_exp.iloc[k - 1]['c2_θ0'])
"""
period = ['07-01 12:00:00', '07-31 12:00:00']
control_for(heat_cool, period)


# =============================================================
# Solar protection
# =============================================================
solar_protection = """
Tisp = 20       # indoor setpoint temperature, °C
Δθ   = 1        # temperature deadband, °C
Kpp  = 3e2      # controller gain

if θ_exp.iloc[k - 1]['c2_θ0'] < Tisp + Δθ:
    u.iloc[k]['c2_θ0'] = 0
else:
    u.iloc[k]['c2_θ0'] = Kpp * (Tisp - θ_exp.iloc[k - 1]['c2_θ0'])
    u.iloc[k]['c2_θ0'] = min(u.iloc[k]['c2_θ0'], 0)
    u.iloc[k]['ow0_θ0'] *= 0.1
"""
period = ['07-15 12:00:00', '07-17 12:00:00']
control_for(solar_protection, period)


# =============================================================
# Passive cooling
# =============================================================
passive_cooling = """
Tisp  = 20.0    # indoor setpoint temperature, °C
Δθ    = 1.0     # temperature deadband, °C

l      = 30.0   # m², room floor area
H      = 3.0    # m, room height
Va     = l * H  # m³, room volume
ACH    = 10.0   # 1/h, air changes per hour (night ventilation)
Va_dot = ACH / 3600 * Va
ρ      = 1.2    # kg/m³
c      = 1000.0 # J/(kg·K)
G_free = ρ * c * Va_dot
q_free = G_free * (u.iloc[k - 1]['ow0_q0'] - θ_exp.iloc[k - 1]['c2_θ0'])

if θ_exp.iloc[k - 1]['c2_θ0'] < Tisp + Δθ:
    u.iloc[k]['c2_θ0'] = 0
else:
    u.iloc[k]['c2_θ0'] = 0
    if u.iloc[k - 1]['ow0_q0'] < Tisp:
        u.iloc[k]['c2_θ0']  = q_free
        u.iloc[k]['ow0_θ0'] *= 0.1
"""
period = ['07-15 12:00:00', '07-17 12:00:00']
control_for(passive_cooling, period)