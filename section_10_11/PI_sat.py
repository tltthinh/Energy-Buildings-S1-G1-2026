import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

# =============================================================
# Physical parameters  (single source of truth)
# =============================================================
materials = {
    'concrete':   {'lam': 1.4,   'rho': 2300, 'c': 880,  'w': 0.32},
    'insulation': {'lam': 0.027, 'rho': 55,   'c': 1210, 'w': 0.08},
    'glass':      {'lam': 1.4,   'rho': 2500, 'c': 1210, 'w': 0.04},
    'wood':       {'lam': 0.2,   'rho': 310,  'c': 1733, 'w': 0.07},
    'air':        {              'rho': 1.2,   'c': 1000.0          },
}

# -- Surfaces [m2]
S_c = 26.7   # outdoor wall
S_g = 3.6    # window glass
S_b = 2.7    # door

# -- Room geometry
A_floor = 30.0   # m2
H       = 3.0    # m
V_a     = A_floor * H   # m3

# -- Convection coefficients [W/m2K]
h_o = 25.0
h_i = 8.0

# -- Ventilation
ACH   = 1.0
V_dot = ACH / 3600.0 * V_a   # m3/s

# -- Optical properties
alpha_wSW = 0.25
alpha_gSW = 0.38
tau_gSW   = 0.30

# -- Design solar irradiance for f-row labels [W/m2]
E_d = 200.0

# -- HVAC
Q_MAX   = 2000.0   # W, saturation limit
Kp_hvac = 1e3      # W/K, proportional gain

# =============================================================
# Derived conductances  [W/K]
# =============================================================
G0  = h_o * S_c
G1  = 2 * materials['concrete']['lam']   * S_c / materials['concrete']['w']
G2  = 2 * materials['concrete']['lam']   * S_c / materials['concrete']['w']
G3  = 2 * materials['insulation']['lam'] * S_c / materials['insulation']['w']
G4  = 2 * materials['insulation']['lam'] * S_c / materials['insulation']['w']
G5  = h_i * S_c
G6  = h_i * S_g
G7  = materials['glass']['lam'] * S_g / materials['glass']['w']
G8  = h_o * S_g
G9  = h_o * S_b
G10 = 2 * materials['wood']['lam'] * S_b / materials['wood']['w']
G11 = 2 * materials['wood']['lam'] * S_b / materials['wood']['w']
G12 = materials['air']['rho'] * materials['air']['c'] * V_dot

# =============================================================
# Derived capacitances  [J/K]
# =============================================================
C1 = materials['concrete']['rho']   * materials['concrete']['c']   * S_c * materials['concrete']['w']
C3 = materials['insulation']['rho'] * materials['insulation']['c'] * S_c * materials['insulation']['w']
C5 = materials['air']['rho']        * materials['air']['c']        * V_a
C7 = materials['glass']['rho']      * materials['glass']['c']      * S_g * materials['glass']['w']
C8 = materials['wood']['rho']       * materials['wood']['c']       * S_b * materials['wood']['w']

# =============================================================
# f-row label values
# =============================================================
Phi_o = alpha_wSW * S_c * E_d
Phi_i = tau_gSW * alpha_wSW * S_g * E_d
Phi_a = alpha_gSW * S_g * E_d
Phi_d = alpha_wSW * S_b * E_d


# =============================================================
# Main simulation function
# =============================================================
def control_for(controller, period, dt=30, nonlinear_controller=True):

    # Obtain state-space representation
    # ==================================

    # -- Disassembled thermal circuits (TCd)
    #    base.py uses:  TCd = dm4bem.bldg2TCd(folder_path, TC_auto_number=True)
    #    Here we have a single TCgen.csv, so we load it as one TC
    #    and wrap it into the same TCd dict structure.
    TCd = dm4bem.file2TC('TCgen.csv', name='c0', auto_number=False)

    # Override every G and C with the dict-derived values
    # (CSV holds numbers only as placeholders for dm4bem to parse)
    TCd['G']['c0q0']  = G0
    TCd['G']['c0q1']  = G1
    TCd['G']['c0q2']  = G2
    TCd['G']['c0q3']  = G3
    TCd['G']['c0q4']  = G4
    TCd['G']['c0q5']  = G5
    TCd['G']['c0q6']  = G6
    TCd['G']['c0q7']  = G7
    TCd['G']['c0q8']  = G8
    TCd['G']['c0q9']  = G9
    TCd['G']['c0q10'] = G10
    TCd['G']['c0q11'] = G11
    TCd['G']['c0q12'] = G12
    TCd['G']['c0q13'] = Kp_hvac   # HVAC controller gain

    TCd['C']['c0θ1']  = C1
    TCd['C']['c0θ3']  = C3
    TCd['C']['c0θ5']  = C5
    TCd['C']['c0θ7']  = C7
    TCd['C']['c0θ8']  = C8

    TCd['f']['c0θ5']  = 'Qaux'   # HVAC injection at indoor air node

    # -- Assembled thermal circuit
    #    base.py uses:  ass_lists  = pd.read_csv('bldg/assembly_lists.csv')
    #                   ass_matrix = dm4bem.assemble_lists2matrix(ass_lists)
    #                   TC = dm4bem.assemble_TCd_matrix(TCd, ass_matrix)
    #    With a single TC there are no inter-circuit connections,
    #    so the assembled TC is identical to TCd.
    TC = TCd

    # -- State-space representation
    [As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

    # -- Eigenvalue analysis
    λ      = np.linalg.eig(As)[0]
    dt_max = 2 * min(-1. / λ)
    dt_max = dm4bem.round_time(dt_max)

    # Simulation with weather data
    # ============================
    start_date = '2000-' + period[0]
    end_date   = '2000-' + period[1]

    filename = './weather_data/FRA_AR_Grenoble.Alpes.Isere.AP.074860_TMYx.2011-2025.epw'
    [data, meta] = dm4bem.read_epw(filename, coerce_year=None)
    weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
    del data
    weather.index = weather.index.map(lambda t: t.replace(year=2000))
    weather = weather.loc[start_date:end_date]

    # Temperature sources
    To = weather['temp_air']

    Ti_day, Ti_night = 20, 16
    Ti_sp = pd.Series(
        [Ti_day if 6 <= hour <= 22 else Ti_night
         for hour in To.index.hour],
        index=To.index)

    # Flow-rate sources
    # base.py reads wall_out from walls_out.csv; here we use the
    # same surface orientation parameters directly
    surface_orientation = {'slope': 90, 'azimuth': 0, 'latitude': 45}
    rad_surf = dm4bem.sol_rad_tilt_surf(weather, surface_orientation,
                                        albedo=0.2)
    Etot = rad_surf.sum(axis=1)

    # Solar radiation flow sources (time-varying, scaled by Etot)
    Φo = alpha_wSW * S_c * Etot          # solar absorbed by outdoor wall
    Φi = tau_gSW * alpha_wSW * S_g * Etot   # solar transmitted through glass
    Φa = alpha_gSW * S_g * Etot          # solar absorbed by glass
    Φd = alpha_wSW * S_b * Etot          # solar absorbed by door

    # Auxiliary (internal) heat source
    Qa = pd.Series(0, index=To.index)

    # Input data set
    # Numeric keys match the f-row values in TCgen.csv (Phi_o, Phi_i, etc.)
    input_data_set = pd.DataFrame({
        'To':    To,
        'Ti':    Ti_sp,
        Phi_o:   Φo,
        Phi_i:   Φi,
        'Qaux':  Qa,
        Phi_a:   Φa,
        Phi_d:   Φd,
        'Etot':  Etot,
    })

    # Time integration
    # ----------------
    # Resample hourly data to time step dt
    input_data_set = input_data_set.resample(
        str(dt) + 'S').interpolate(method='linear')

    # Get input vector from input_data_set
    u = dm4bem.inputs_in_time(us, input_data_set)

    # Initial conditions
    θ0    = 20.0
    θ_exp = pd.DataFrame(index=u.index)
    θ_exp[As.columns] = θ0

    integral = [0.0]

    # Euler explicit time integration
    I = np.eye(As.shape[0])
    for k in range(1, u.shape[0] - 1):
        if nonlinear_controller:
            exec(controller)
        θ_exp.iloc[k + 1] = (I + dt * As) @ θ_exp.iloc[k] \
                             + dt * Bs @ u.iloc[k]

    # Outputs
    y = (Cs @ θ_exp.T + Ds @ u.T).T

    Kp = TC['G']['c0q13']   # W/K, controller gain

    if nonlinear_controller:
        q_HVAC = u['c0θ5']
    else:
        q_HVAC = Kp * (u['c0q13'] - y['c0θ5']) / A_floor

    # Plot
    data = pd.DataFrame({
        'To':     input_data_set['To'],
        'θi':     y['c0θ5'],
        'Etot':   input_data_set['Etot'],
        'q_HVAC': q_HVAC,
    })

    fig, axs = plt.subplots(2, 1, sharex=True)
    data[['To', 'θi']].plot(ax=axs[0], xticks=[],
                             ylabel='Temperature, $θ$ / °C')
    axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'], loc='upper right')
    axs[0].grid(True)
    data[['Etot', 'q_HVAC']].plot(ax=axs[1],
                                   ylabel='Heat rate, $q$ / (W·m⁻²)')
    axs[1].set(xlabel='Time')
    axs[1].legend(['$E_{total}$', '$q_{HVAC}$'], loc='upper right')
    axs[1].grid(True)
    print(TC['C']['c0θ5'])
    plt.show()


# =============================================================
# Scenarios
# =============================================================

#################
# Free running  #
#################
start_date = '02-01 12:00:00'
end_date   = '02-03 18:00:00'
period     = [start_date, end_date]

control_for("free running", period, dt=30, nonlinear_controller=False)


#################
# Heating       #
#################
heating = """
Tisp = 20       # indoor setpoint temperature, °C
Kpp  = 1e3      # controller gain
Ki   = 0.1      # integral gain
q_max = Q_MAX

error = Tisp - θ_exp.iloc[k - 1]['c0θ5']
u_pi  = Kpp * error + Ki * integral[0]
u_sat = float(np.clip(u_pi, 0, q_max))
u.iloc[k]['c0θ5'] = u_sat

if 0 < u_pi < q_max:
    integral[0] = integral[0] + error * dt
"""
control_for(heating, period)


#################
# Cooling       #
#################
cooling = """
Tisp  = 20      # indoor setpoint temperature, °C
Δθ    = 5       # temperature deadband, °C
Kpp   = 1e2     # controller gain
Ki    = 0.05
q_max = Q_MAX

error = Tisp - θ_exp.iloc[k - 1]['c0θ5']
u_pi  = Kpp * error + Ki * integral[0]

if θ_exp.iloc[k - 1]['c0θ5'] < Tisp + Δθ:
    u.iloc[k]['c0θ5'] = 0
    integral[0]       = 0.0
else:
    u_sat = float(np.clip(u_pi, -q_max, 0))
    u.iloc[k]['c0θ5'] = u_sat
    if -q_max < u_pi < 0:
        integral[0] = integral[0] + error * dt
"""
period = ['07-01 12:00:00', '07-03 12:00:00']
control_for(cooling, period)


####################################
# Heating and cooling with deadband #
####################################
heat_cool = """
Tisp  = 20      # indoor setpoint temperature, °C
Δθ    = 5       # temperature deadband, °C
Kpp   = 3e2     # controller gain
Ki    = 0.1
q_max = Q_MAX

error = Tisp - θ_exp.iloc[k - 1]['c0θ5']

if Tisp < θ_exp.iloc[k - 1]['c0θ5'] < Tisp + Δθ:
    u.iloc[k]['c0θ5'] = 0
    integral[0]       = 0.0
else:
    u_pi  = Kpp * error + Ki * integral[0]
    u_sat = float(np.clip(u_pi, -q_max, q_max))
    u.iloc[k]['c0θ5'] = u_sat
    if -q_max < u_pi < q_max:
        integral[0] = integral[0] + error * dt
"""
period = ['05-01 12:00:00', '05-03 12:00:00']
control_for(heat_cool, period)


#####################
# Solar protection  #
#####################
solar_protection = """
Tisp  = 20      # indoor setpoint temperature, °C
Δθ    = 1       # temperature deadband, °C
Kpp   = 3e2     # controller gain
q_max = Q_MAX

if θ_exp.iloc[k - 1]['c0θ5'] < Tisp + Δθ:
    u.iloc[k]['c0θ5'] = 0
else:
    u.iloc[k]['c0θ5'] = Kpp * (Tisp - θ_exp.iloc[k - 1]['c0θ5'])
    u.iloc[k]['c0θ5'] = min(u.iloc[k]['c0θ5'], 0)
    u.iloc[k]['c0θ0'] *= 0.1
"""
period = ['05-01 12:00:00', '05-03 12:00:00']
control_for(solar_protection, period)


##############################
# Passive cooling            #
##############################
passive_cooling = """
Tisp  = 20.0    # indoor setpoint temperature, °C
Δθ    = 1.0     # temperature deadband, °C
q_max = Q_MAX

ACH_n   = 10.0
V_dot_n = ACH_n / 3600.0 * V_a
G_free  = materials['air']['rho'] * materials['air']['c'] * V_dot_n
q_free  = G_free * (u.iloc[k - 1]['c0q0'] - θ_exp.iloc[k - 1]['c0θ5'])
q_free  = float(np.clip(q_free, -q_max, q_max))

if θ_exp.iloc[k - 1]['c0θ5'] < Tisp + Δθ:
    u.iloc[k]['c0θ5'] = 0
else:
    u.iloc[k]['c0θ5'] = 0
    if u.iloc[k - 1]['c0q0'] < Tisp:
        u.iloc[k]['c0θ5'] = q_free
        u.iloc[k]['c0θ0'] *= 0.1
"""
period = ['05-01 12:00:00', '05-03 12:00:00']
control_for(passive_cooling, period)
