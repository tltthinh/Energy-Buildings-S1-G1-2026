import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import dm4bem
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['CMU Serif']

controller = False
neglect_air_glass_capacity = True
imposed_time_step = False
Δt = 498    # s, imposed time step
Tout = 10
Tin = 20
print('Matrices and vectors for thermal circuit from TC_ss.csv')
df = pd.read_csv('TC_ss.csv')
df.style.apply(lambda x: ['background-color: yellow'
                          if x.name in df.index[-3:] or c in df.columns[-2:]
                          else '' for c in df.columns], axis=1)

# MODEL
# =====
# Thermal circuit
TC = dm4bem.file2TC('TC_ss.csv', name='', auto_number=False)

# by default TC['G']['q13'] = 0, i.e. Kp -> 0, no controller (free-floating)
if controller:
    TC['G']['q13'] = 1e3        # Kp -> ∞, almost perfect controller



# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

# DAE steady-state: To=10°C, Ti=20°C, nominal constant flow sources
# b vector has 14 entries (q0–q13):
#   To  at q0, q8, q9, q12  -> indices 0, 8, 9, 12
#   Ti  at q13               -> index 13
bss = np.zeros(14)
bss[[0, 8, 9, 12]] = Tout    # outdoor temperature
bss[[13]] = Tin             # indoor set-point temperature

fss = np.zeros(10)

A = TC['A']
G = TC['G']
diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)

θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
print(f'θss = {np.around(θss, 2)} °C')
print('=================================================')



# DAE steady-state with zero temperatures, only heat flow sources
bss_Q = np.zeros(14)

fss_Q = np.zeros(10)
fss_Q[0] = 667.5
fss_Q[4] = 54.0
fss_Q[7] = 273.6
fss_Q[8] = 135.0

θssQ = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss_Q + fss_Q)
print(f'θssQ = {np.around(θssQ, 2)} °C')
print('=================================================')
# State-space steady-state
# us has 9 entries: [To, To, To, To, Ti, 667.5, 54.0, 273.6, 135.0]
bT = np.array([Tout, Tout, Tout, Tout, Tin])        # [To, To, To, To, Ti]
fQ = np.array([0, 0, 0, 0]) # constant flow sources at nominal values
uss = np.hstack([bT, fQ])
print(f'uss = {uss}')

inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)

print(f'As = {inv_As.shape}')
print(f'As = {inv_As}')
print(f'Bs = {Bs}')
print(f'Bs = {Bs.shape}')
print(f'Cs = {Cs}')
print(f'Ds = {Ds}')
yss = (-Cs @ inv_As @ Bs + Ds) @ uss

yss = float(yss.values[0])
print(f'yss = {yss:.2f} °C')

print(f'Error between DAE and state-space: {abs(θss[5] - yss):.2e} °C')
print('=================================================')
# State-space: zero temperatures, same nominal constant flows
bT_Q = np.array([0, 0, 0, 0, 0])
fQ_Q = np.array([667.5, 54.0, 273.6, 135.0])
uss_Q = np.hstack([bT_Q, fQ_Q])

inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)
yssQ = (-Cs @ inv_As @ Bs + Ds) @ uss_Q

yssQ = float(yssQ.values[0])
print(f'uss = {uss_Q}')
print(f'yssQ = {yssQ:.2f} °C')

print(f'Error between DAE and state-space: {abs(θssQ[5] - yssQ):.2e} °C')
print('=================================================')
# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As
print(f'λ = {λ}')
print('=================================================')
# time step
Δtmax = 2 * min(-1. / λ)    # max time step for stability of Euler explicit
dm4bem.print_rounded_time('Δtmax', Δtmax)

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(Δtmax)

dm4bem.print_rounded_time('dt', dt)
print(f"dt = {dt:.0f} s")

if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")
print('=================================================')
# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)

# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                     periods=n, freq=f"{int(dt)}s")

To = 10 * np.ones(n)        # outdoor temperature
Ti = 20 * np.ones(n)        # indoor temperature set point

# Constant heat flow sources (numeric column names matching us values)
f667 = 667.5 * np.ones(n)   # heat source at θ0
f54  = 54.0  * np.ones(n)   # heat source at θ4
f274 = 273.6 * np.ones(n)   # heat source at θ7
f135 = 135.0 * np.ones(n)   # heat source at θ8

data = {'To': To, 'Ti': Ti,
        667.5: f667, 54.0: f54, 273.6: f274, 135.0: f135}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 0.0                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])

# outputs
y_exp = (Cs @ θ_exp.T + Ds @ u.T).T
y_imp = (Cs @ θ_imp.T + Ds @ u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)

ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {Δtmax:.0f} s')
plt.show()
print('=================================================')
print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {float(θss[5]):.4f} °C')
print(f'- state-space model: {float(yss):.4f} °C')
print(f'- steady-state response to step input: \
{y_exp["θ5"].tail(1).values[0]:.4f} °C')