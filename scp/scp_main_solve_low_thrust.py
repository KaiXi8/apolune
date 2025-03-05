import sys
import os

# Construct the full path to the directory containing the package
project_path = '/Users/hofmannc/git/apolune'

# Add the directory to sys.path
sys.path.append(project_path)


import numpy as np
from copy import deepcopy
from scipy.integrate import odeint
import dynamics.crtbp_dynamics_control as crtbp
# import dynamics.crtbp_dynamics as crtbp
# import dynamics_coeff.dynamics_general_jit as dyn_coeff
import dynamics_coeff.dynamics_general_control_jit as dyn_coeff
# import dynamics_coeff.dynamics_general_control as dyn_coeff
import propagation.propagator as propagation
import cvxpy as cvx
import matplotlib.pyplot as plt
import spiceypy as spice
import init.load_kernels as krn
import dynamics_coeff.rnbp_rpf_utils as rnbp_utils
import time as tm
import scp_core_low_thrust as scp_core
import dynamics_coeff.homotopy as homotopy
import scp_solve_low_thrust as scp_solve
import copy


def initial_guess(auxdata):
    n_x = auxdata['problem']['n_x']
    n_u = auxdata['problem']['n_u']
    N = auxdata['problem']['N']
    
    time = auxdata['param']['time_vec']
    x0 = auxdata['param']['x0']
#     x0[3:6] *= 1.001 # converges
#     x0[3:6] *= 1.004 # failed
#     x0[3:6] *= 1.002 # converges

#     x0[3:6] *= 1.002445 # converges for crtbp when propagating nb dynamics
#     x0 *= 1.000005 # 
    
    
    if auxdata['problem']['free_tf'] == 1:
        tf = auxdata['param']['tf']
        time = time * tf
        p_guess = np.array([tf])
    else:
        p_guess = np.empty(0)
    
    u_guess = np.zeros((N, n_u))
    x_guess = propagation.propagate(crtbp.dynamics, x0, np.zeros(n_u), p_guess, time, auxdata)
    # x_guess = propagation.propagate(crtbp.dynamics, x0, u_guess, p_guess, time, auxdata)
#     x_guess = propagation.propagate(dyn_coeff.dynamics, x0, np.zeros(n_u), p_guess, time, auxdata)
    

    return x_guess, u_guess, p_guess



mu_earth_moon = 1.21506683e-2
LU = 3.84405000e5 # km
TU = 375197.5822850085 # seconds
VU = LU / TU # km/s
ACU = VU / TU # Acceleration unit [km/s^2]

g0_dim = 9.80665e-3 # [km s^-2]

foh_flag = 1

# crtbp
mu = mu_earth_moon

# bcrfbp
mu_sun = 3.28900541e5 # adim
a_sun = 3.88811143e2 # adim
om_sun = -9.25195985e-1
sun_angle_t0 = 0.0

# rnbp_rpf
krn.load_kernels()

G = 6.67408e-20  # [km^3 kg^−1 s^−2]
# AU = 149597870.693 # km
AU = 1.495978706136889e+08 # [km] from SPICE

# spacecraft data
m0_dim = 22.6 # [kg]
Tmax_dim = 2.2519*1e-3 # [N]
# Tmax_dim = 2.2519*1e-2 # [N]
Isp_dim = 3067 # [s]

MU = m0_dim
FU = MU * ACU / 1e-3

m0 = m0_dim / MU
Tmax = Tmax_dim / FU 
Isp = Isp_dim / TU
g0 = g0_dim / ACU

observer_id = 0

# (index, spice_id) pairs
id_mer = [0, 1]
id_ven = [1, 2]
id_ear = [2, 399]
id_mar = [3, 4]
id_jup = [4, 5]
id_sat = [5, 6]
id_ura = [6, 7]
id_nep = [7, 8]
id_plu = [8, 9]
id_moo = [9, 301]
id_sun = [10, 10]

# Standard gravitational parameter ( μ [km^3 s−2] )
GM_sun = spice.bodvrd(str(id_sun[1]), "GM", 1)[1][0] 
GM_mer = spice.bodvrd(str(id_mer[1]), "GM", 1)[1][0] 
GM_ven = spice.bodvrd(str(id_ven[1]), "GM", 1)[1][0] 
GM_ear = spice.bodvrd(str(id_ear[1]), "GM", 1)[1][0] 
GM_moo = spice.bodvrd(str(id_moo[1]), "GM", 1)[1][0] 
GM_mar = spice.bodvrd(str(id_mar[1]), "GM", 1)[1][0] 
GM_jup = spice.bodvrd(str(id_jup[1]), "GM", 1)[1][0] 
GM_sat = spice.bodvrd(str(id_sat[1]), "GM", 1)[1][0] 
GM_ura = spice.bodvrd(str(id_ura[1]), "GM", 1)[1][0] 
GM_nep = spice.bodvrd(str(id_nep[1]), "GM", 1)[1][0] 
GM_plu = spice.bodvrd(str(id_plu[1]), "GM", 1)[1][0] 

id_primary = id_ear[0]
id_secondary = id_moo[0]

id_bodies = np.array([id_mer[0], id_ven[0], id_ear[0], id_mar[0], id_jup[0], id_sat[0], id_ura[0], id_nep[0], id_plu[0], id_moo[0], id_sun[0]])
naif_id_bodies = np.array([id_mer[1], id_ven[1], id_ear[1], id_mar[1], id_jup[1], id_sat[1], id_ura[1], id_nep[1], id_plu[1], id_moo[1], id_sun[1]])
mu_bodies = np.array([GM_mer, GM_ven, GM_ear, GM_mar, GM_jup, GM_sat, GM_ura, GM_nep, GM_plu, GM_moo, GM_sun])

# MU = mu_bodies[id_primary] + mu_bodies[id_secondary]

# epoch_t0 = spice.str2et('23 September 2022 00:00:00 TDB')
epoch_t0 = spice.str2et('1 June 2024 00:00:00 TDB')
reference_frame = "j2000"
reference_frame_encoded = rnbp_utils.frame_encoder("J2000")
mu_p = mu_bodies[id_primary]
mu_s = mu_bodies[id_secondary]
naif_id_p = naif_id_bodies[id_primary]
naif_id_s = naif_id_bodies[id_secondary]

ode_rtol = 1e-12
ode_atol = 1e-12
tau_0 = 0.0
tau_f = 100.0
t0 = 0.0
n_points = 5000
tau_vec_input = np.linspace(tau_0, tau_f, n_points)
# tau_vec, t_vec = rnbp_utils.compute_time(tau_vec_input, t0, epoch_t0, mu_p, mu_s, naif_id_p, naif_id_s, reference_frame, ode_rtol = ode_rtol, ode_atol = ode_atol)
tau_vec, t_vec, l_vec = rnbp_utils.compute_time_distance(tau_vec_input, t0, epoch_t0, mu_p, mu_s, naif_id_p, naif_id_s, reference_frame, ode_rtol = ode_rtol, ode_atol = ode_atol)
# tau_vec_v2, t_vec_v2, l_vec = rnbp_utils.compute_time_distance(tau_vec_input, t0, epoch_t0, mu_p, mu_s, naif_id_p, naif_id_s, reference_frame, ode_rtol = ode_rtol, ode_atol = ode_atol)
# dtau_vec = tau_vec_v2 - tau_vec
# dt_vec = t_vec_v2 - t_vec
# dtau_max = max(dtau_vec.min(), dtau_vec.max(), key=abs)
# dt_max = max(dt_vec.min(), dt_vec.max(), key=abs)

pos_p = np.array([-mu, 0.0, 0.0])
pos_s = np.array([1-mu, 0.0, 0.0])

n_x = 7
n_u = 4
N = 150
Ns = N - 1

# 0: fixed final time, 1: free final time
# NOTE: i only implemented the case for free_tf = 0 so far
free_tf = 0

# model = 1 # crtbp
# model = 2 # bcrfbp
model = 3 # rnbp_rpf
# model = 4 # rnbp_rpf with fft, interpolation, and homotopy

if model == 4: 
    # Define parameters for approximation methods
    n_components_fft = 100 # not used for now since not removing high frequency components
    num_segments_piecewise = 20
    polynomial_degree = 3
    
    # Select homotopy method and extrapolation point
    sel_homotopy = 1 # 1: FFT ; 2: Piecewise ; 3: Polynomial
    if sel_homotopy == 1:
        homotopy_param = n_components_fft
    elif sel_homotopy == 2:
        homotopy_param = num_segments_piecewise
    elif sel_homotopy == 3:
        homotopy_param = polynomial_degree
    
    coeff_3bp_tmp, coeff_nbp_list, f_precomputed_list = homotopy.get_homotopy_coefficients(sel_homotopy, homotopy_param, tau_vec, t_vec, id_primary, id_secondary, mu_bodies, naif_id_bodies, observer_id, reference_frame_encoded, epoch_t0, use_jit=True)
    coeff_nbp = np.array((coeff_nbp_list), dtype=np.float64)
    f_precomputed = np.array((f_precomputed_list), dtype=np.float64)
    coeff_3bp = np.ascontiguousarray(coeff_3bp_tmp)



# node indices where maneuvers are applied; numpy array within [0, Ns]
# man_index = np.array([0, 30, 60, Ns])
# man_index = np.array([0, 50, Ns])
# man_index = np.array([0, Ns])

# initial and final boundary conditions
# x0 = np.array([8.2338046140454002e-01, 0, 1.3886061447073000e-02, 0, 1.2947638542136800e-01, 0]) # adim
# xf = np.array([1.1194488032482977e+00, 2.3502976908423845e-13, -1.1371675247773910e-02, 2.2969820490104098e-13, 1.7876223257953414e-01, -2.7620527393024119e-13]) # adim
# t0 = 0.0
# tf = 6.9083604301186052e-01

# x0 = np.array([0.870183, -0.059444, 0, -0.010471, -0.175136, 0, 0]) # adim; original
# x0 = np.array([0.870186, -0.059447, 1e-3, -0.0105, -0.17514, 1e-3, 0]) # adim, mod
x0 = np.array([0.870186, -0.059447, 5e-3, -0.0105, -0.17514, 1e-3, 0]) # adim, mod
xf = np.array([1.11559, -0.056398, 0, -0.008555, 0.157211, 0]) # adim
t0 = 0.0
tf = 12.34 / 4.34811305 * 1.91 # adim

# x0_dim = copy.deepcopy(x0)
# xf_dim = copy.deepcopy(xf)
# x0_dim[:3] *= LU
# x0_dim[3:6] *= VU
# xf_dim[:3] *= LU
# xf_dim[3:6] *= VU
# tf_dim = tf * TU
# 
# LU_t0 = np.interp(t0, tau_vec, l_vec)
# LU_tf = np.interp(tf, tau_vec, l_vec)
# TU_t0 = np.sqrt(LU_t0**3 / (mu_p + mu_s))
# TU_tf = np.sqrt(LU_tf**3 / (mu_p + mu_s))
# VU_t0 = LU_t0 / TU_t0
# VU_tf = LU_tf / TU_tf
# 
# x0[:3] = x0_dim[:3] / LU_t0
# x0[3:6] = x0_dim[3:6] / VU_t0
# xf[:3] = xf_dim[:3] / LU_tf
# xf[3:6] = xf_dim[3:6] / VU_tf
# tf = tf_dim / TU_tf

# bounds for states, controls, and dv per maneuver
states_lower = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, np.log(0.1)])
states_upper = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, np.log(1.0)])
controls_lower = np.array([-10.0, -10.0, -10.0, 0.0])
controls_upper = np.array([10.0, 10.0, 10.0, 10.0])

# n_p: number of free parameters in the optimization
# here: no free parameters
# n_man = len(man_index)
n_p = 0

tf_ind = 0

# lower and upper bounds of the free variable vector p
p_lower = np.empty(0)
p_upper = np.empty(0)

# functions for dynamics and corresponding jacobian
# jacobian_x_function = crtbp.jacobian_x
# jacobian_u_function = crtbp.jacobian_u
# dynamics_function = crtbp.dynamics

jacobian_x_function = dyn_coeff.jacobian_x
jacobian_u_function = dyn_coeff.jacobian_u
dynamics_function = dyn_coeff.dynamics

# tolerances for the integrator
# we use different values for piecewise integration (from t_k to t_k+1) and full integration (from t0 to tf)
ode_atol_piecewise = 1e-10
ode_rtol_piecewise = 1e-10
ode_atol = 1e-12
ode_rtol = 1e-12

# discretized time vector
time = np.linspace(t0, tf, N)

# trust region parameters
tr_dict = {}
tr_dict['radius'] = 100.0
tr_dict['rho'] = 100.0
tr_dict['rho0'] = 0.01
tr_dict['rho1'] = 0.2
tr_dict['rho2'] = 0.85
tr_dict['alpha'] = 1.5
tr_dict['beta'] = 1.5
tr_dict['delta'] = 1.0
tr_dict['alpha_min'] = 1.01
tr_dict['alpha_max'] = 4.0
tr_dict['beta_min'] = 1.01
tr_dict['beta_max'] = 4.0
tr_dict['radius_min'] = 1e-7
tr_dict['radius_max'] = 500.0


# -----
# some internal definitions -> probably no need to change anything here for now

x_len = N*n_x
u_len = N*n_u
t_len = 0
p_len = n_p

xu_len = x_len + u_len
sol_len = xu_len + t_len + p_len
x_ind = slice(0, x_len) 
u_ind = slice(x_len, xu_len)
p_ind = slice(xu_len, xu_len+p_len)
tf_ind_sol = xu_len + tf_ind

stm_x_len = n_x*n_x
stm_u_len = n_x*n_u
stm_t_len = n_x * free_tf
stm_const_len = n_x

V_len = n_x + stm_x_len + 2*stm_u_len + stm_t_len + stm_const_len
    
x_ind_stm = slice(0, n_x)
stm_x_ind = slice(n_x, stm_x_len+n_x)
stm_uk_ind = slice(stm_x_len+n_x, stm_x_len+n_x+stm_u_len)
stm_uk1_ind = slice(stm_x_len+n_x+stm_u_len, stm_x_len+n_x+stm_u_len+stm_u_len)
stm_t_ind = slice(stm_x_len+n_x+stm_u_len+stm_u_len, stm_x_len+n_x+stm_u_len+stm_u_len+stm_t_len)
stm_const_ind = slice(stm_x_len+n_x+stm_u_len+stm_u_len+stm_t_len, stm_x_len+n_x+stm_u_len+stm_u_len+stm_t_len+stm_const_len)

V0 = np.zeros(V_len)
V0[stm_x_ind] = np.identity(n_x).flatten()
 
x_stm = np.zeros((N,n_x))
stm_x = np.zeros((Ns,n_x*n_x))
stm_uk = np.zeros((Ns,n_x*n_u))
stm_uk1 = np.zeros((Ns,n_x*n_u))
stm_t = np.zeros((Ns,stm_t_len))
stm_const = np.zeros((Ns,stm_const_len))

if free_tf == 1:
    time = np.linspace(0.0, 1.0, N)
else:
    time = np.linspace(t0, tf, N)


scaling_dict = {'LU': LU, 'VU': VU, 'TU': TU, 'ACU': ACU, 'FU': FU, 'MU': MU}
lengths_dict = {'x_len': x_len, 'u_len': u_len, 'xu_len': xu_len, 'p_len': p_len, 't_len': t_len, 'sol_len': sol_len, 'stm_x_len': stm_x_len, 'stm_u_len': stm_u_len, 'stm_t_len': stm_t_len, 'stm_const_len': stm_const_len, 'V_len': V_len}
indices_dict = {'x_ind': x_ind, 'u_ind': u_ind, 'p_ind': p_ind, 'tf_ind': tf_ind, 'tf_ind_sol': tf_ind_sol, 'x_ind_stm': x_ind_stm, 'stm_x_ind': stm_x_ind, 'stm_uk_ind': stm_uk_ind, 'stm_uk1_ind': stm_uk1_ind, 'stm_t_ind': stm_t_ind, 'stm_const_ind': stm_const_ind}
problem_dict = {'n_x': n_x, 'n_u': n_u, 'n_p': n_p, 'N': N, 'Ns': Ns, 'free_tf': free_tf, \
    'dynamics': dynamics_function, 'jacobian_x': jacobian_x_function, 'jacobian_u': jacobian_u_function, \
    'model': model}
param_dict = {'mu': mu, 't0': t0, 'tf': tf, 'x0': x0, 'xf': xf, 'time_vec': time, \
    'g0': g0, 'Isp': Isp, 'Tmax': Tmax, 'g0_dim': g0_dim, 'Isp_dim': Isp_dim, 'Tmax_dim': Tmax_dim, \
    'ode_atol': ode_atol, 'ode_rtol': ode_rtol, 'ode_atol_piecewise': ode_atol_piecewise, 'ode_rtol_piecewise': ode_rtol_piecewise}
boundaries_dict = {'states_lower': states_lower, 'states_upper': states_upper, 'controls_lower': controls_lower, 'controls_upper': controls_upper, \
    'p_lower': p_lower, 'p_upper': p_upper}
discretization_dict = {'foh_flag': foh_flag, 'V0': V0, 'x': x_stm, 'stm_x': stm_x, 'stm_uk': stm_uk, 'stm_uk1': stm_uk1, 'stm_t': stm_t, 'stm_const': stm_const}

auxdata = {'problem': problem_dict, 'lengths': lengths_dict, 'indices': indices_dict, 'param': param_dict, 'discretization': discretization_dict, 'boundaries': boundaries_dict, \
    'scaling': scaling_dict}


auxdata["g0"] = g0
auxdata["Isp"] = Isp
auxdata["Tmax"] = Tmax
auxdata["g0_dim"] = g0_dim
auxdata["Isp_dim"] = Isp_dim
auxdata["Tmax_dim"] = Tmax_dim
auxdata["model"] = model

# for bcrfbp
auxdata["mu_crtbp"] = mu
auxdata["mu_sun"] = mu_sun
auxdata["om_sun"] = om_sun
auxdata["sun_angle_t0"] = sun_angle_t0
auxdata["a_sun"] = a_sun

# for rnbp_rpf
auxdata["id_primary"] = id_primary
auxdata["id_secondary"] = id_secondary
auxdata["mu_bodies"] = mu_bodies
auxdata["naif_id_bodies"] = naif_id_bodies
auxdata["observer_id"] = observer_id
# auxdata["reference_frame"] = reference_frame # needed if no numba / jit is used
auxdata["reference_frame"] = reference_frame_encoded # needed for numba and jit
auxdata["epoch_t0"] = epoch_t0
auxdata["tau_vec"] = tau_vec
auxdata["t_vec"] = t_vec
auxdata["l_vec"] = l_vec
auxdata["mu_p"] = mu_p
auxdata["mu_s"] = mu_s


#for homotopy
if model == 4:
    homot_param = 1.0 # Homotopy parameter (0 <= eps <= 1), would update this with a function in the future
    auxdata['coeff_3bp'] = coeff_3bp
    auxdata['coeff_nbp'] = coeff_nbp
    auxdata['f_precomputed'] = f_precomputed
    auxdata['homot_param'] = homot_param
    auxdata['sel_homotopy'] = sel_homotopy

# print("coeff_3bp.shape: ", coeff_3bp.shape)
# print("coeff_nbp.shape: ", coeff_nbp.shape)
# print("f_precomputed.shape: ", f_precomputed.shape)

# print("type coeff_3bp: ", type(coeff_3bp))
# print("type coeff_nbp: ", type(coeff_nbp))
# print("type f_precomputed: ", type(f_precomputed))

# generate initial guess 
state_guess, control_guess, p_guess = initial_guess(auxdata)

guess_dict = {}
guess_dict['state'] = state_guess
guess_dict['control'] = control_guess
guess_dict['p'] = p_guess
guess_dict['time'] = time

# plt.figure()
# ax = plt.axes(projection ='3d')
# ax.plot(state_guess[:,0], state_guess[:,1], state_guess[:,2], label='guess')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.legend()
# plt.show()

scp_data = {}

scp_data['verbose_solver'] = False

scp_data['feasibility_tol'] = 1e-7
scp_data['optimality_tol'] = 1e-4
scp_data['step_tol'] = 1e-8

scp_data['max_iterations'] = 100
scp_data['factor_nonlin'] = 10.0
scp_data['factor_lin'] = 10.0

scp_data['objective_old'] = 1e-3


solution_data = scp_solve.solve(guess_dict, scp_data, tr_dict, auxdata)

state_sol = solution_data['state']
control_sol = solution_data['control']
mass = np.exp(state_sol[:,-1])
mf_kg = mass[-1] * MU
propellant = m0_dim - mf_kg
u_mag = control_sol[:,-1]
T_mag = mass * u_mag

print("final mass in kg: ", mf_kg)
print("required propellant in kg: ", propellant)

pos_secondary = np.array([1-mu, 0.0, 0.0])
pos_sc_secondary = np.zeros((N,3))
for i in range(N):
    pos_sc_secondary[i] = state_sol[i, :3] - pos_secondary

pos_sc_secondary_norm = np.linalg.norm(pos_sc_secondary, axis=1)
dist_min = np.min(pos_sc_secondary_norm)
dist_max = np.max(pos_sc_secondary_norm)
print("min distance between spacecraft and moon (adim): ", dist_min)
print("min distance between spacecraft and moon (km): ", dist_min*LU)
print("max distance between spacecraft and moon (adim): ", dist_max)
print("max distance between spacecraft and moon (km): ", dist_max*LU)


# 3D Trajectory Plot
plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot(state_sol[:, 0], state_sol[:, 1], state_sol[:, 2], label='transfer')
ax.plot(state_guess[:, 0], state_guess[:, 1], state_guess[:, 2], label='guess')
ax.scatter(x0[0], x0[1], x0[2], marker = 's', color = [0, 0, 0], label='x0')
ax.scatter(xf[0], xf[1], xf[2], marker = 'X', color = [0, 0, 0], label='xf')
# ax.scatter(pos_p[0], pos_p[1], pos_p[2], marker = 'o', color = [0, 0.4470, 0.7410], label='earth')
ax.scatter(pos_s[0], pos_s[1], pos_s[2], marker = 'o', color = [0.5, 0.5, 0.5], label='moon')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()  # Ensure legend is within the figure

# 2D Projection (x-y)
plt.figure(2)
plt.plot(state_sol[:, 0], state_sol[:, 1], label='transfer')
plt.plot(state_guess[:, 0], state_guess[:, 1], label='guess')
plt.scatter(x0[0], x0[1], marker = 's', color = [0, 0, 0], label='x0')
plt.scatter(xf[0], xf[1], marker = 'X', color = [0, 0, 0], label='xf')
# plt.scatter(pos_p[0], pos_p[1], marker = 'o', color = [0, 0.4470, 0.7410], label='earth')
plt.scatter(pos_s[0], pos_s[1], marker = 'o', color = [0.5, 0.5, 0.5], label='moon')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()  # Legend must be added to the correct figure


if model == 1 or model == 2:
    ACU_vec = ACU
    FU_vec = FU
else:
    LU_vec, TU_vec, VU_vec, ACU_vec, FU_vec = rnbp_utils.get_scaling_param_ndim(time, tau_vec, l_vec, mu_p, mu_s, MU)

Tmag_dim = T_mag * FU_vec
umag_dim = u_mag * ACU_vec

fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots

# First subplot (Thrust Magnitude)
axes[0, 0].plot(time, T_mag, label='Tmag_adim')
axes[0, 0].set_xlabel("time, adim")
axes[0, 0].set_ylabel("T_mag, adim")
axes[0, 0].legend()
axes[0, 0].set_title("Thrust Magnitude")

# Second subplot (Control Magnitude)
axes[0, 1].plot(time, u_mag, label='umag_adim')
axes[0, 1].set_xlabel("time, adim")
axes[0, 1].set_ylabel("u_mag, adim")
axes[0, 1].legend()
axes[0, 1].set_title("Control Magnitude")

# Third subplot (Dimensional Thrust Magnitude)
axes[1, 0].plot(time, Tmag_dim, label='Tmag_dim')
axes[1, 0].set_xlabel("time, adim")
axes[1, 0].set_ylabel("T_mag, N")
axes[1, 0].legend()
axes[1, 0].set_title("Dimensional Thrust Magnitude")

# Fourth subplot (Dimensional Control Magnitude)
axes[1, 1].plot(time, umag_dim, label='umag_dim')
axes[1, 1].set_xlabel("time, adim")
axes[1, 1].set_ylabel("u_mag, km/s^2")
axes[1, 1].legend()
axes[1, 1].set_title("Dimensional Control Magnitude")

# Adjust layout to prevent overlap
plt.tight_layout()


# Show all figures
plt.show()


# plt.figure(1)
# ax = plt.axes(projection ='3d')
# ax.plot(state_sol[:,0], state_sol[:,1], state_sol[:,2], label='transfer')
# ax.plot(state_guess[:,0], state_guess[:,1], state_guess[:,2], label='guess')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.legend()
# 
# fig, ax = plt.subplots()
# ax.plot(state_sol[:,0], state_sol[:,1], label='transfer')
# ax.plot(state_guess[:,0], state_guess[:,1], label='guess')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# 
# plt.figure(2)
# plt.plot(time, u_mag)
# 
# plt.legend()
# plt.show()

    
