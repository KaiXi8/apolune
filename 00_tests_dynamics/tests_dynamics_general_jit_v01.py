import sys
import os

# Construct the full path to the directory containing the package
project_path = '/Users/hofmannc/git/apolune'

# Add the directory to sys.path
sys.path.append(project_path)

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import dynamics_coeff.dynamics_utils_jit as dyn_utils
import propagation.propagator as propagation
import spiceypy as spice
import init.load_kernels as krn
import matplotlib.pyplot as plt
import dynamics_coeff.rnbp_rpf_utils as rnbp_utils
import dynamics_coeff.rnbp_rpf_dynamics_nonuniform_jit as rnbp_dyn
import copy

krn.load_kernels()

G = 6.67408e-20  # [km^3 kg^−1 s^−2]
# AU = 149597870.693 # km
AU = 1.495978706136889e+08 # [km] from SPICE

# Attractors and their gravitational parameters
# spice.furnsh("kernels/pck/pck00010.tpc")  

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

# id_primary = id_sun[0]
# id_secondary = id_ear[0]

id_bodies = np.array([id_mer[0], id_ven[0], id_ear[0], id_mar[0], id_jup[0], id_sat[0], id_ura[0], id_nep[0], id_plu[0], id_moo[0], id_sun[0]])
naif_id_bodies = np.array([id_mer[1], id_ven[1], id_ear[1], id_mar[1], id_jup[1], id_sat[1], id_ura[1], id_nep[1], id_plu[1], id_moo[1], id_sun[1]])
mu_bodies = np.array([GM_mer, GM_ven, GM_ear, GM_mar, GM_jup, GM_sat, GM_ura, GM_nep, GM_plu, GM_moo, GM_sun])

MU = mu_bodies[id_primary] + mu_bodies[id_secondary]
# https://www.jpl.nasa.gov/_edu/pdfs/scaless_reference.pdf
# https://www.jpl.nasa.gov/_edu/pdfs/ssbeads_answerkey.pdf
if id_primary == id_ear[0] and id_secondary == id_moo[0]:
    LU = 384400 # [km]
elif id_primary == id_sun[0] and id_secondary == id_ear[0]:
    LU = 1*AU 
elif id_primary == id_sun[0] and id_secondary == id_jup[0]:
    LU = 5.2*AU
else:
    raise ValueError("Invalid primary and secondary body combination")

TU = np.sqrt(LU**3 / MU) # scaling factor for time [s]
VU = LU / TU # [km/s]
om0 = 1/TU # constant [1/s] 

# mu_crtbp = 1.21506683e-2
# mu_sun = 3.28900541e5
mu_crtbp = mu_bodies[id_secondary] / MU
mu_sun = mu_bodies[id_sun[0]] / MU

sun_angle_t0 = 0.671
om_sun = -9.25195985e-1
a_sun = 3.88811143e2;

LU = 3.84405000e5 # km
TU = 4.34811305*86400 # seconds
VU = 1.02323281 # km/s

auxdata = {}
model = 1
auxdata = {"model": model, "mu_crtbp": mu_crtbp, "mu_sun": mu_sun, "a_sun": a_sun, "om_sun": om_sun, "sun_angle_t0": sun_angle_t0}

tau = 1.2345
state_syn = np.array([0.8741, 0.563, -0.265, 1.189, -0.486, 0.567])
control = 0.0
p_param = 0.0

epoch_t0 = spice.str2et('23 September 2022 00:00:00 TDB')
reference_frame = "j2000"
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
tau_vec, t_vec = rnbp_utils.compute_time(tau_vec_input, t0, epoch_t0, mu_p, mu_s, naif_id_p, naif_id_s, reference_frame, ode_rtol = ode_rtol, ode_atol = ode_atol)

auxdata["id_primary"] = id_primary
auxdata["id_secondary"] = id_secondary
auxdata["mu_bodies"] = mu_bodies
auxdata["naif_id_bodies"] = naif_id_bodies
auxdata["observer_id"] = 0
auxdata["reference_frame"] = reference_frame
auxdata["epoch_t0"] = epoch_t0
auxdata["tau_vec"] = tau_vec_input
auxdata["t_vec"] = t_vec

naif_id_bodies = np.array([naif_id_bodies[id_primary], naif_id_bodies[id_secondary], naif_id_bodies[id_sun[0]]])
num_bodies = len(naif_id_bodies)
observer_id = 0
ref_frame = rnbp_utils.frame_encoder("J2000")

auxdata["model"] = 3
rho_p = np.array([
    [1.28, 0.71, -0.78],
    [0.673, 0.871, -0.567],
    [2.456, 0.514, 0.1234]
    ])
mu_bodies_adim = np.array([mu_p / MU, mu_s / MU, mu_bodies[id_sun[0]] / MU])
rho = state_syn[:3]
pos_bodies = np.zeros((num_bodies,3))
state_bodies = np.zeros((num_bodies,6))
lt_ob = np.zeros(1)
t = 1.2 * epoch_t0

omega = dyn_utils.compute_omega(rho, rho_p, mu_bodies_adim)
# print("omega: ", omega)

grad_omega = dyn_utils.compute_grad_omega(rho, rho_p, mu_bodies_adim)
# print("grad_omega: ", grad_omega)

jac_grad_omega = dyn_utils.compute_jac_grad_omega(rho, rho_p, mu_bodies_adim)
# print("jac_grad_omega: ", jac_grad_omega)

r_ob = np.empty((3,)) # dummy array for the cspice output
lt_ob = np.empty((1,)) # dummy array for the cspice output
e_t = 1e7 # Ephemeris time (seconds past J2000)

# rnbp_dyn.get_body_positions(t, naif_id_bodies, observer_id, ref_frame, pos_bodies, lt_ob)
pos_bod = rnbp_dyn.get_body_positions(t, naif_id_bodies, observer_id, ref_frame)
# print("pos_bodies: ", pos_bodies)
# print("pos_bod: ", pos_bod)

# rnbp_dyn.get_body_states(t, naif_id_bodies, observer_id, ref_frame, state_bodies, lt_ob)
state_bod = rnbp_dyn.get_body_states(t, naif_id_bodies, observer_id, ref_frame)
# print("state_bodies: ", state_bodies)
# print("state_bod: ", state_bod)

tau = 5.678
time_int = rnbp_dyn.compute_epoch_time(tau, tau_vec, t_vec, epoch_t0)
# print("time_int: ", time_int)

pos_bd = state_bod[:,0:3].copy() # [km]
vel_bd = state_bod[:,3:6].copy() # [km/s]
mu_bod_dim = mu_bodies_adim.copy() * MU
res = rnbp_dyn.frame_rnbprpf_coeff(id_primary, id_secondary, pos_bd, vel_bd, mu_bod_dim)
a, j, h, h1d, b, b2d, k, k1d, k2d, C = res
# print(f"a = {a}, j = {j}")
# print(f"b = {b}, C = {C}")

res2 = rnbp_dyn.frame_rnbprpf(id_primary, id_secondary, pos_bd, vel_bd, mu_bod_dim)
b_2, b2d_2, k_2, k1d_2, k2d_2, C_2, C1d_2, C2d_2 = res2
# print(f"b_2 = {b_2}, C_2 = {C_2}")

rho_bod = rnbp_dyn.inertialToSynodic_pos(pos_bod, b, C, k)
# print("rho_bod: ", rho_bod)

distance = 1e5
distance_prime = 1e2
tau_prime = rnbp_dyn.compute_tau_prime(distance, mu_p, mu_s)
tau_double_prime = rnbp_dyn.compute_tau_double_prime(distance, distance_prime, mu_p, mu_s)

grad_om = rnbp_dyn.grad_omega(rho, pos_bod, b, C, k, mu_bodies_adim)
jac_grad_om = rnbp_dyn.jac_grad_omega(rho, pos_bod, b, C, k, mu_bodies_adim)


coeff_grad_sol = rnbp_dyn.compute_coeff_grad(tau, state_syn, id_primary, id_secondary, mu_bod_dim, naif_id_bodies, observer_id, ref_frame, epoch_t0, tau_vec, t_vec)
coeff_jac_grad_sol = rnbp_dyn.compute_coeff_jac_grad(tau, state_syn, id_primary, id_secondary, mu_bod_dim, naif_id_bodies, observer_id, ref_frame, epoch_t0, tau_vec, t_vec)

# dyn_rnbp_rpf_coeff = dyn.dynamics(tau, state_syn, control, p_param, auxdata)
# jac_rnbp_rpf_coeff = dyn.jacobian(tau, state_syn, control, p_param, auxdata)


