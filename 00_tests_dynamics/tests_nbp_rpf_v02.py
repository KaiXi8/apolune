import sys
import os

# Construct the full path to the directory containing the package
project_path = '/Users/hofmannc/git/apolune'

# Add the directory to sys.path
sys.path.append(project_path)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import dynamics.rnbp_rpf_dynamics as rnp_rpf
import dynamics.rnbp_rpf_dynamics_coeff as rnp_rpf_coeff
import propagation.propagator as propagation
import spiceypy as spice
import init.load_kernels as krn
import matplotlib.pyplot as plt
from scipy import interpolate
import time as tm

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

# id_primary = id_sun[0]
# id_secondary = id_ear[0]
id_primary = id_ear[0]
id_secondary = id_moo[0]

id_bodies = [id_mer[0], id_ven[0], id_ear[0], id_mar[0], id_jup[0], id_sat[0], id_ura[0], id_nep[0], id_plu[0], id_moo[0], id_sun[0]]
naif_id_bodies = [id_mer[1], id_ven[1], id_ear[1], id_mar[1], id_jup[1], id_sat[1], id_ura[1], id_nep[1], id_plu[1], id_moo[1], id_sun[1]]
mu_bodies = np.array([GM_mer, GM_ven, GM_ear, GM_mar, GM_jup, GM_sat, GM_ura, GM_nep, GM_plu, GM_moo, GM_sun])

MU = mu_bodies[id_primary] + mu_bodies[id_secondary]
# https://www.jpl.nasa.gov/_edu/pdfs/scaless_reference.pdf
# https://www.jpl.nasa.gov/_edu/pdfs/ssbeads_answerkey.pdf
if id_primary == id_ear[0] and id_secondary == id_moo[0]:
    LU = 384400 # [km]
elif id_primary == id_sun[0] and id_secondary == id_ear[0]:
    LU = 1*AU 
else:
    raise ValueError("Invalid primary and secondary body combination")


TU = np.sqrt(LU**3 / MU) # scaling factor for time [s]
VU = LU / TU # [km/s]
om0 = 1/TU # constant [1/s] 

# [km^3 s−2] to [adim]
# mu_bodies = mu_bodies / VU**2 / LU 
# mu_bodies = mu_bodies * TU**2 / LU**3

inertial_frame = "J2000"
origin_frame = "SBB"
abcorr_flag = "None"
et0 = spice.str2et('1 Jan 2017 00:00:00 TDB')

scaling = {"LU": LU, "TU": TU, "VU": VU}

param = {"om0": om0, "et0": et0, "naif_id_bodies": naif_id_bodies, "mu_bodies": mu_bodies, \
    "id_primary": id_primary, "id_secondary": id_secondary, "inertial_frame": inertial_frame, \
    "abcorr_flag": abcorr_flag, "origin_frame": origin_frame}
# rnbprpf_dict["param"]["om0"] = om0
# rnbprpf_dict["param"]["et0"] = et0 # for non-dimensional epoch: et0/TU
# rnbprpf_dict["param"]["naif_id_bodies"] = naif_id_bodies
# rnbprpf_dict["param"]["mu_bodies"] = mu_bodies
# rnbprpf_dict["param"]["id_primary"] = id_primary
# rnbprpf_dict["param"]["id_secondary"] = id_secondary
# rnbprpf_dict["scaling"]["LU"] = LU
# rnbprpf_dict["scaling"]["VU"] = VU
# rnbprpf_dict["scaling"]["TU"] = TU
# rnbprpf_dict["param"]["inertial_frame"] = "J2000" # J2000
# rnbprpf_dict["param"]["abcorr_flag"] = "None"
# rnbprpf_dict["param"]["origin_frame"] = "SSB" # SSB - 0
# # rnbprpf_dict["origin_frame"] = "399" # SSB - 0

rnbprpf_dict = {"param": param, "scaling": scaling}

state_dyn = np.array([0.8741, 0.563, -0.265, 1.189, -0.486, 0.567])
tau = 1.234
# state_dot_dyn = rnbprpf_dynamics(tau, state_dyn, rnbprpf_dict)

time = 86400 * np.array([10.0, 20.0]) / TU
states_syn = np.array([ [0.8741, 0.563, -0.265, 1.189, -0.486, 0.567],
             [ -0.784, 1.245, 0.841, 0.235, 0.654, -0.312] ])
# states_syn2in = synodicToInertial(time, states_syn, rnbprpf_dict)
# states_in2syn = inertialToSynodic(time, states_syn2in, rnbprpf_dict)
# print("states_syn2in: ", states_syn2in)
# 
t1 = 86400 * 10.0 / TU
state_syn = np.array([0.8741, 0.563, -0.265, 1.189, -0.486, 0.567])
# state_syn2in = synodicToInertial(t1, state_syn, rnbprpf_dict)
# states_in2syn = inertialToSynodic(t1, state_syn2in, rnbprpf_dict)
# 
# print("state_syn2in: ", state_syn2in)
# print("states_in2syn: ", states_in2syn)

# print(TU)
# print(6.9083604301186052e-01 * 86400 / TU)

abcorr_flag = "None"
ref_frame = "J2000"
id_primary = id_ear[0]
id_secondary = id_moo[0]
mu_p = mu_bodies[id_primary]
mu_s = mu_bodies[id_secondary]
naif_id_p = naif_id_bodies[id_primary]
naif_id_s = naif_id_bodies[id_secondary]
et0 = spice.str2et('1 Jan 2017 00:00:00 TDB')
et1 = spice.str2et('30 June 2017 00:00:00 TDB')
et2 = spice.str2et('23 September 2022 00:00:00 TDB')
et3 = spice.str2et('23 September 2023 00:00:00 TDB')
t_vec_ = np.linspace(et2, et3, 365)
t = et2
dist = rnp_rpf_coeff.get_distance(t_vec_, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
# dist = rnp_rpf_coeff.get_distance(t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
# print(dist)
# plt.plot(t_vec_, dist)
# plt.show()


epoch_time = et2

tau_0 = 0.0
tau_f = 100.0
t0 = 0.0
n_points = 5000
# n_points = 10
# tau_vec, t_vec = rnp_rpf_coeff.compute_times(tau_0, tau_f, t0, n_points, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
tau_vec, t_vec = rnp_rpf_coeff.compute_times(tau_0, tau_f, t0, epoch_time, n_points, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
# t_vec_days = t_vec / 86400
t_vec_days = t_vec / 86400

_, _, l_vec = rnp_rpf_coeff.compute_times_distances(tau_0, tau_f, t0, epoch_time, n_points, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)


t_int = rnp_rpf_coeff.compute_time(tau_0, tau_f, t0, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
t_int_days = t_int / 86400

print("tau_vec.shape: ", tau_vec.shape)
print("t_vec.shape: ", t_vec.shape)
print("TU, days: ", TU / 86400)
print("tau_0: ", tau_vec[0])
print("tau_f: ", tau_vec[-1])
print("t_0, days: ", t_vec_days[0])
print("t_f, days: ", t_vec_days[-1])
print("t_int, days: ", t_int_days)

tau = 53.678
# tau = 25
t_int_tau = rnp_rpf_coeff.compute_time(tau_0, tau, t0, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
t_int_tau_s = t_int_tau
t_int_tau_days = t_int_tau / 86400

t_int_tau_interp = np.interp(tau, tau_vec, t_vec)
t_int_tau_interp_days = np.interp(tau, tau_vec, t_vec/86400)
# t_int_tau_interp_days = (t_int_tau_interp - t0) / 86400

f_interp = interpolate.interp1d(tau_vec, t_vec, kind = 'cubic')
tau_scipy_interp = f_interp(tau) / 86400

# t_int_tau_interp_days = t_int_tau_interp / 86400
print("t_int_tau, s: ", t_int_tau_s)
print("t_int_tau_interp, s: ", t_int_tau_interp)
print("t_int_tau, days: ", t_int_tau_days)
print("t_int_tau_interp, days: ", t_int_tau_interp_days)
# print(t_vec_days[499])
print("tau_scipy_interp, days: ", tau_scipy_interp)
print("difference for s, days: ", np.abs(t_int_tau_interp - t_int_tau_s)/86400)
print("difference for days, days: ", np.abs(t_int_tau_days - t_int_tau_interp_days))
print("difference for s, s: ", np.abs(t_int_tau_interp - t_int_tau_s))

tau_prime = rnp_rpf_coeff.compute_tau_prime(l_vec, mu_p, mu_s)
# l_vec_interp = np.interp(tau_vec, tau_vec, l_vec)
l_vec_interp = np.interp(tau_vec, tau_vec, l_vec)
t_prime_interp = rnp_rpf_coeff.compute_tau_prime(l_vec_interp, mu_p, mu_s)

# plt.plot(tau_vec, l_vec, label='int')
# plt.plot(tau_vec, l_vec_interp, label='interp')
# plt.legend()
# plt.show()


# tau_comp = rnp_rpf_coeff.compute_tau(t_vec, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
tau_comp = rnp_rpf_coeff.compute_tau_v2(t_vec, l_vec, mu_p, mu_s)

# print("t_vec[:5]: ", t_vec[:5])
# print("tau_vec[:5]: ", tau_vec[:5])
# print("tau_vec[-5:]: ", tau_vec[-5:])
# print("tau_comp[:5]: ", tau_comp[:5])
# print("tau_comp[-5:]: ", tau_comp[-5:])
# print("tau_vec: ", tau_vec)
# print("tau_comp: ", tau_comp)
# print("l in compute_times_distances: ", l_vec)


tau_0 = 0.0
tau_vec_comp_int = rnp_rpf_coeff.compute_tau_int(t_vec, tau_0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
# print("tau_vec_comp_int: ", tau_vec_comp_int)


tau_interp = np.interp(t_vec, t_vec, tau_vec)
# print("tau_vec[:5]: ", tau_vec[:5])
# print("tau_vec[-5:]: ", tau_vec[-5:])
# print("tau_interp[:5]: ", tau_interp[:5])
# print("tau_interp[-5:]: ", tau_interp[-5:])

TU_vec = np.sqrt(l_vec**3 / (mu_p + mu_s))
TU_min = np.min(TU_vec)
TU_max = np.max(TU_vec)
print("TU_min: ", TU_min)
print("TU_max: ", TU_max)

print("error in seconds: ", 1e-2*TU_max)
# print("error in days: ", 1e-2*TU_max)


"""
 coefficients tests
"""

b13 = (mu_p + mu_s) / (l_vec_interp**3 * t_prime_interp**2)
print("min b13: ", np.min(b13))
print("max b13: ", np.max(b13))

# plt.plot(tau_vec, t_prime_interp, label='int')
# plt.plot(tau_vec, b13, label='b13')
# plt.show()

rnbprpf_dict["param"]["om0"] = om0
rnbprpf_dict["param"]["et0"] = t0
rnbprpf_dict["param"]["naif_id_bodies"] = naif_id_bodies
rnbprpf_dict["param"]["mu_bodies"] = mu_bodies
rnbprpf_dict["param"]["id_primary"] = id_primary
rnbprpf_dict["param"]["id_secondary"] = id_secondary
rnbprpf_dict["scaling"]["LU"] = LU
rnbprpf_dict["scaling"]["VU"] = VU
rnbprpf_dict["scaling"]["TU"] = TU
rnbprpf_dict["param"]["inertial_frame"] = "J2000" # J2000
rnbprpf_dict["param"]["abcorr_flag"] = "None"
rnbprpf_dict["param"]["origin_frame"] = "SSB" # SSB - 0

rnbprpf_dict["param"]["tau_vec"] = tau_vec
rnbprpf_dict["param"]["t_vec"] = t_vec
rnbprpf_dict["param"]["distance_vec"] = l_vec

# gna = rnp_rpf_coeff.dynamics(tau, state_dyn, 0, 0, rnbprpf_dict)

n = 100
start_time = tm.perf_counter()
for i in range(n):
    drho_eom, deta_eom = rnp_rpf_coeff.dynamics_coeff_eom(tau, state_dyn, 0, 0, rnbprpf_dict)
end_time = tm.perf_counter()
print(end_time - start_time)

start_time = tm.perf_counter()
for i in range(n):
    drho_table, deta_table = rnp_rpf_coeff.dynamics_coeff_table(tau, state_dyn, 0, 0, rnbprpf_dict)
end_time = tm.perf_counter()
print(end_time - start_time)

# diff_drho = drho_eom - drho_table
# diff_deta = deta_eom - deta_table
# print("diff_drho, adim: ", diff_drho)
# print("diff_deta, adim: ", diff_deta)



