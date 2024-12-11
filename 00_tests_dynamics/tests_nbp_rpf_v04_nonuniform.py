import sys
import os

# Construct the full path to the directory containing the package
project_path = '/Users/hofmannc/git/apolune'

# Add the directory to sys.path
sys.path.append(project_path)

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# import dynamics.rnbp_rpf_dynamics_uniform as rnp_rpf
import dynamics.rnbp_rpf_dynamics_nonuniform as rnp_rpf
import propagation.propagator as propagation
# import frames.rnbp_rpf_transformations_uniform as rnbp_rpf_transformations
import frames.rnbp_rpf_transformations_nonuniform as rnbp_rpf_transformations
import spiceypy as spice
import init.load_kernels as krn
import matplotlib.pyplot as plt
from scipy import interpolate
import time as tm
import copy
from scipy.optimize import approx_fprime

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

# Selection

# 1: Atira ; 2: Aten ; 3: Apollo ; 4: Amor ; 5: Eureka ; 6: Einstein ; 7: Fortuna ; 8: Hermione ; 9: Hektor ; 10: Damocles ; 11: Chaos ; 12: Ceres
sel_state = 2
# 1: short ; 2: medium ; 3: long
sel_time = 1 
sel_points = int(1e3) # number of points for the integration

id_primary = id_sun[0]
id_secondary = id_jup[0]

# id_primary = id_ear[0]
# id_secondary = id_moo[0]

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
elif id_primary == id_sun[0] and id_secondary == id_jup[0]:
    LU = 5.2*AU
else:
    raise ValueError("Invalid primary and secondary body combination")


TU = np.sqrt(LU**3 / MU) # scaling factor for time [s]
VU = LU / TU # [km/s]
om0 = 1/TU # constant [1/s] 

# [km^3 s−2] to [adim]
# mu_bodies = mu_bodies / VU**2 / LU 
# mu_bodies = mu_bodies * TU**2 / LU**3

inertial_frame = "J2000"
origin_frame = "SSB"
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
t_vec = np.linspace(et2, et3, 365)
t = et2
dist = rnp_rpf.get_body_distance(t_vec, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
# dist = rnp_rpf.get_body_distance(t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
# print(dist)
# plt.plot(t_vec, dist)
# plt.show()

ode_rtol = 1e-12
ode_atol = 1e-12

tau_0 = 0.0
tau_f = 100.0
t0 = 0.0
epoch_time = et2
n_points = 1000
tau_vec_input = np.linspace(tau_0, tau_f, n_points)
tau_vec, t_vec = rnp_rpf.compute_time(tau_vec_input, t0, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)
t_vec_days = t_vec / 86400

_, _, l_vec = rnp_rpf.compute_time_distance(tau_vec_input, t0, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)

tau_int, t_int = rnp_rpf.compute_time([tau_0, tau_f], t0, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)
t_int_days = t_int / 86400

# # print("tau_vec.shape: ", tau_vec.shape)
# # print("t_vec.shape: ", t_vec.shape)
# # print("TU, days: ", TU / 86400)
# print("tau_0: ", tau_vec[0])
# print("tau_f: ", tau_vec[-1])
# print("t_0, days: ", t_vec_days[0])
# print("t_f, days: ", t_vec_days[-1])
# print("t_int[-1], days: ", t_int_days[-1])
# print("tau_int[-1]: ", tau_int[-1])

tau = 53.678
# tau = 25
_, t_int_tau = rnp_rpf.compute_time([tau_0, tau], t0, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)
t_int_tau_s = t_int_tau[-1]
t_int_tau_days = t_int_tau_s / 86400

t_int_tau_interp = np.interp(tau, tau_vec, t_vec)
t_int_tau_interp_days = np.interp(tau, tau_vec, t_vec / 86400)

f_interp = interpolate.interp1d(tau_vec, t_vec, kind = 'cubic')
tau_scipy_interp = f_interp(tau) / 86400

# print("t_int_tau, s: ", t_int_tau_s)
# print("t_int_tau_interp, s: ", t_int_tau_interp)
# print("t_int_tau, days: ", t_int_tau_days)
# print("t_int_tau_interp, days: ", t_int_tau_interp_days)
# print("tau_scipy_interp, days: ", tau_scipy_interp)
print("difference for s, days: ", np.abs(t_int_tau_interp - t_int_tau_s)/86400)
print("difference for days, days: ", np.abs(t_int_tau_days - t_int_tau_interp_days))
print("difference for s, s: ", np.abs(t_int_tau_interp - t_int_tau_s))



param["t_vec"] = t_vec
param["tau_vec"] = tau_vec
param["et0"] = et2


time_in = np.array([123.543, 297.519]) * 86400
tau_syn = np.interp(time_in, t_vec, tau_vec)
states_syn = np.array([ [0.8741, 0.563, -0.265, 1.189, -0.486, 0.567],
             [ -0.784, 1.245, 0.841, 0.235, 0.654, -0.312] ])
states_syn2in = rnbp_rpf_transformations.synodicToInertial(tau_syn, states_syn, param)
states_in2syn = rnbp_rpf_transformations.inertialToSynodic(time_in, states_syn2in, param)
states_syn2in_2 = rnbp_rpf_transformations.synodicToInertial(tau_syn, states_in2syn, param)
diff_state_syn = states_syn - states_in2syn
diff_state_in = states_syn2in - states_syn2in_2
# print("states_syn2in: ", states_syn2in)
# print("states_in2syn: ", states_in2syn)
print("diff_state_syn: ", diff_state_syn)
print("diff_state_in: ", diff_state_in)

# print("t_vec[0]: ", t_vec[0])
# print("t_vec[-1]: ", t_vec[-1])

t0 = 0.0
epoch_time = et2
t_vec_v2, tau_vec_v2 = rnp_rpf.compute_tau(t_vec, tau_0, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)
# print("tau_vec_v2[0]: ", tau_vec_v2[0])
# print("tau_vec_v2[-1]: ", tau_vec_v2[-1])
# print("t_vec_v2[0]: ", t_vec_v2[0])
# print("t_vec_v2[-1]: ", t_vec_v2[-1])


t_vec_v3, tau_vec_v3 = rnp_rpf.compute_tau([t_vec[0], t_vec[-1]], tau_0, epoch_time, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)
# print("tau_vec_v3[0]: ", tau_vec_v3[0])
# print("tau_vec_v3[-1]: ", tau_vec_v3[-1])
# print("t_vec_v3[0]: ", t_vec_v3[0])
# print("t_vec_v3[-1]: ", t_vec_v3[-1])
# print("len t_vec_v3: ", len(t_vec_v3))





sel_time = 1

ode_rtol = 2.5e-14
ode_atol = 2.5e-14

id_primary = id_sun[0]
id_secondary = id_jup[0]
param["id_primary"] = id_primary
param["id_secondary"] = id_secondary
mu_p = mu_bodies[id_primary]
mu_s = mu_bodies[id_secondary]
naif_id_p = naif_id_bodies[id_primary]
naif_id_s = naif_id_bodies[id_secondary]

# Short, medium and long term epochs
ep0_s = "31-Aug-2016 00:00:00"
epf_s = "31-Aug-2031 00:00:00"
ep0_0 = "31-Aug-2016 00:00:00"
epf_0 = "31-Aug-2018 00:00:00"
ep0_m = "31-Aug-2016 00:00:00"
epf_m = "31-Aug-2091 00:00:00"
ep0_l = "01-Jan-1900 00:00:00"
epf_l = "02-Jan-2100 00:00:00"
ep0_4 = "31-Aug-2016 00:00:00"
epf_4 = "31-Aug-2081 00:00:00"

if sel_time == 0:
    ep0 = spice.str2et(ep0_0)
    epf = spice.str2et(epf_0)
    num_years = 2
elif sel_time == 1:
    ep0 = spice.str2et(ep0_s)
    epf = spice.str2et(epf_s)
    num_years = 15
elif sel_time == 2:
    ep0 = spice.str2et(ep0_m)
    epf = spice.str2et(epf_m)
    num_years = 75
elif sel_time == 3:
    ep0 = spice.str2et(ep0_l)
    epf = spice.str2et(epf_l)
    num_years = 200
elif sel_time == 4:
    ep0 = spice.str2et(ep0_4)
    epf = spice.str2et(epf_4)
    num_years = 65
    
tau0 = 0.0
t0 = 0.0
n_points_per_year = 5000
t_vec = np.linspace(t0, epf - ep0, n_points_per_year*num_years)
t_vec, tau_vec, distance_vec = rnp_rpf.compute_tau_distance(t_vec, tau0, ep0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)

param["t_vec"] = t_vec
param["tau_vec"] = tau_vec
param["distance_vec"] = distance_vec
param["et0"] = ep0

# print(f"t_vec[0] = {t_vec[0]} s, t_vec[-1] = {t_vec[-1]} s, t_vec[-1] = {t_vec[-1]/86400} days, t_vec[-1] = {t_vec[-1]/86400/365} years")
# print(f"tau_vec[0] = {tau_vec[0]}, tau_vec[-1] = {tau_vec[-1]}")

krn.load_rpf_validation_kernels()

rnbprpf_dict = param

# full state = [position; velocity] ; 1900-01-01 ; 2101-01-01
if sel_state == 1:
    state0_in, _ = spice.spkezr('20163693', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Atira"
elif sel_state == 2:
    state0_in, _ = spice.spkezr('20002062', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Aten"
elif sel_state == 3:
    state0_in, _ = spice.spkezr('20001862', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Apollo"
elif sel_state == 4:
    state0_in, _ = spice.spkezr('20001221', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Amor"
elif sel_state == 5:
    state0_in, _ = spice.spkezr('20005261', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Eureka"
elif sel_state == 6:
    state0_in, _ = spice.spkezr('20002001', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Einstein"
elif sel_state == 7:
    state0_in, _ = spice.spkezr('20000019', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Fortuna"
elif sel_state == 8:
    state0_in, _ = spice.spkezr('20000121', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Hermione"
elif sel_state == 9:
    state0_in, _ = spice.spkezr('20000624', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Hektor"
elif sel_state == 10:
    state0_in, _ = spice.spkezr('20005335', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Damocles"
elif sel_state == 11:
    state0_in, _ = spice.spkezr('20019521', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Chaos"
elif sel_state == 12:
    state0_in, _ = spice.spkezr('20000001', ep0, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
    body_name = "Ceres"
else:
    body_name = "Unknown"
    


# Propagate in RPF 
tt = np.vstack((0,0))
SV_0 = np.vstack((state0_in,state0_in))
XF_0 = rnbp_rpf_transformations.inertialToSynodic(tt, SV_0, rnbprpf_dict)
tau_linspace = np.linspace(0, tau_vec[-1], sel_points)



def propagate_ode(ode_fun, x0, tvec, aux, ode_rtol, ode_atol):
    sol = odeint(ode_fun, x0, tvec, args=(0.0, 0.0, aux,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
    return sol
def propagate_ivp(ode_fun, x0, tvec, aux, ode_rtol, ode_atol):
    t_span = (tvec[0], tvec[-1])
    sol = solve_ivp(ode_fun, t_span, x0, t_eval=tvec, args=(0.0, 0.0, aux,), method='DOP853', rtol=ode_rtol, atol=ode_atol)
    return sol.y.T

param_copy = copy.deepcopy(param)
rnbprpf_dict_v2 = {}
rnbprpf_dict_v2 = {'param': param_copy}

# print(f"XF_0[0]: {XF_0[0]}")
# print(f"tau_linspace[:5]: {tau_linspace[:5]}")
# print(f"tau_linspace[-5:]: {tau_linspace[-5:]}")

# ode_rtol = 2.5e-14
# ode_atol = 2.5e-14

start_time = tm.perf_counter()
# sol_prop_syn = propagate_ode(rnp_rpf.dynamics_coeff_eom, XF_0[0], tau_linspace, rnbprpf_dict_v2, ode_rtol, ode_atol)
sol_prop_syn = propagate_ode(rnp_rpf.dynamics_coeff_table, XF_0[0], tau_linspace, rnbprpf_dict_v2, ode_rtol, ode_atol)

# sol_prop_syn = propagate_ode(rnbprpf_dynamics, XF_0[0], tau_linspace, rnbprpf_dict)
# sol_prop_syn = propagate_ivp(rnp_rpf.dynamics_coeff_eom, XF_0[0], tau_linspace, rnbprpf_dict_v2, ode_rtol, ode_atol)
propagation_time = tm.perf_counter() - start_time
print(f"Propagation time: {propagation_time:.4f} seconds")

# Convert epochs to ephemeris time (ET)
et_linspace = np.interp(tau_linspace, tau_vec, t_vec)
et_linspace_spice = et_linspace + ep0


# utc_epochs = [spice.et2utc(et, "C", 3) for et in (ep0 + tau_linspace / om0)]
start_time = tm.perf_counter()
if sel_state == 1:
    state0_in, _ = spice.spkezr('20163693', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 2:
    state0_in, _ = spice.spkezr('20002062', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 3:
    state0_in, _ = spice.spkezr('20001862', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 4:
    state0_in, _ = spice.spkezr('20001221', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 5:
    state0_in, _ = spice.spkezr('20005261', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 6:
    state0_in, _ = spice.spkezr('20002001', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 7:
    state0_in, _ = spice.spkezr('20000019', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 8:
    state0_in, _ = spice.spkezr('20000121', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 9:
    state0_in, _ = spice.spkezr('20000624', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 10:
    state0_in, _ = spice.spkezr('20005335', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 11:
    state0_in, _ = spice.spkezr('20019521', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])
elif sel_state == 12:
    state0_in, _ = spice.spkezr('20000001', et_linspace_spice, rnbprpf_dict["inertial_frame"], rnbprpf_dict["abcorr_flag"], rnbprpf_dict["origin_frame"])


sol_spice_in = np.array(state0_in).reshape(-1,6)
spice_time = tm.perf_counter() - start_time
print(f"Spice time: {spice_time:.4f} seconds")

start_time = tm.perf_counter()
sol_spice_syn = rnbp_rpf_transformations.inertialToSynodic(et_linspace, sol_spice_in, rnbprpf_dict)
transf_time = tm.perf_counter() - start_time
print(f"Transformation - Inert2Syn time: {transf_time:.4f} seconds")

start_time = tm.perf_counter()
sol_prop_in = rnbp_rpf_transformations.synodicToInertial(tau_linspace, sol_prop_syn, rnbprpf_dict)
transf_time = tm.perf_counter() - start_time
print(f"Transformation - Syn2Inert time: {transf_time:.4f} seconds")


print(f"sol_spice_syn[0] = {sol_spice_syn[0]}, sol_spice_syn[-1] = {sol_spice_syn[-1]}")
print(f"sol_prop_syn[0] = {sol_prop_syn[0]}, sol_prop_syn[-1] = {sol_prop_syn[-1]}")



# Calculate the absolute and relative error in Inertial Frame
abs_err_pos = np.linalg.norm(sol_prop_in[:,:3] - sol_spice_in[:,:3], axis=1)
abs_err_vel = np.linalg.norm(sol_prop_in[:,3:] - sol_spice_in[:,3:], axis=1)
rel_err_pos = abs_err_pos/np.linalg.norm(sol_spice_in[:,:3], axis=1)
rel_err_vel = abs_err_vel/np.linalg.norm(sol_spice_in[:,3:], axis=1)

print(f"max abs_err_pos: {np.max(abs_err_pos)}, max abs_err_vel: {np.max(abs_err_vel)}")
print(f"max rel_err_pos: {np.max(rel_err_pos)}, max rel_err_vel: {np.max(rel_err_vel)}")


print(f"Selected body: {body_name}")

# fig, axes = plt.subplots(2, 2, figsize=(10, 5)) 
# # Plot 1: Absolute Error
# axes[0,0].plot(et_linspace, abs_err_pos)
# axes[0,0].set_ylabel("Error")
# axes[0,0].set_xlabel("Ephemeris Time [s]")
# axes[0,0].set_title("Absolute Error - Pos.")
# axes[0,0].grid(True)
# # Plot 2: Relative Error
# axes[0,1].plot(et_linspace, rel_err_pos)
# axes[0,1].set_ylabel("Error")
# axes[0,1].set_xlabel("Ephemeris Time [s]")
# axes[0,1].set_title("Relative Error - Pos.")
# axes[0,1].grid(True)
# # Plot 3: Absolute Error
# axes[1,0].plot(et_linspace, abs_err_vel)
# axes[1,0].set_ylabel("Error")
# axes[1,0].set_xlabel("Ephemeris Time [s]")
# axes[1,0].set_title("Absolute Error - Vel.")
# axes[1,0].grid(True)
# # Plot 4: Relative Error
# axes[1,1].plot(et_linspace, rel_err_vel)
# axes[1,1].set_ylabel("Error")
# axes[1,1].set_xlabel("Ephemeris Time [s]")
# axes[1,1].set_title("Relative Error - Vel.")
# axes[1,1].grid(True)
# plt.tight_layout()
# plt.show()
# 
# fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
# # Plot 1: X vs Y
# axes[0].plot(sol_spice_syn[:, 0], sol_spice_syn[:, 1], label='Spice RPF')
# axes[0].scatter(sol_prop_syn[0, 0], sol_prop_syn[0, 1], color='black', s=50, label='Ep0')
# axes[0].set_xlabel("X")
# axes[0].set_ylabel("Y")
# axes[0].set_title("X-Y Projection")
# axes[0].legend()
# axes[0].grid(True)
# # Plot 2: X vs Z
# axes[1].plot(sol_spice_syn[:, 0], sol_spice_syn[:, 2], label='Spice RPF')
# axes[1].scatter(sol_prop_syn[0, 0], sol_prop_syn[0, 2], color='black', s=50, label='Ep0')
# axes[1].set_xlabel("X")
# axes[1].set_ylabel("Z")
# axes[1].set_title("X-Z Projection")
# axes[1].legend()
# axes[1].grid(True)
# # Plot 3: Y vs Z
# axes[2].plot(sol_spice_syn[:, 1], sol_spice_syn[:, 2], label='Spice RPF')
# axes[2].scatter(sol_prop_syn[0, 1], sol_prop_syn[0, 2], color='black', s=50, label='Ep0')
# axes[2].set_xlabel("Y")
# axes[2].set_ylabel("Z")
# axes[2].set_title("Y-Z Projection")
# axes[2].legend()
# axes[2].grid(True)
# plt.tight_layout()
# plt.show()
# # 


# fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
# # Plot 1: X vs Y
# axes[0].plot(sol_prop_syn[:, 0], sol_prop_syn[:, 1], label='Prop. RPF')
# axes[0].plot(sol_spice_syn[:, 0], sol_spice_syn[:, 1], label='Spice RPF')
# axes[0].scatter(sol_prop_syn[0, 0], sol_prop_syn[0, 1], color='black', s=50, label='Ep0')
# axes[0].set_xlabel("X")
# axes[0].set_ylabel("Y")
# axes[0].set_title("X-Y Projection")
# axes[0].legend()
# axes[0].grid(True)
# # Plot 2: X vs Z
# axes[1].plot(sol_prop_syn[:, 0], sol_prop_syn[:, 2], label='Prop. RPF')
# axes[1].plot(sol_spice_syn[:, 0], sol_spice_syn[:, 2], label='Spice RPF')
# axes[1].scatter(sol_prop_syn[0, 0], sol_prop_syn[0, 2], color='black', s=50, label='Ep0')
# axes[1].set_xlabel("X")
# axes[1].set_ylabel("Z")
# axes[1].set_title("X-Z Projection")
# axes[1].legend()
# axes[1].grid(True)
# # Plot 3: Y vs Z
# axes[2].plot(sol_prop_syn[:, 1], sol_prop_syn[:, 2], label='Prop. RPF')
# axes[2].plot(sol_spice_syn[:, 1], sol_spice_syn[:, 2], label='Spice RPF')
# axes[2].scatter(sol_prop_syn[0, 1], sol_prop_syn[0, 2], color='black', s=50, label='Ep0')
# axes[2].set_xlabel("Y")
# axes[2].set_ylabel("Z")
# axes[2].set_title("Y-Z Projection")
# axes[2].legend()
# axes[2].grid(True)
# plt.tight_layout()
# plt.show()
# 
# 
# fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
# # Plot 1: X vs Y
# axes[0].plot(sol_prop_in[:, 0], sol_prop_in[:, 1], label='Prop. Inert.')
# axes[0].plot(sol_spice_in[:, 0], sol_spice_in[:, 1], label='Spice Inert.')
# axes[0].scatter(sol_prop_in[0, 0], sol_prop_in[0, 1], color='black', s=50, label='Ep0')
# axes[0].set_xlabel("X")
# axes[0].set_ylabel("Y")
# axes[0].set_title("X-Y Projection")
# axes[0].legend()
# axes[0].grid(True)
# # Plot 2: X vs Z
# axes[1].plot(sol_prop_in[:, 0], sol_prop_in[:, 2], label='Prop. Inert.')
# axes[1].plot(sol_spice_in[:, 0], sol_spice_in[:, 2], label='Spice Inert.')
# axes[1].scatter(sol_prop_in[0, 0], sol_prop_in[0, 2], color='black', s=50, label='Ep0')
# axes[1].set_xlabel("X")
# axes[1].set_ylabel("Z")
# axes[1].set_title("X-Z Projection")
# axes[1].legend()
# axes[1].grid(True)
# # Plot 3: Y vs Z
# axes[2].plot(sol_prop_in[:, 1], sol_prop_in[:, 2], label='Prop. Inert.')
# axes[2].plot(sol_spice_in[:, 1], sol_spice_in[:, 2], label='Spice Inert.')
# axes[2].scatter(sol_prop_in[0, 1], sol_prop_in[0, 2], color='black', s=50, label='Ep0')
# axes[2].set_xlabel("Y")
# axes[2].set_ylabel("Z")
# axes[2].set_title("Y-Z Projection")
# axes[2].legend()
# axes[2].grid(True)
# plt.tight_layout()
# plt.show()

# fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5)) 
# # Plot 1: X vs Y
# axes2[0].plot(sol_prop_syn[:, 0], sol_prop_syn[:, 1], label='Prop syn RPF')
# axes2[0].scatter(sol_prop_syn[0, 0], sol_prop_syn[0, 1], color='black', s=50, label='Ep0')
# axes2[0].set_xlabel("X")
# axes2[0].set_ylabel("Y")
# axes2[0].set_title("X-Y Projection")
# axes2[0].legend()
# axes2[0].grid(True)
# # Plot 2: X vs Z
# axes2[1].plot(sol_prop_syn[:, 0], sol_prop_syn[:, 2], label='Prop syn RPF')
# axes2[1].scatter(sol_prop_syn[0, 0], sol_prop_syn[0, 2], color='black', s=50, label='Ep0')
# axes2[1].set_xlabel("X")
# axes2[1].set_ylabel("Z")
# axes2[1].set_title("X-Z Projection")
# axes2[1].legend()
# axes2[1].grid(True)
# # Plot 3: Y vs Z
# axes2[2].plot(sol_prop_syn[:, 1], sol_prop_syn[:, 2], label='Prop syn RPF')
# axes2[2].scatter(sol_prop_syn[0, 1], sol_prop_syn[0, 2], color='black', s=50, label='Ep0')
# axes2[2].set_xlabel("Y")
# axes2[2].set_ylabel("Z")
# axes2[2].set_title("Y-Z Projection")
# axes2[2].legend()
# axes2[2].grid(True)
# plt.tight_layout()
# plt.show()


"""
    check jacobians
"""

t = 1.2345
x = np.array([0.7655, -0.7094, 0.1190, 1.7513, 5.5472, -0.8143])

def wrapped_nbp_eph_dynamics(state):
    return rnp_rpf.dynamics(t, state, 0, 0, rnbprpf_dict_v2)

jac_nbp_eph = rnp_rpf.jacobian_x_standalone(t, x, 0, 0, rnbprpf_dict_v2)
# jac_fd_nbp_eph = approx_fprime(x, wrapped_nbp_eph_dynamics, 1.49e-08)
jac_fd_nbp_eph = approx_fprime(x, wrapped_nbp_eph_dynamics)
diff_jac_nbp_eph = jac_nbp_eph - jac_fd_nbp_eph
# print("jac: ", jac_nbp_eph)
# print("jac_fd_nbp_eph: ", jac_fd_nbp_eph)
print("diff_jac_nbp_eph: ", diff_jac_nbp_eph)
print(np.max(np.abs(diff_jac_nbp_eph)))
