import numpy as np
import spiceypy as spice
from numba import jit, prange
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


def frame_encoder(frame):
    if isinstance(frame, str):  # single frame name to encode
        return np.array([ord(c) for c in frame + '\0'], dtype=np.int8)
    if isinstance(frame, (tuple, list)):  # multiple frame names to encode
        nb_chars = [len(f) for f in frame]  # number of characters for each frame name
        nb_cols = max(nb_chars) + 1  # number of columns in the output array
        padded_frame = [f + '\0' * (nb_cols - nb_chars[i]) for i, f in enumerate(frame)]
        encoded_frame = np.asarray([ord(c) for c in ''.join(padded_frame)], dtype=np.int8)
        return encoded_frame.reshape(len(frame), max(nb_chars) + 1)
    raise Exception('frame must be a string or a tuple of strings representing valid frame names')



def get_body_distance(t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame):
    pos_p_s, _ = spice.spkgps(naif_id_s, t, ref_frame, naif_id_p)
    
    if len(np.atleast_1d(t)) > 1:
        l = np.linalg.norm(pos_p_s, axis=1) # Distance between primary and secondary for each time in t
    else:
        l = np.linalg.norm(pos_p_s)  # scalar distance between primary and secondary
        
    return l

    
def dt_dtau(tau, t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame):
    l = get_body_distance(t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame)
    
    dtau_dt = np.sqrt((mu_p + mu_s) / (l**3))
    
    return 1 / dtau_dt




def compute_time(tau_vec, t0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = 1e-12, ode_atol = 1e-12):
    t0_total = t0 + et0
    
#     sol = odeint(dt_dtau, t0_total, tau_vec, args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ), tfirst=True, rtol=ode_rtol, atol=ode_atol)
#     sol_y = sol[:,0]
#     sol_t = tau_vec
    
    if len(np.atleast_1d(tau_vec)) > 2:
        sol = solve_ivp(dt_dtau, [tau_vec[0], tau_vec[-1]], [t0_total], t_eval = tau_vec, args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ), method="DOP853", rtol=ode_rtol, atol=ode_atol)
    else:
        sol = solve_ivp(dt_dtau, [tau_vec[0], tau_vec[-1]], [t0_total], args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ), method="DOP853", rtol=ode_rtol, atol=ode_atol)
        
    sol_y = sol.y[0]
    sol_t = sol.t
    
    return sol_t, sol_y - et0



def compute_time_distance(tau_vec, t0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = 1e-12, ode_atol = 1e-12):

    sol_t, sol_y = compute_time(tau_vec, t0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol)

    l_vec = get_body_distance(sol_y + et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame)
    
    return sol_t, sol_y, l_vec
    


def dtau_dt(t, tau, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame):
    l = get_body_distance(t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame)
    
    dtau_dt_ = np.sqrt((mu_p + mu_s) / (l**3))
    
    return dtau_dt_


def compute_tau(t_vec, tau0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = 1e-12, ode_atol = 1e-12):
    t_vec = np.atleast_1d(t_vec)
    
    tvec_total = t_vec + et0
    t0 = tvec_total[0]
    tf = tvec_total[-1]
    
#     tau_vec = odeint(dtau_dt, tau0, tvec_total, args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ), tfirst=True, rtol=ode_rtol, atol=ode_atol)[:,0]
    
    if len(np.atleast_1d(t_vec)) > 2:
        sol = solve_ivp(dtau_dt, [tvec_total[0], tvec_total[-1]], [tau0], t_eval = tvec_total, args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ), method="DOP853", rtol=ode_rtol, atol=ode_atol)
    else:
        sol = solve_ivp(dtau_dt, [tvec_total[0], tvec_total[-1]], [tau0], args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ), method="DOP853", rtol=ode_rtol, atol=ode_atol)
    
    sol_y = sol.y[0]
    sol_t = sol.t
    
    return sol_t - et0, sol_y



def compute_tau_distance(t_vec, tau0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = 1e-12, ode_atol = 1e-12):
    
    sol_t, sol_y = compute_tau(t_vec, tau0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol)
    l_vec = get_body_distance(t_vec + et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame)
    
    return sol_t, sol_y, l_vec