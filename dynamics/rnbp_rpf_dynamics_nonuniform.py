import numpy as np
import spiceypy as spice
from numba import jit, prange
from scipy.integrate import odeint
from scipy.integrate import solve_ivp



def get_body_distance(t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag):
    pos_p_s, _ = spice.spkezr(str(naif_id_s), t, ref_frame, str(abcorr_flag), str(naif_id_p))
    
    if len(np.atleast_1d(t)) > 1:
        l = np.linalg.norm(pos_p_s, axis=1) # Distance between primary and secondary for each time in t
    else:
        l = np.linalg.norm(pos_p_s)  # scalar distance between primary and secondary
        
    return l

    
def dt_dtau(tau, t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag):
    l = get_body_distance(t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
    
    dtau_dt = np.sqrt((mu_p + mu_s) / (l**3))
    
    return 1 / dtau_dt




def compute_time(tau_vec, t0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = 1e-12, ode_atol = 1e-12, abcorr_flag = 'None'):
    t0_total = t0 + et0
    
#     sol = odeint(dt_dtau, t0_total, tau_vec, args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag, ), tfirst=True, rtol=ode_rtol, atol=ode_atol)
#     sol_y = sol[:,0]
#     sol_t = tau_vec
    
    if len(np.atleast_1d(tau_vec)) > 2:
        sol = solve_ivp(dt_dtau, [tau_vec[0], tau_vec[-1]], [t0_total], t_eval = tau_vec, args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag, ), method="DOP853", rtol=ode_rtol, atol=ode_atol)
    else:
        sol = solve_ivp(dt_dtau, [tau_vec[0], tau_vec[-1]], [t0_total], args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag, ), method="DOP853", rtol=ode_rtol, atol=ode_atol)
        
    sol_y = sol.y[0]
    sol_t = sol.t
    
    return sol_t, sol_y - et0



def compute_time_distance(tau_vec, t0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = 1e-12, ode_atol = 1e-12, abcorr_flag = 'None'):

    sol_t, sol_y = compute_time(tau_vec, t0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)

    l_vec = get_body_distance(sol_y + et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
    
    return sol_t, sol_y, l_vec
    


def dtau_dt(t, tau, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag):
    l = get_body_distance(t, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
    
    dtau_dt_ = np.sqrt((mu_p + mu_s) / (l**3))
    
    return dtau_dt_


def compute_tau(t_vec, tau0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = 1e-12, ode_atol = 1e-12, abcorr_flag = 'None'):
    t_vec = np.atleast_1d(t_vec)
    
    tvec_total = t_vec + et0
    t0 = tvec_total[0]
    tf = tvec_total[-1]
    
#     tau_vec = odeint(dtau_dt, tau0, tvec_total, args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag, ), tfirst=True, rtol=ode_rtol, atol=ode_atol)[:,0]
    
    if len(np.atleast_1d(t_vec)) > 2:
        sol = solve_ivp(dtau_dt, [tvec_total[0], tvec_total[-1]], [tau0], t_eval = tvec_total, args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag, ), method="DOP853", rtol=ode_rtol, atol=ode_atol)
    else:
        sol = solve_ivp(dtau_dt, [tvec_total[0], tvec_total[-1]], [tau0], args=(mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag, ), method="DOP853", rtol=ode_rtol, atol=ode_atol)
    
    sol_y = sol.y[0]
    sol_t = sol.t
    
    return sol_t - et0, sol_y



def compute_tau_distance(t_vec, tau0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = 1e-12, ode_atol = 1e-12, abcorr_flag = 'None'):
    
    sol_t, sol_y = compute_tau(t_vec, tau0, et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, ode_rtol = ode_rtol, ode_atol = ode_atol, abcorr_flag = abcorr_flag)
    l_vec = get_body_distance(t_vec + et0, mu_p, mu_s, naif_id_p, naif_id_s, ref_frame, abcorr_flag)
    
    return sol_t, sol_y, l_vec


@jit(nopython=True, cache=True, nogil=True)
def compute_tau_prime(distance, mu_p, mu_s):
    return np.sqrt((mu_p + mu_s) / (distance**3))


@jit(nopython=True, cache=True, nogil=True)
def compute_tau_double_prime(distance, distance_prime, mu_p, mu_s):
    t_prime = compute_tau_prime(distance, mu_p, mu_s)
    return - 3*t_prime*distance_prime / (2*distance)




""" Computes the gradient of the potential energy in the synodic n-body RPF

            Parameters
            ----------
            pos_bodies : nx3 array
                Position vectors of the n bodies [km]
            b : 1d array
                Barycenter of primaries [km]
            C: 3x3 array
                Rotation matrix [-]
            k: scalars
                Scaling factor (instantaneous distance between primary and secondary) [km]
            rho : 1d array
                Position in RPF [adim]
            mu_bodies: 1d array
                Mu values for bodies [km^3/s^2]

            Return
            ------
            grad: 1d array
                Gradient of potential energy in n-body RPF [km^2/s^2]

"""
@jit(nopython=True, cache=True, nogil=True)
def grad_rnbprpf(pos_bodies, b, C, k, rho, mu_bodies):
    # Synodic position of celestial bodies (eq. 3.20) 
    rhop = (pos_bodies - b) @ C / k # [adim]

    # Synodic relative position of spacecraft wrt celestial bodies (last term of eq. 3.19)
    diff_rho = rho - rhop # [adim]
    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) # [adim]

    # Gradient of potential energy (last term of eq. 3.19)
    factor = (mu_bodies / norm_rho**3)[:, np.newaxis] # [km^3/s^2]
    grad_om = - factor * diff_rho # [km^3/s^2]

    # Sum along the rows to get the final gradient
    grad = grad_om.sum(axis=0) # [km^2/s^2]

    return grad 


"""
    used for testing only
"""
def r_ddot(pos_bodies, b, C, k, rho, mu_bodies):
    r = b + k * C @ rho # [km]
    
    diff_r = r - pos_bodies # [km]
    norm_diff_r = np.sqrt(np.sum(diff_r ** 2, axis=1)) # [km]
    
    factor = (mu_bodies / norm_diff_r**3)[:, np.newaxis] # [km^3/s^2]
    grad_om = - factor * diff_r # [km^2/s^2]

    rddot = grad_om.sum(axis=0) # [km^2/s^2]

    return rddot




""" Definition of the synodic n-body RPF (for the dynamics)

            Parameters
            ----------
            id_primary : scalar
                ID of the primary in pos_bodies and mu_bodies
            id_secondary : scalar
                ID of the secondary in pos_bodies and mu_bodies
            pos_bodies : nx3 array
                Position vectors of the n bodies [adim]
            vel_bodies : nx3 array
                Velocity vectors of the n bodies [adim]
            num_bodies: scalar
                number of bodies
            mu_bodies: 1d array
                Mu values for bodies [km^3/s^2]

            Return
            ------
            b, b2d: 1d arrays
                Barycenter vector and second derivative. Units: b [km], b2d [km/s^2]
            k, k1d, k2d: scalars
                Scaling factor and derivatives (instantaneous distance between primary and secondary). Units: k [km], k1d [km/s], k2d [km/s^2]
            C, C1d, C2d: 3x3 arrays
                Rotation matrix and derivatives. Units: C [-], C1d [1/s], C2d [1/s^2]

"""
@jit(nopython=True, cache=True, nogil=True)
def frame_rnbprpf(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies):
# def frame_rnbprpf(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies, LU, VU):

    # Initialize acceleration and jerk arrays for only the primary and secondary bodies
    A = np.zeros((2, 3))
    J = np.zeros((2, 3))

    # Calculate accelerations and jerks for primary and secondary bodies
    for idx, i in enumerate((id_primary, id_secondary)):
        for j in prange(num_bodies):
            if j == i:
                continue  # Skip self-interaction
            
            R_vec = pos_bodies[j] - pos_bodies[i]
            V_vec = vel_bodies[j] - vel_bodies[i]
            norm_R = np.sqrt(np.dot(R_vec, R_vec))
            
            # Calculate acceleration and jerk (eqs. 3.17 and 3.18)
            acc = (mu_bodies[j] / norm_R**3) * R_vec
            jerk = (mu_bodies[j] / norm_R**5) * (norm_R**2 * V_vec - 3 * np.dot(R_vec, V_vec) * R_vec)

            # Sum up acceleration and jerk for the primary and secondary bodies only
            A[idx] += acc # [km^3/s^2]
            J[idx] += jerk # [km^3/s^2]

    # Extract primary and secondary positions, velocities, accelerations, and jerks
    Rp = pos_bodies[id_primary] # position of primary [km]
    Rs = pos_bodies[id_secondary] # position of secondary [km]
    Vp = vel_bodies[id_primary] # velocity of primary [km/s]
    Vs = vel_bodies[id_secondary] # velocity of secondary [km/s]
    Ap = A[0] # acceleration of primary [km/s^2]
    As = A[1] # acceleration of secondary [km/s^2]
    Jp = J[0] # jerk of primary [km/s^3]
    Js = J[1] # jerk of secondary [km/s^3]
    mu_p = mu_bodies[id_primary] # gravitational parameter of primary [km^3/s^2]
    mu_s = mu_bodies[id_secondary] # gravitational parameter of secondary [km^3/s^2]

    # Calculate relative vectors and barycenter (eq. 3.16)
    r = Rs - Rp # [km] 
    v = Vs - Vp # [km/s]
    a = As - Ap # [km/s^2]
    j = Js - Jp # [km/s^3]

    # Primaries barycenter (eq. 3.2)
    b = (mu_s * Rs + mu_p * Rp) / (mu_s + mu_p) # [km]
    b2d = (mu_s * As + mu_p * Ap) / (mu_s + mu_p) # [km/s^2]

    # Scaling factor (eqs. 3.3 and 3.12)
    k = np.sqrt(np.dot(r, r)) # [km]
    k1d = np.dot(r, v) / k # [km/s]
    k2d = (k * (np.dot(v, v) + np.dot(r, a)) - k1d * np.dot(r, v)) / k**2 # [km/s^2]

    # Direction cosines and rotation matrix
    crv = np.cross(r, v)
    cra = np.cross(r, a)
    cva = np.cross(v, a)
    crj = np.cross(r, j)
    
    # angular momentum and derivatives (eq. 3.15)
    h = np.linalg.norm(crv) # [km^2/s]
    h1d = np.dot(crv, cra) / h # [km^2/s^2]
    h2d = (h * (np.dot(cra, cra) + np.dot(crv, (cva + crj))) - h1d * np.dot(crv, cra)) / h**2 # [km^2/s^3]

    # Rotation matrix and derivatives (eq. 3.4)
    e1 = r / k # [-]
    e3 = crv / h # [-]
    e2 = np.cross(e3, e1) # [-]
    C = np.column_stack((e1, e2, e3)) # [-]

    # Rotation matrix 1st derivatives (eq. 3.13)
    e1d = (k * v - k1d * r) / k**2 # [1/s]
    e3d = (h * cra - h1d * crv) / h**2 # [1/s]
    e2d = np.cross(e3d, e1) + np.cross(e3, e1d) # [1/s]
    C1d = np.column_stack((e1d, e2d, e3d)) # [1/s]

    # Rotation matrix 2nd derivatives (eq. 3.14)
    e1d2 = ((2 * k1d**2 - k * k2d) * r - 2 * k * k1d * v + k**2 * a) / k**3 # [1/s^2]
    e3d2 = (h**2 * (cva + crj) - 2 * h * h1d * cra + (2 * h1d**2 - h * h2d) * crv) / h**3 # [1/s^2]
    e2d2 = np.cross(e3d2, e1) + 2 * np.cross(e3d, e1d) + np.cross(e3, e1d2) # [1/s^2]
    C2d = np.column_stack((e1d2, e2d2, e3d2)) # [1/s^2]

    return b, b2d, k, k1d, k2d, C, C1d, C2d




@jit(nopython=True, cache=True, nogil=True)
def frame_rnbprpf_coeff(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies):
# def frame_rnbprpf_coeff(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies, LU, VU):

    # Initialize acceleration and jerk arrays for only the primary and secondary bodies
    A = np.zeros((2, 3))
    J = np.zeros((2, 3))

    # Calculate accelerations and jerks for primary and secondary bodies
    for idx, i in enumerate((id_primary, id_secondary)):
        for j in prange(num_bodies):
            if j == i:
                continue  # Skip self-interaction
            
            R_vec = pos_bodies[j] - pos_bodies[i]
            V_vec = vel_bodies[j] - vel_bodies[i]
            norm_R = np.sqrt(np.dot(R_vec, R_vec))
            
            # Calculate acceleration and jerk (eqs. 3.17 and 3.18)
            acc = (mu_bodies[j] / norm_R**3) * R_vec
            jerk = (mu_bodies[j] / norm_R**5) * (norm_R**2 * V_vec - 3 * np.dot(R_vec, V_vec) * R_vec)

            # Sum up acceleration and jerk for the primary and secondary bodies only
            A[idx] += acc # [km^3/s^2]
            J[idx] += jerk # [km^3/s^2]

    # Extract primary and secondary positions, velocities, accelerations, and jerks
    Rp = pos_bodies[id_primary] # position of primary [km]
    Rs = pos_bodies[id_secondary] # position of secondary [km]
    Vp = vel_bodies[id_primary] # velocity of primary [km/s]
    Vs = vel_bodies[id_secondary] # velocity of secondary [km/s]
    Ap = A[0] # acceleration of primary [km/s^2]
    As = A[1] # acceleration of secondary [km/s^2]
    Jp = J[0] # jerk of primary [km/s^3]
    Js = J[1] # jerk of secondary [km/s^3]
    mu_p = mu_bodies[id_primary] # gravitational parameter of primary [km^3/s^2]
    mu_s = mu_bodies[id_secondary] # gravitational parameter of secondary [km^3/s^2]

    # Calculate relative vectors and barycenter (eq. 3.16)
    r = Rs - Rp # [km] 
    v = Vs - Vp # [km/s]
    a = As - Ap # [km/s^2]
    j = Js - Jp # [km/s^3]

    # Primaries barycenter (eq. 3.2)
    b = (mu_s * Rs + mu_p * Rp) / (mu_s + mu_p) # [km]
    b2d = (mu_s * As + mu_p * Ap) / (mu_s + mu_p) # [km/s^2]

    # Scaling factor (eqs. 3.3 and 3.12)
    k = np.sqrt(np.dot(r, r)) # [km]
    k1d = np.dot(r, v) / k # [km/s]
    k2d = (k * (np.dot(v, v) + np.dot(r, a)) - k1d * np.dot(r, v)) / k**2 # [km/s^2]

    # Direction cosines and rotation matrix
    crv = np.cross(r, v)
    cra = np.cross(r, a)
    
    # angular momentum and derivatives (eq. 3.15)
    h = np.linalg.norm(crv) # [km^2/s]
    h1d = np.dot(crv, cra) / h # [km^2/s^2]

    # Rotation matrix and derivatives (eq. 3.4)
    e1 = r / k # [-]
    e3 = crv / h # [-]
    e2 = np.cross(e3, e1) # [-]
    C = np.column_stack((e1, e2, e3)) # [-]

    return a, j, h, h1d, b, b2d, k, k1d, k2d, C




    
    

""" Dynamics for synodic n-body RPF

            Parameters
            ----------
            tau : scalar
                Time [adim]
            state : 1d array
                State in synodic n-body RPF [adim]
            rnbprpf_dict : dictionary
                Contains parameters for n-body RPF

            Return
            ------
            1d array
                Derivative of states in synodic n-body RPF [adim]

"""
def dynamics(tau, state, control, p, input_dict):
    return dynamics_coeff_table(tau, state, control, p, input_dict)



def dynamics_coeff_eom(tau, state, control, p, input_dict):

    et0 = input_dict["param"]["et0"]
    naif_id_bodies = input_dict["param"]["naif_id_bodies"]
    mu_bodies = input_dict["param"]["mu_bodies"]
    id_primary = input_dict["param"]["id_primary"]
    id_secondary = input_dict["param"]["id_secondary"]
    inertial_frame = input_dict["param"]["inertial_frame"]
    abcorr_flag = input_dict["param"]["abcorr_flag"]
    origin_frame = input_dict["param"]["origin_frame"]
    tau_vec = input_dict["param"]["tau_vec"]
    t_vec = input_dict["param"]["t_vec"]
#     distance_vec = input_dict["param"]["distance_vec"]

    num_bodies = len(mu_bodies)
    identity3 = np.identity(3)
    
    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    mu_p = mu_bodies[id_primary]
    mu_s = mu_bodies[id_secondary]
    mu_p_adim = mu_p / (mu_p + mu_s)
    mu_s_adim = mu_s / (mu_p + mu_s)
    t = np.interp(tau, tau_vec, t_vec) # [s]
#     distance = np.interp(tau, tau_vec, distance_vec) # [km]

    # eq. 3.1b
    et = et0 + t
               
    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_bodies[i,:] = state_body # [km, km/s]

    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]
    
    b, b2d, k, k1d, k2d, C, C1d, C2d = \
        frame_rnbprpf(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies)
 
    distance = k
    
    tau_prime = compute_tau_prime(distance, mu_p, mu_s)
    tau_prime_squared = tau_prime**2
 
    distance_prime = k1d
    distance_double_prime = k2d
    tau_double_prime = compute_tau_double_prime(distance, distance_prime, mu_p, mu_s)
    
    grad_om_adim = grad_rnbprpf(pos_bodies, b, C, k, rho, mu_bodies / (mu_p + mu_s))
    
    b13 = (mu_p + mu_s) / (distance**3 * tau_prime_squared)

    b_123 = - C.T @ b2d / (tau_prime_squared * distance)
    
    rho_dot_matrix = -1/tau_prime * ( (2*distance_prime/distance + tau_double_prime/tau_prime) * identity3 + 2 * C.T @ C1d)
    rho_matrix = -1/tau_prime_squared * ( distance_double_prime/distance * identity3 + 2*distance_prime/distance * C.T @ C1d + C.T @ C2d)

    drho = eta
    deta = b_123 + rho_dot_matrix @ eta + rho_matrix @ rho + b13 * grad_om_adim
    
    return np.concatenate((drho, deta)) # [adim, adim]



def dynamics_coeff_table(tau, state, control, p, input_dict):

    et0 = input_dict["param"]["et0"]
    naif_id_bodies = input_dict["param"]["naif_id_bodies"]
    mu_bodies = input_dict["param"]["mu_bodies"]
    id_primary = input_dict["param"]["id_primary"]
    id_secondary = input_dict["param"]["id_secondary"]
    inertial_frame = input_dict["param"]["inertial_frame"]
    abcorr_flag = input_dict["param"]["abcorr_flag"]
    origin_frame = input_dict["param"]["origin_frame"]
    tau_vec = input_dict["param"]["tau_vec"]
    t_vec = input_dict["param"]["t_vec"]
#     distance_vec = input_dict["param"]["distance_vec"]

    num_bodies = len(mu_bodies)
    identity3 = np.identity(3)
    
    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    mu_p = mu_bodies[id_primary]
    mu_s = mu_bodies[id_secondary]
    t = np.interp(tau, tau_vec, t_vec) # [s]
#     distance = np.interp(tau, tau_vec, distance_vec) # [km]
    
    # eq. 3.1b
    et = et0 + t
               
    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_bodies[i,:] = state_body # [km, km/s]

    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]

    acc_p_s, jerk_p_s, h, h1d, b, b2d, k, k1d, k2d, C = \
        frame_rnbprpf_coeff(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies)
 
    distance = k
    
    tau_prime = compute_tau_prime(distance, mu_p, mu_s)
    tau_prime_squared = tau_prime**2
 
    distance_prime = k1d
    distance_double_prime = k2d
    tau_double_prime = compute_tau_double_prime(distance, distance_prime, mu_p, mu_s)
    
    grad_om_adim = grad_rnbprpf(pos_bodies, b, C, k, rho, mu_bodies / (mu_p + mu_s))
     
    b_123 = - C.T @ b2d / (tau_prime_squared * distance)

    b4 = - distance_prime / (2*tau_prime*distance)
    b5 = 2*h / (tau_prime*distance**2)
    b6 = 2*distance / (tau_prime*h) * np.dot(acc_p_s, C[:,2])
    b7 = -distance_double_prime / (tau_prime_squared*distance) + h**2 / (tau_prime_squared*distance**4)
    b8 = -1 / (tau_prime_squared*distance) * np.dot(acc_p_s, C[:,2])
    b9 = h1d / (tau_prime_squared*distance**2)
    b10 = -distance_double_prime / (tau_prime_squared*distance) + h**2 / (tau_prime_squared*distance**4) + distance**2 / (tau_prime_squared * h**2) * (np.dot(acc_p_s, C[:,2]))**2
    b11 = (3*h*distance_prime - 2*distance*h1d ) * np.dot(acc_p_s, C[:,2]) / (tau_prime_squared * h**2) + distance * np.dot(jerk_p_s, C[:,2]) / (tau_prime_squared * h)
    b12 = -distance_double_prime / (tau_prime_squared*distance) +  distance**2 / (tau_prime_squared * h**2) * (np.dot(acc_p_s, C[:,2]))**2
    b13 = (mu_p + mu_s) / (distance**3 * tau_prime_squared)
    
    rho_dot_matrix = np.array([
        [b4, b5, 0.0],
        [-b5, b4, b6],
        [0.0, -b6, b4]
        ])
    rho_matrix = np.array([
        [b7, b9, b8],
        [-b9, b10, b11],
        [b8, -b11, b12]
        ])

    drho = eta
    deta = b_123 + rho_dot_matrix @ eta + rho_matrix @ rho + b13 * grad_om_adim

    return np.concatenate((drho, deta)) # [adim, adim]




""" Computes the Jacobian matrix in the synodic n-body RPF

            Parameters
            ----------
            pos_bodies: nx3 array
                Position vectors of the n bodies [km]
            b: 1d array
                Barycenter of primaries [km]
            rho: 1d array
                Position in RPF [adim]
            num_bodies: scalar
                number of bodies
            C, C1d, C2d: 3x3 arrays
                Rotation matrix and derivatives. Units: C [-], C1d [1/s], C2d [1/s^2]
            k, k1d, k2d: scalars
                Scaling factor and derivatives (instantaneous distance between primary and secondary). Units: k [km], k1d [km/s], k2d [km/s^2]
            tau_prime: scalar
                Scaling factor for time [1/s]
            mu_bodies: 1d array
                Mu values for bodies [km^3/s^2]

            Return
            ------
            jac: 6x6 array
                Jacobian in n-body RPF [adim]

"""
def jacobian_x(pos_bodies, b, rho, num_bodies, C, C1d, C2d, k, k1d, k2d, tau_prime, mu_bodies, id_primary, id_secondary):

    mu_p = mu_bodies[id_primary]
    mu_s = mu_bodies[id_secondary]
    
    mu_bodies_adim = mu_bodies / (mu_p + mu_s)

    I3 = np.identity(3)
    Z3 = np.zeros((3, 3))
    
    # Synodic position of celestial bodies (eq. 3.20)
    rhop = (pos_bodies - b) @ C / k 

    # Synodic relative position of spacecraft wrt celestial bodies (last term of eq. 3.19)
    diff_rho = rho - rhop 
    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) 

    # Gradient of potential energy (eq. 5.19)
    gSTM_om1a = np.sum(mu_bodies_adim / norm_rho**3) * I3

     # Compute om_mat for gradient calculation (eq. 5.19)
    om_mat = (mu_bodies_adim / norm_rho**5)[:, np.newaxis] * diff_rho
    gSTM_om1b = np.zeros((3, 3))
    for jj in range(num_bodies):
        gSTM_om1b += 3 * np.outer(om_mat[jj, :], diff_rho[jj, :])

    distance = k
    
    tau_prime_squared = tau_prime**2
 
    distance_prime = k1d
    distance_double_prime = k2d
    tau_double_prime = compute_tau_double_prime(distance, distance_prime, mu_p, mu_s)
    
    rho_dot_matrix = -1/tau_prime * ( (2*distance_prime/distance + tau_double_prime/tau_prime) * I3 + 2 * C.T @ C1d)
    rho_matrix = -1/tau_prime_squared * ( distance_double_prime/distance * I3 + 2*distance_prime/distance * C.T @ C1d + C.T @ C2d)
    
    b13 = (mu_p + mu_s) / (distance**3 * tau_prime_squared)
    
    # sub-matrices for jacobian (eqs. 5.18 and 5.19)
    A11 = Z3  # drhodot / drho
    A12 = I3  # drhodot / deta
    A21 = rho_matrix + b13 * (- gSTM_om1a + gSTM_om1b) # detadot / drho
    A22 = rho_dot_matrix # detadot / deta

    # Assemble Jacobian
    jac = np.block([[A11, A12], [A21, A22]])

    return jac



""" Computes the Jacobian matrix in the synodic n-body RPF (standalone version)

            Parameters
            ----------
            tau : scalar
                Time [adim]
            state : 1d array
                State in synodic n-body RPF [adim]
            input_dict : dictionary
                Contains parameters for n-body RPF

            Return
            ------
            jac: 6x6 array
                Jacobian in n-body RPF [adim]

"""
def jacobian_x_standalone(tau, state, control, p, input_dict):

    et0 = input_dict["param"]["et0"]
    naif_id_bodies = input_dict["param"]["naif_id_bodies"]
    mu_bodies = input_dict["param"]["mu_bodies"]
    id_primary = input_dict["param"]["id_primary"]
    id_secondary = input_dict["param"]["id_secondary"]
    inertial_frame = input_dict["param"]["inertial_frame"]
    abcorr_flag = input_dict["param"]["abcorr_flag"]
    origin_frame = input_dict["param"]["origin_frame"]
    tau_vec = input_dict["param"]["tau_vec"]
    t_vec = input_dict["param"]["t_vec"]
#     distance_vec = input_dict["param"]["distance_vec"]
    
    num_bodies = len(mu_bodies)
    identity3 = np.identity(3)
    
    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    mu_p = mu_bodies[id_primary]
    mu_s = mu_bodies[id_secondary]
    t = np.interp(tau, tau_vec, t_vec) # [s]
#     distance = np.interp(tau, tau_vec, distance_vec) # [km]
    
    # eq. 3.1b
    et = et0 + t
               
    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_bodies[i,:] = state_body # [km, km/s]

    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]
    
    b, _, k, k1d, k2d, C, C1d, C2d = \
        frame_rnbprpf(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies)

    tau_prime = compute_tau_prime(k, mu_p, mu_s)
    
    jac = jacobian_x(pos_bodies, b, rho, num_bodies, C, C1d, C2d, k, k1d, k2d, tau_prime, mu_bodies, id_primary, id_secondary)

    return jac



""" Computes the dynamics and STM in the synodic n-body RPF

            Parameters
            ----------
            tau : scalar
                Time [adim]
            state_stm : 1d array
                Flattened state and STM matrix in synodic n-body RPF [adim]
            input_dict : dictionary
                Contains parameters for n-body RPF

            Return
            ------
            1d array
                Flattened derivative of state and STM matrix in n-body RPF [adim]

"""
def dynamics_stm(tau, state_stm, control, p, input_dict):

    et0 = input_dict["param"]["et0"]
    naif_id_bodies = input_dict["param"]["naif_id_bodies"]
    mu_bodies = input_dict["param"]["mu_bodies"]
    id_primary = input_dict["param"]["id_primary"]
    id_secondary = input_dict["param"]["id_secondary"]
    LU = input_dict["scaling"]["LU"]
    VU = input_dict["scaling"]["VU"]
    TU = input_dict["scaling"]["TU"]
    inertial_frame = input_dict["param"]["inertial_frame"]
    abcorr_flag = input_dict["param"]["abcorr_flag"]
    origin_frame = input_dict["param"]["origin_frame"]
    tau_vec = input_dict["param"]["tau_vec"]
    t_vec = input_dict["param"]["t_vec"]
    
    num_bodies = len(mu_bodies)
    identity3 = np.identity(3)
    
    rho = state_stm[0:3] # [adim]
    eta = state_stm[3:6] # [adim]
    stm_x = state_stm[6:42].reshape((6, 6)) # [adim]
    
    mu_p = mu_bodies[id_primary]
    mu_s = mu_bodies[id_secondary]
    
    t = np.interp(tau, tau_vec, t_vec) # [s]
    
    # eq. 3.1b
    et = et0 + t
               
    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_bodies[i,:] = state_body # [km, km/s]

    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]
    
    b, b2d, k, k1d, k2d, C, C1d, C2d = \
        frame_rnbprpf(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies)
     
    distance = k
    
    tau_prime = compute_tau_prime(distance, mu_p, mu_s)
    tau_prime_squared = tau_prime**2
 
    distance_prime = k1d
    distance_double_prime = k2d
    tau_double_prime = compute_tau_double_prime(distance, distance_prime, mu_p, mu_s)
    
    grad_om_adim = grad_rnbprpf(pos_bodies, b, C, k, rho, mu_bodies / (mu_p + mu_s))
    
    b13 = (mu_p + mu_s) / (distance**3 * tau_prime_squared)

    b_123 = - C.T @ b2d / (tau_prime_squared * distance)
    
    rho_dot_matrix = -1/tau_prime * ( (2*distance_prime/distance + tau_double_prime/tau_prime) * identity3 + 2 * C.T @ C1d)
    rho_matrix = -1/tau_prime_squared * ( distance_double_prime/distance * identity3 + 2*distance_prime/distance * C.T @ C1d + C.T @ C2d)

    drho = eta
    deta = b_123 + rho_dot_matrix @ eta + rho_matrix @ rho + b13 * grad_om_adim
    
    jac_x = jacobian_x(pos_bodies, b, rho, num_bodies, C, C1d, C2d, k, k1d, k2d, tau_prime, mu_bodies, id_primary, id_secondary)
    dstm = jac_x.dot(stm_x) 
    
    return np.concatenate((drho, deta, dstm.flatten())) # [adim, adim, adim]

