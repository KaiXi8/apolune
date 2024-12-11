import numpy as np
import spiceypy as spice
from numba import jit, prange
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import dynamics_coeff.dynamics_utils as dyn_utils


def get_body_positions(t, naif_id_bodies, observer_id, reference_frame):
    num_bodies = len(naif_id_bodies)

    # retrieve states of celestial bodies from SPICE
    pos_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        pos_body, _ = spice.spkgps(naif_id_bodies[i], t, reference_frame, observer_id)
        pos_bodies[i,:] = pos_body # [km, km/s]

    return pos_bodies
    

def get_body_states(t, naif_id_bodies, observer_id, reference_frame):
    num_bodies = len(naif_id_bodies)

    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkgeo(naif_id_bodies[i], t, reference_frame, observer_id)
        state_bodies[i,:] = state_body # [km, km/s]

    return state_bodies



def compute_epoch_time(tau, tau_vec, t_vec, epoch_t0):
    t = np.interp(tau, tau_vec, t_vec) # [s]
    return t + epoch_t0
    
    

@jit(nopython=True, cache=True, nogil=True)
def frame_rnbprpf_coeff(id_primary, id_secondary, pos_bodies, vel_bodies, mu_bodies):

    # Initialize acceleration and jerk arrays for only the primary and secondary bodies
    A = np.zeros((2, 3))
    J = np.zeros((2, 3))
    
    num_bodies = len(mu_bodies)

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



@jit(nopython=True, cache=True, nogil=True)
def frame_rnbprpf(id_primary, id_secondary, pos_bodies, vel_bodies, mu_bodies):

    # Initialize acceleration and jerk arrays for only the primary and secondary bodies
    A = np.zeros((2, 3))
    J = np.zeros((2, 3))
    
    num_bodies = len(mu_bodies)

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


def inertialToSynodic_pos(pos_bodies, b, C, k):
    return (pos_bodies - b) @ C / k # [adim]



@jit(nopython=True, cache=True, nogil=True)
def compute_tau_prime(distance, mu_p, mu_s):
    return np.sqrt((mu_p + mu_s) / (distance**3))


@jit(nopython=True, cache=True, nogil=True)
def compute_tau_double_prime(distance, distance_prime, mu_p, mu_s):
    t_prime = compute_tau_prime(distance, mu_p, mu_s)
    return - 3*t_prime*distance_prime / (2*distance)
    


def grad_omega(rho, pos_bodies, b, C, k, mu_adim):
    rho_p = inertialToSynodic_pos(pos_bodies, b, C, k)
    return dyn_utils.compute_grad_omega(rho, rho_p, mu_adim)



def jac_grad_omega(rho, pos_bodies, b, C, k, mu_adim):
    rho_p = inertialToSynodic_pos(pos_bodies, b, C, k)
    return dyn_utils.compute_jac_grad_omega(rho, rho_p, mu_adim)
    



def compute_coeff_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec):

    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    mu_p = mu_bodies_dim[id_primary]
    mu_s = mu_bodies_dim[id_secondary]
    
    epoch_time = compute_epoch_time(tau, tau_vec, t_vec, epoch_t0)
    
    state_bodies = get_body_states(epoch_time, naif_id_bodies, observer_id, reference_frame)
    
    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]

    acc_p_s, jerk_p_s, h, h1d, b, b2d, k, k1d, k2d, C = \
        frame_rnbprpf_coeff(id_primary, id_secondary, pos_bodies, vel_bodies, mu_bodies_dim)
 
    distance = k
    
    tau_prime = compute_tau_prime(distance, mu_p, mu_s)
    tau_prime_squared = tau_prime**2
 
    distance_prime = k1d
    distance_double_prime = k2d
         
    b_123 = - C.T @ b2d / (tau_prime_squared * distance)
    
    b1, b2, b3 = b_123
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
        
    mu_adim = mu_bodies_dim / (mu_p + mu_s)
    grad_om_adim = grad_omega(rho, pos_bodies, b, C, k, mu_adim)
    
    return np.array([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13]), grad_om_adim



def compute_coeff_jac_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec):

    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    mu_p = mu_bodies_dim[id_primary]
    mu_s = mu_bodies_dim[id_secondary]
    
    epoch_time = compute_epoch_time(tau, tau_vec, t_vec, epoch_t0)
    
    state_bodies = get_body_states(epoch_time, naif_id_bodies, observer_id, reference_frame)
    
    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]

    acc_p_s, jerk_p_s, h, h1d, b, b2d, k, k1d, k2d, C = \
        frame_rnbprpf_coeff(id_primary, id_secondary, pos_bodies, vel_bodies, mu_bodies_dim)
 
    distance = k
    
    tau_prime = compute_tau_prime(distance, mu_p, mu_s)
    tau_prime_squared = tau_prime**2
 
    distance_prime = k1d
    distance_double_prime = k2d
         
    b_123 = - C.T @ b2d / (tau_prime_squared * distance)
    
    b1, b2, b3 = b_123
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
    
    mu_adim = mu_bodies_dim / (mu_p + mu_s)
    jac_grad_om_adim = jac_grad_omega(rho, pos_bodies, b, C, k, mu_adim)
    
    
    return np.array([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13]), jac_grad_om_adim