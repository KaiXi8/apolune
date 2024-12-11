import numpy as np
from numba import jit, prange
import spiceypy as spice


def get_mu_bodies_crtbp(mu):
    return np.array([1-mu, mu])
    

def get_mu_bodies_bcrfbp(mu, mu_4):
    return np.array([1-mu, mu, mu_4])



def get_body_positions_crtbp(mu):
    rho_p = np.array([-mu, 0.0, 0.0])
    rho_s = np.array([1-mu, 0.0, 0.0])

    return np.vstack((rho_p, rho_s))



def get_body_positions_bcrfbp(tau, mu, a_sun, om_sun, sun_angle_t0):
    rho_ps = get_body_positions_crtbp(mu)
    
    rho_sun = compute_sun_position_bcrfbp(tau, a_sun, om_sun, sun_angle_t0)

    return np.vstack((rho_ps, rho_sun))



def get_body_positions_rnbp(t, naif_id_bodies, observer_id, reference_frame):
    num_bodies = len(naif_id_bodies)

    # retrieve states of celestial bodies from SPICE
    pos_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        pos_body, _ = spice.spkgps(naif_id_bodies[i], t, reference_frame, observer_id)
        pos_bodies[i,:] = pos_body # [km, km/s]

    return pos_bodies
    

def get_body_states_rnbp(t, naif_id_bodies, observer_id, reference_frame):
    num_bodies = len(naif_id_bodies)

    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkgeo(naif_id_bodies[i], t, reference_frame, observer_id)
        state_bodies[i,:] = state_body # [km, km/s]

    return state_bodies




def omega(rho, rho_p, mu_bodies_adim):
    # Synodic relative position of spacecraft wrt celestial bodies
    diff_rho = rho - rho_p
    
    diff_rho = np.atleast_1d(diff_rho)
    mu_bodies_adim = np.atleast_1d(mu_bodies_adim)

    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) # [adim]
    
    # calculate omega
    om = (mu_bodies_adim / norm_rho)[:, np.newaxis] # [adim]

    return om.sum(axis=0)



def grad_omega(rho, rho_p, mu_bodies_adim):
    # Synodic relative position of spacecraft wrt celestial bodies
    diff_rho = rho - rho_p
    
    diff_rho = np.atleast_1d(diff_rho)
    mu_bodies_adim = np.atleast_1d(mu_bodies_adim)
    
    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) # [adim]

    # Gradient of potential energy 
    factor = (mu_bodies_adim / norm_rho**3)[:, np.newaxis] # [adim]
    grad_om = - factor * diff_rho # [adim]

    # Sum along the rows to get the final gradient
    grad = grad_om.sum(axis=0) # [adim]

    return grad 



def jac_grad_omega(rho, rho_p, mu_bodies_adim):
    # Synodic relative position of spacecraft wrt celestial bodies
    diff_rho = rho - rho_p 
    
    diff_rho = np.atleast_1d(diff_rho)
    mu_bodies_adim = np.atleast_1d(mu_bodies_adim)
    
    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) 

    I3 = np.identity(3)
    
    num_bodies = len(mu_bodies_adim)
    
    # Term 1 of gradient of potential energy
    grad_om_term1 = - np.sum(mu_bodies_adim / norm_rho**3) * I3

    # Term 2 of gradient of potential energy
    om_mat = (mu_bodies_adim / norm_rho**5)[:, np.newaxis] * diff_rho
    grad_om_term2 = np.zeros((3, 3))
    for jj in range(num_bodies):
        grad_om_term2 += 3 * np.outer(om_mat[jj, :], diff_rho[jj, :])

    return grad_om_term1 + grad_om_term2




def omega_crtbp(rho, mu):
    x, y = rho[:2]
    rho_p = get_body_positions_crtbp(mu)
    mu_adim = get_mu_bodies_crtbp(mu)
    
    om1 = omega(rho, rho_p, mu_adim)
    om2 = (x**2 + y**2) / 2
    
    return om1 + om2



def grad_omega_crtbp(rho, mu):
    x, y = rho[:2]
    rho_p = get_body_positions_crtbp(mu)
    mu_adim = get_mu_bodies_crtbp(mu)
    
    grad_om = grad_omega(rho, rho_p, mu_adim)
    
    return grad_om



def jac_grad_omega_crtbp(rho, mu):
    rho_p = get_body_positions_crtbp(mu)
    mu_adim = get_mu_bodies_crtbp(mu)

    return jac_grad_omega(rho, rho_p, mu_adim)

 

def compute_coeff_crtbp():
    b = np.zeros(13)
    b[4] = 2.0
    b[6] = 1.0
    b[9] = 1.0
    b[12] = 1.0
    
    return b
    



def compute_sun_angle_bcrfbp(tau, om_sun, sun_angle_t0):
    return om_sun * tau + sun_angle_t0



def compute_sun_position_bcrfbp(tau, a_sun, om_sun, sun_angle_t0):
    sun_angle = compute_sun_angle_bcrfbp(tau, om_sun, sun_angle_t0)
    rho_sun = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0])
    return rho_sun


def omega_bcrfbp(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0):

    rho_p = get_body_positions_bcrfbp(tau, mu_crtbp, a_sun, om_sun, sun_angle_t0)
    mu_adim = get_mu_bodies_bcrfbp(mu_crtbp, mu_sun)
    
    return omega(rho, rho_p, mu_adim)



def grad_omega_bcrfbp(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0):

    rho_p = get_body_positions_bcrfbp(tau, mu_crtbp, a_sun, om_sun, sun_angle_t0)
    mu_adim = get_mu_bodies_bcrfbp(mu_crtbp, mu_sun)
    
    return grad_omega(rho, rho_p, mu_adim)



def jac_grad_omega_bcrfbp(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0):

    rho_p = get_body_positions_bcrfbp(tau, mu_crtbp, a_sun, om_sun, sun_angle_t0)
    mu_adim = get_mu_bodies_bcrfbp(mu_crtbp, mu_sun)
    
    return jac_grad_omega(rho, rho_p, mu_adim)




def compute_coeff_bcrfbp(tau, mu_sun, a_sun, om_sun, sun_angle_t0):
    b = np.zeros(13)
    
    sun_angle = compute_sun_angle_bcrfbp(tau, om_sun, sun_angle_t0)
    tmp = - mu_sun / a_sun**2
    b[0] = tmp * np.cos(sun_angle)
    b[1] = tmp * np.sin(sun_angle)
    b[4] = 2.0
    b[6] = 1.0
    b[9] = 1.0
    b[12] = 1.0
    
    return b




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


def nbp_rpf_inertialToSynodic_pos(pos_bodies, b, C, k):
    return (pos_bodies - b) @ C / k # [adim]



@jit(nopython=True, cache=True, nogil=True)
def compute_tau_prime(distance, mu_p, mu_s):
    return np.sqrt((mu_p + mu_s) / (distance**3))


@jit(nopython=True, cache=True, nogil=True)
def compute_tau_double_prime(distance, distance_prime, mu_p, mu_s):
    t_prime = compute_tau_prime(distance, mu_p, mu_s)
    return - 3*t_prime*distance_prime / (2*distance)
    


def grad_omega_rnbp_rpf(rho, pos_bodies, b, C, k, mu_adim):
    rho_p = nbp_rpf_inertialToSynodic_pos(pos_bodies, b, C, k)
    return grad_omega(rho, rho_p, mu_adim)



def jac_grad_omega_rnbp_rpf(rho, pos_bodies, b, C, k, mu_adim):
    rho_p = nbp_rpf_inertialToSynodic_pos(pos_bodies, b, C, k)
    return jac_grad_omega(rho, rho_p, mu_adim)
    



def compute_coeff_grad_rnbp_rpf(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec):

    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    mu_p = mu_bodies_dim[id_primary]
    mu_s = mu_bodies_dim[id_secondary]
    
    epoch_time = compute_epoch_time(tau, tau_vec, t_vec, epoch_t0)
    
    state_bodies = get_body_states_rnbp(epoch_time, naif_id_bodies, observer_id, reference_frame)
    
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
    grad_om_adim = grad_omega_rnbp_rpf(rho, pos_bodies, b, C, k, mu_adim)
    
    return np.array([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13]), grad_om_adim



def compute_coeff_jac_grad_rnbp_rpf(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec):

    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    mu_p = mu_bodies_dim[id_primary]
    mu_s = mu_bodies_dim[id_secondary]
    
    epoch_time = compute_epoch_time(tau, tau_vec, t_vec, epoch_t0)
    
    state_bodies = get_body_states_rnbp(epoch_time, naif_id_bodies, observer_id, reference_frame)
    
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
    jac_grad_om_adim = jac_grad_omega_rnbp_rpf(rho, pos_bodies, b, C, k, mu_adim)
    
    
    return np.array([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13]), jac_grad_om_adim




def dynamics(tau, state, control, p, auxdata):
    rho = state[:3]
    eta = state[3:6]
    
    model = auxdata["model"]
    
    if model == 1: # crtbp
        mu_crtbp = auxdata["mu_crtbp"]

        b = compute_coeff_crtbp()
        grad_om_adim = grad_omega_crtbp(rho, mu_crtbp)
    elif model == 2: # bcrfbp
        mu_crtbp = auxdata["mu_crtbp"]
        mu_sun = auxdata["mu_sun"]
        a_sun = auxdata["a_sun"]
        om_sun = auxdata["om_sun"]
        sun_angle_t0 = auxdata["sun_angle_t0"]
        
        b = compute_coeff_bcrfbp(tau, mu_sun, a_sun, om_sun, sun_angle_t0)
        grad_om_adim = grad_omega_bcrfbp(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0)
    elif model == 3:
        id_primary = auxdata["id_primary"]
        id_secondary = auxdata["id_secondary"]
        mu_bodies_dim = auxdata["mu_bodies"]
        naif_id_bodies = auxdata["naif_id_bodies"]
        observer_id = auxdata["observer_id"]
        reference_frame = auxdata["reference_frame"]
        epoch_t0 = auxdata["epoch_t0"]
        tau_vec = auxdata["tau_vec"]
        t_vec = auxdata["t_vec"]
        b, grad_om_adim = compute_coeff_grad_rnbp_rpf(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec)
    
    return dynamics_coeff(state, b, grad_om_adim)



def dynamics_coeff(state, b, grad_om_adim):
    rho = state[:3]
    eta = state[3:6]
    
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = b
    
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
    deta = b[:3] + rho_dot_matrix @ eta + rho_matrix @ rho + b13 * grad_om_adim
    
    return np.concatenate((drho, deta))



def jacobian_coeff(b, jac_grad_om):

    I3 = np.identity(3)
    Z3 = np.zeros((3, 3))
    
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = b
    
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

    A11 = Z3  # drhodot / drho
    A12 = I3  # drhodot / deta
    A21 = rho_matrix + b13 * jac_grad_om # detadot / drho
    A22 = rho_dot_matrix # detadot / deta

    # Assemble Jacobian
    return np.block([[A11, A12], [A21, A22]])



def jacobian(tau, state, control, p, auxdata):
    rho = state[:3]
    eta = state[3:6]
    
    model = auxdata["model"]
    
    if model == 1: # crtbp
        mu_crtbp = auxdata["mu_crtbp"]

        b = compute_coeff_crtbp()
        jac_grad_om_adim = jac_grad_omega_crtbp(rho, mu_crtbp)
    elif model == 2: # bcrfbp
        mu_crtbp = auxdata["mu_crtbp"]
        mu_sun = auxdata["mu_sun"]
        a_sun = auxdata["a_sun"]
        om_sun = auxdata["om_sun"]
        sun_angle_t0 = auxdata["sun_angle_t0"]
        
        b = compute_coeff_bcrfbp(tau, mu_sun, a_sun, om_sun, sun_angle_t0)
        jac_grad_om_adim = jac_grad_omega_bcrfbp(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0)
    elif model == 3:
        id_primary = auxdata["id_primary"]
        id_secondary = auxdata["id_secondary"]
        mu_bodies_dim = auxdata["mu_bodies"]
        naif_id_bodies = auxdata["naif_id_bodies"]
        observer_id = auxdata["observer_id"]
        reference_frame = auxdata["reference_frame"]
        epoch_t0 = auxdata["epoch_t0"]
        tau_vec = auxdata["tau_vec"]
        t_vec = auxdata["t_vec"]
        
        b, jac_grad_om_adim = compute_coeff_jac_grad_rnbp_rpf(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec)
    
    return jacobian_coeff(b, jac_grad_om_adim)



def u_bar_first_partials(state, mu_cr3bp):

    r_13 = ((state[0] + mu_cr3bp) * (state[0] + mu_cr3bp) +
            state[1] * state[1] + state[2] * state[2]) ** (-1.5)  # 1/r1^3
    r_23 = (((state[0] - 1.0 + mu_cr3bp) * (state[0] - 1.0 + mu_cr3bp)) +
            state[1] * state[1] + state[2] * state[2]) ** (-1.5)  # 1/r2^3

    u_bar = np.empty(3)  # preallocate array to store the partials

    u_bar[0] = mu_cr3bp * (state[0] - 1.0 + mu_cr3bp) * r_23 + \
        (1.0 - mu_cr3bp) * (state[0] + mu_cr3bp) * r_13 - state[0]
    u_bar[1] = mu_cr3bp * state[1] * r_23 + (1.0 - mu_cr3bp) * state[1] * r_13 - state[1]
    u_bar[2] = mu_cr3bp * state[2] * r_23 + (1.0 - mu_cr3bp) * state[2] * r_13

    return u_bar


def u_bar_second_partials(state, mu_cr3bp):

    r_12 = (state[0] + mu_cr3bp) * (state[0] + mu_cr3bp) + state[1] * state[1] + \
        state[2] * state[2]  # r1^2
    r_22 = ((state[0] - 1.0 + mu_cr3bp) * (state[0] - 1.0 + mu_cr3bp)) + state[1] * state[1] + \
        state[2] * state[2]  # r2^2

    r_13 = r_12 ** (-1.5)  # 1/r1^3
    r_23 = r_22 ** (-1.5)  # 1/r2^3
    r_15 = r_12 ** (-2.5)  # 1/r1^5
    r_25 = r_22 ** (-2.5)  # 1/r2^5

    u_bar2 = np.empty((3, 3))

    u_bar2[0, 0] = mu_cr3bp * r_23 + (1.0 - mu_cr3bp) * r_13 - \
        3.0 * mu_cr3bp * (state[0] - 1.0 + mu_cr3bp) * (state[0] - 1.0 + mu_cr3bp) * r_25 - \
        3.0 * (state[0] + mu_cr3bp) * (state[0] + mu_cr3bp) * (1.0 - mu_cr3bp) * r_15 - 1.0

    u_bar2[0, 1] = 3.0 * state[1] * (mu_cr3bp + state[0]) * (mu_cr3bp - 1.0) * r_15 - \
        3.0 * mu_cr3bp * state[1] * (state[0] - 1.0 + mu_cr3bp) * r_25

    u_bar2[0, 2] = 3.0 * state[2] * (mu_cr3bp + state[0]) * (mu_cr3bp - 1.0) * r_15 - \
        3.0 * mu_cr3bp * state[2] * (state[0] - 1.0 + mu_cr3bp) * r_25

    u_bar2[1, 0] = u_bar2[0, 1]

    u_bar2[1, 1] = mu_cr3bp * r_23 - (mu_cr3bp - 1.0) * r_13 + \
        3.0 * state[1] * state[1] * (mu_cr3bp - 1.0) * r_15 - \
        3.0 * mu_cr3bp * state[1] ** 2 * r_25 - 1.0

    u_bar2[1, 2] = 3.0 * state[1] * state[2] * (mu_cr3bp - 1.0) * r_15 - \
        3.0 * mu_cr3bp * state[1] * state[2] * r_25

    u_bar2[2, 0] = u_bar2[0, 2]
    u_bar2[2, 1] = u_bar2[1, 2]

    u_bar2[2, 2] = mu_cr3bp * r_23 - (mu_cr3bp - 1.0) * r_13 + \
        3.0 * state[2] ** 2 * (mu_cr3bp - 1.0) * r_15 - 3.0 * mu_cr3bp * state[2] ** 2 * r_25

    return u_bar2


def dynamics_crtbp_ref(state, mu_cr3bp):
    
    u_bar = u_bar_first_partials(state, mu_cr3bp)
    
    print("u_bar: ", u_bar)

    dot_state = np.empty(6)  # preallocate array for state derivatives
    dot_state[0] = state[3]
    dot_state[1] = state[4]
    dot_state[2] = state[5]
    dot_state[3] = 2.0 * state[4] - u_bar[0]
    dot_state[4] = - 2.0 * state[3] - u_bar[1]
    dot_state[5] = - u_bar[2]

    return dot_state


def jacobian_crtbp_ref(states, mu_cr3bp):

    # first-order ODEs for STM
    x_stm = np.zeros((6, 6))
    x_stm[0:3, 3:6] = np.eye(3)
    x_stm[3:6, 0:3] = - u_bar_second_partials(states, mu_cr3bp)
    x_stm[3, 4] = 2.0
    x_stm[4, 3] = - 2.0
    
    return x_stm
    

def u_bar(state, mu_var):
    """Augmented potential (from Koon et al. 2011, chapter 2). """
    x_var, y_var, z_var = state
    mu1 = 1 - mu_var
    mu2 = mu_var
    # where r1 and r2 are expressed in rotating coordinates
    r1_var = ((x_var+mu2)**2 + y_var**2 + z_var**2)**(1/2)
    r2_var = ((x_var-mu1)**2 + y_var**2 + z_var**2)**(1/2)
    aug_pot = -1/2*(x_var**2+y_var**2) - mu1/r1_var - mu2/r2_var - 1/2*mu1*mu2
    return aug_pot



def dynamics_bcrfbp_ref(t, state, mu, mu_s, a_sun, om_sun, sun_angle_t0):
    
    x, y, z, xdot, ydot, zdot = state

    dx = np.zeros(6)

    sun_angle = om_sun * t + sun_angle_t0
    rs = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0])

    # Com_sunpute the relative positions
    re_sc = np.array([x + mu, y, z])
    rm_sc = np.array([x - 1 + mu, y, z])
    rs_sc = np.array([x - rs[0], y - rs[1], z - rs[2]])

    # Com_sunpute the distances cubed
    re_sc3 = np.dot(re_sc, re_sc)**1.5
    rm_sc3 = np.dot(rm_sc, rm_sc)**1.5
    rs_sc3 = np.dot(rs_sc, rs_sc)**1.5
    
    a_sun3 = a_sun**3

    # Compute the accelerations
    xddot = (2 * ydot + x 
             - (1 - mu) * (x + mu) / re_sc3 
             - mu * (x - 1 + mu) / rm_sc3 
             - mu_s * (x - rs[0]) / rs_sc3 
             - mu_s * rs[0] / a_sun3)

    yddot = (- 2*xdot + y
             - (1 - mu) * y / re_sc3 
             - mu * y / rm_sc3 
             - mu_s * (y - rs[1]) / rs_sc3 
             - mu_s * rs[1] / a_sun3)

    zddot = (- (1 - mu) * z / re_sc3 
             - mu * z / rm_sc3 
             - mu_s * (z - rs[2]) / rs_sc3 
             - mu_s * rs[2] / a_sun3)

    # Fill in the derivatives
    dx[0] = xdot
    dx[1] = ydot
    dx[2] = zdot
    dx[3] = xddot
    dx[4] = yddot
    dx[5] = zddot

    return dx
    
   
   
def jacobian_bcrfbp_ref(t, state, mu, mu_s, a_sun, om_sun, sun_angle_t0):

    x, y, z, xdot, ydot, zdot = state

    sun_angle = om_sun * t + sun_angle_t0
    rs = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0])
    rs0, rs1, rs2 = rs
    
    # Com_sunpute the relative positions
    re_sc = np.array([x + mu, y, z])
    rm_sc = np.array([x - 1 + mu, y, z])
    rs_sc = np.array([x - rs0, y - rs1, z])

    # Com_sunpute the distances cubed
    re_sc3 = np.dot(re_sc, re_sc)**1.5
    rm_sc3 = np.dot(rm_sc, rm_sc)**1.5
    rs_sc3 = np.dot(rs_sc, rs_sc)**1.5
    
    re_sc5 = np.dot(re_sc, re_sc)**2.5
    rm_sc5 = np.dot(rm_sc, rm_sc)**2.5
    rs_sc5 = np.dot(rs_sc, rs_sc)**2.5
        
    jac_x = np.array([
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1], 
        [-3.0*mu*(mu + x)**2/re_sc5 + mu/re_sc3 + 3.0*mu*(mu + x - 1)**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*(rs0 - x)**2/rs_sc5 - mu_s/rs_sc3 + 3.0*(mu + x)**2/re_sc5 - 1/re_sc3 + 1, 3.0*(mu*re_sc5*rs_sc5*y*(mu + x - 1) + mu_s*re_sc5*rm_sc5*(rs0 - x)*(rs1 - y) - rm_sc5*rs_sc5*y*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 3.0*z*(mu*re_sc5*rs_sc5*(mu + x - 1) - mu_s*re_sc5*rm_sc5*(rs0 - x) - rm_sc5*rs_sc5*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 0, 2, 0], 
        [3.0*(mu*re_sc5*rs_sc5*y*(mu + x - 1) + mu_s*re_sc5*rm_sc5*(rs0 - x)*(rs1 - y) - rm_sc5*rs_sc5*y*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), -3.0*mu*y**2/re_sc5 + mu/re_sc3 + 3.0*mu*y**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*(rs1 - y)**2/rs_sc5 - mu_s/rs_sc3 + 3.0*y**2/re_sc5 - 1/re_sc3 + 1, 3.0*z*(mu*re_sc5*rs_sc5*y + mu_s*re_sc5*rm_sc5*(-rs1 + y) + rm_sc5*rs_sc5*y*(1 - mu))/(re_sc5*rm_sc5*rs_sc5), -2, 0, 0], 
        [3.0*z*(mu*re_sc5*rs_sc5*(mu + x - 1) - mu_s*re_sc5*rm_sc5*(rs0 - x) - rm_sc5*rs_sc5*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 3.0*z*(mu*re_sc5*rs_sc5*y + mu_s*re_sc5*rm_sc5*(-rs1 + y) + rm_sc5*rs_sc5*y*(1 - mu))/(re_sc5*rm_sc5*rs_sc5), -3.0*mu*z**2/re_sc5 + mu/re_sc3 + 3.0*mu*z**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*z**2/rs_sc5 - mu_s/rs_sc3 + 3.0*z**2/re_sc5 - 1/re_sc3, 0, 0, 0]
        ])

    return jac_x
