import numpy as np
import spiceypy as spice
import init.load_kernels as krn
from numba import jit

# test

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
            id_primary : scalar
                ID of the primary in pos_bodies and mu_bodies
            id_secondary : scalar
                ID of the secondary in pos_bodies and mu_bodies

            Return
            ------
            grad: 1d array
                Gradient of potential energy in n-body RPF [km^2/s^2]

"""
@jit(nopython=True, cache=True, nogil=True)
def grad_rnbprpf(pos_bodies, b, C, k, rho, mu_bodies, id_primary, id_secondary):
    # Synodic position of celestial bodies (eq. 3.20) 
    rhop = (pos_bodies - b) @ C / k # [adim]

    # Synodic relative position of spacecraft wrt celestial bodies (last term of eq. 3.19)
    diff_rho = rho - rhop # [adim]
    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) # [adim]

    # Gradient of potential energy (last term of eq. 3.19)
    factor = (mu_bodies / norm_rho**3)[:, np.newaxis] # [km^3/s^2]
    grad_om = factor * diff_rho / k # [km^2/s^2]

    # Sum along the rows to get the final gradient
    grad = grad_om.sum(axis=0) # [km^2/s^2]

    return grad 


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
            LU: scalar
                Scaling factor for length [km]
            VU: scalar
                Scaling factor for velocity [km/s]

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
def frame_rnbprpf(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies, LU, VU):

    # Initialize acceleration and jerk arrays for only the primary and secondary bodies
    A = np.zeros((2, 3))
    J = np.zeros((2, 3))

    # Calculate accelerations and jerks for primary and secondary bodies
    for idx, i in enumerate((id_primary, id_secondary)):
        for j in range(num_bodies):
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
    
    # Rescale positions, velocities, accelerations, and jerks
    pos_bodies *= LU # [km]
    vel_bodies *= VU # [km/s]
    A /= LU**2 # [km/s^2]
    J *= VU / LU**3 # [km/s^3]

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

    omega = input_dict["param"]["om0"]
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

    num_bodies = len(mu_bodies)
    identity3 = np.identity(3)
    
    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    # eq. 3.1b
    et = et0 + tau / omega
               
    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_bodies[i,:] = state_body # [km, km/s]

    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]
    
    b, b2d, k, k1d, k2d, C, C1d, C2d = \
        frame_rnbprpf(id_primary, id_secondary, pos_bodies / LU, vel_bodies / VU, num_bodies, mu_bodies, LU, VU)
 
    grad = grad_rnbprpf(pos_bodies, b, C, k, rho, mu_bodies, id_primary, id_secondary)
    
    # EoMs (eq. 3.21)
    drho = eta # [adim]
    deta = -1/omega * (2 * k1d / k * identity3 + 2 * C.T @ C1d) @ eta \
           -1/omega**2 * ( (k2d / k * identity3 + 2 * k1d /k * C.T @ C1d + C.T @ C2d) @ rho ) \
           -1/k / omega**2 * C.T @ b2d \
           -1/k**2 / omega**2 * grad # [adim]

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
            omega: scalar
                Scaling factor for time [1/s]
            mu_bodies: 1d array
                Mu values for bodies [km^3/s^2]

            Return
            ------
            jac: 6x6 array
                Jacobian in n-body RPF [adim]

"""
def jacobian_x(pos_bodies, b, rho, num_bodies, C, C1d, C2d, k, k1d, k2d, omega, mu_bodies):

    I3 = np.identity(3)
    Z3 = np.zeros((3, 3))
    
    # Synodic position of celestial bodies (eq. 3.20)
    rhop = (pos_bodies - b) @ C / k 

    # Synodic relative position of spacecraft wrt celestial bodies (last term of eq. 3.19)
    diff_rho = rho - rhop 
    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) 

    # Gradient of potential energy (eq. 5.19)
    gSTM_om1a = np.sum(mu_bodies / norm_rho**3) * I3

     # Compute om_mat for gradient calculation (eq. 5.19)
    om_mat = (mu_bodies / norm_rho**5)[:, np.newaxis] * diff_rho
    gSTM_om1b = np.zeros((3, 3))
    for jj in range(num_bodies):
        gSTM_om1b -= 3 * np.outer(om_mat[jj, :], diff_rho[jj, :])

    # sub-matrices for jacobian (eqs. 5.18 and 5.19)
    A11 = Z3  # df1/drho
    A12 = I3  # df1/deta
    A21 = (-1 / omega**2) * (k2d / k * I3 + 2 * k1d / k * C.T @ C1d + C.T @ C2d) - (1 / (omega**2 * k**3)) * (gSTM_om1a + gSTM_om1b)
    A22 = (-1 / omega) * (2 * k1d / k * I3 + 2 * C.T @ C1d)

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

    omega = input_dict["param"]["om0"]
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

    num_bodies = len(mu_bodies)
    identity3 = np.identity(3)
    
    rho = state[0:3] # [adim]
    eta = state[3:6] # [adim]
    
    # eq. 3.1b
    et = et0 + tau / omega
               
    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_bodies[i,:] = state_body # [km, km/s]

    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]
    
    b, _, k, k1d, k2d, C, C1d, C2d = \
        frame_rnbprpf(id_primary, id_secondary, pos_bodies / LU, vel_bodies / VU, num_bodies, mu_bodies, LU, VU)

    jac = jacobian_x(pos_bodies, b, rho, num_bodies, C, C1d, C2d, k, k1d, k2d, omega, mu_bodies)

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

    omega = input_dict["param"]["om0"]
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
    
    num_bodies = len(mu_bodies)
    identity3 = np.identity(3)
    
    rho = state_stm[0:3] # [adim]
    eta = state_stm[3:6] # [adim]
    stm_x = state_stm[6:42].reshape((6, 6)) # [adim]
    
    # eq. 3.1b
    et = et0 + tau / omega
               
    # retrieve states of celestial bodies from SPICE
    state_bodies = np.zeros((num_bodies, 6))
    for i in range(num_bodies):
        state_body, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_bodies[i,:] = state_body # [km, km/s]

    pos_bodies = state_bodies[:,0:3] # [km]
    vel_bodies = state_bodies[:,3:6] # [km/s]
    
    b, b2d, k, k1d, k2d, C, C1d, C2d = \
        frame_rnbprpf(id_primary, id_secondary, pos_bodies / LU, vel_bodies / VU, num_bodies, mu_bodies, LU, VU)
 
    grad = grad_rnbprpf(pos_bodies, b, C, k, rho, mu_bodies, id_primary, id_secondary)
    
    jac_x = jacobian_x(pos_bodies, b, rho, num_bodies, C, C1d, C2d, k, k1d, k2d, omega, mu_bodies)
    
    # EoMs (eq. 3.21)
    drho = eta # [adim]
    deta = -1/omega * (2 * k1d / k * identity3 + 2 * C.T @ C1d) @ eta \
           -1/omega**2 * ( (k2d / k * identity3 + 2 * k1d /k * C.T @ C1d + C.T @ C2d) @ rho ) \
           -1/k / omega**2 * C.T @ b2d \
           -1/k**2 / omega**2 * grad # [adim]
    dstm = jac_x.dot(stm_x) 
    
    return np.concatenate((drho, deta, dstm.flatten())) # [adim, adim, adim]

