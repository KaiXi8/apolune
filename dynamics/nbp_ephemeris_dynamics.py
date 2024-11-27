import numpy as np
import spiceypy as spice
import init.load_kernels as krn


""" Dynamics for synodic n-body RPF

            Parameters
            ----------
            t : scalar
                Time [adim]
            state : 1d array
                State in inertial ephemeris frame [adim]
            input_dict : dictionary
                Contains relevant parameters 

            Return
            ------
            1d array
                Derivative of states in inertial ephemeris frame [adim]

"""
def dynamics(t, state, control, p, input_dict):

    et0 = input_dict["param"]["et0"]
    naif_id_bodies = input_dict["param"]["naif_id_bodies"]
    mu_bodies_dim = input_dict["param"]["mu_bodies"]
    id_primary = input_dict["param"]["id_primary"]
    LU = input_dict["scaling"]["LU"]
    VU = input_dict["scaling"]["VU"]
    TU = input_dict["scaling"]["TU"]
    inertial_frame = input_dict["param"]["inertial_frame"]
    abcorr_flag = input_dict["param"]["abcorr_flag"]
    origin_frame = input_dict["param"]["origin_frame"]

    mu_bodies = mu_bodies_dim / LU**3 * TU**2
    num_bodies = len(mu_bodies)
    
    r_osc = state[0:3] # [adim]
    
    et = et0 + t * TU
               
    # retrieve states of celestial bodies from SPICE and compute perturbing accelerations
    vdot = np.zeros(3)
    for i in range(num_bodies):
        if i == id_primary:
            continue
        # state_body_dim, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_body_dim, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(naif_id_bodies[id_primary]))
        r_ob = state_body_dim[0:3] / LU
        r_bsc = r_osc - r_ob # position of spacecraft wrt to body
        vdot = vdot - mu_bodies[i] * (r_bsc / np.linalg.norm(r_bsc)**3 + r_ob / np.linalg.norm(r_ob)**3)
        
    vdot = vdot - mu_bodies[id_primary] * r_osc / np.linalg.norm(r_osc)**3

    return np.concatenate((state[3:6], vdot)) # [adim, adim]



""" Computes the Jacobian matrix in the inertial ephemeris frame

            Parameters
            ----------
            t : scalar
                Time [adim]
            state : 1d array
                State in inertial ephemeris frame [adim]
            input_dict : dictionary
                Contains relevant parameters 

            Return
            ------
            jac: 6x6 array
                Jacobian in inertial ephemeris frame [adim]

"""
def jacobian_x(t, state, control, p, input_dict):

    et0 = input_dict["param"]["et0"]
    naif_id_bodies = input_dict["param"]["naif_id_bodies"]
    mu_bodies_dim = input_dict["param"]["mu_bodies"]
    id_primary = input_dict["param"]["id_primary"]
    LU = input_dict["scaling"]["LU"]
    VU = input_dict["scaling"]["VU"]
    TU = input_dict["scaling"]["TU"]
    inertial_frame = input_dict["param"]["inertial_frame"]
    abcorr_flag = input_dict["param"]["abcorr_flag"]
    origin_frame = input_dict["param"]["origin_frame"]
    
    I3 = np.identity(3)
    Z3 = np.zeros((3, 3))

    mu_bodies = mu_bodies_dim / LU**3 * TU**2
    num_bodies = len(mu_bodies)
    
    r_osc = state[0:3] # [adim]
    
    et = et0 + t * TU
               
    # retrieve states of celestial bodies from SPICE and compute perturbing accelerations
    norm_r_osc = np.linalg.norm(r_osc)
    aag = - mu_bodies[id_primary] * r_osc / norm_r_osc**3
    omega = mu_bodies[id_primary] * ( 3 * np.outer(r_osc, r_osc) - I3 * norm_r_osc**2) / norm_r_osc**5
    for i in range(num_bodies):
        if i == id_primary:
            continue
        # state_body_dim, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        state_body_dim, _ = spice.spkezr(str(naif_id_bodies[i]), et, inertial_frame, str(abcorr_flag), str(naif_id_bodies[id_primary]))
        r_ob = state_body_dim[0:3] / LU
        
        r_bsc = r_osc - r_ob # position of spacecraft wrt to body
        norm_r_bsc = np.linalg.norm(r_bsc)
        aag = aag - mu_bodies[i] * (r_bsc/norm_r_bsc**3 + r_ob / np.linalg.norm(r_ob)**3)
        omega = omega + mu_bodies[i] * (3 * np.outer(r_bsc, r_bsc) - I3 * norm_r_bsc**2) / norm_r_bsc**5

    # sub-matrices for jacobian
    A11 = Z3
    A12 = I3
    A21 = omega
    A22 = Z3

    # Assemble Jacobian
    jac = np.block([[A11, A12], [A21, A22]])

    return jac



""" Computes the dynamics and STM in the inertial ephemeris frame

            Parameters
            ----------
            t : scalar
                Time [adim]
            state_stm : 1d array
                Flattened state and STM matrix in inertial ephemeris frame [adim]
            input_dict : dictionary
                Contains parameters for inertial ephemeris frame

            Return
            ------
            1d array
                Flattened derivative of state and STM matrix in inertial ephemeris frame [adim]

"""
def dynamics_stm(t, state_stm, control, p, input_dict):
    
    state = state_stm[0:6] # [adim]
    stm_x = state_stm[6:42].reshape((6, 6)) # [adim]

    state_dot = dynamics(t, state, control, p, input_dict)
    jac_x = jacobian_x(t, state, control, p, input_dict)
    
    dstm = jac_x.dot(stm_x) 
    
    return np.concatenate((state, dstm.flatten())) # [adim, adim]



