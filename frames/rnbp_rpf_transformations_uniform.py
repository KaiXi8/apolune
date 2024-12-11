import numpy as np
import spiceypy as spice


""" Definition of the synodic n-body RPF (for transformations between synodic and inertial coordinates)

            Parameters
            ----------
            id_primary : scalar
                ID of the primary in pos_bodies and mu_bodies
            id_secondary : scalar
                ID of the secondary in pos_bodies and mu_bodies
            pos_bodies : nx3 array
                Position vectors of the n bodies [km]
            vel_bodies : nx3 array
                Velocity vectors of the n bodies [km/s]
            num_bodies: scalar
                number of bodies
            mu_bodies: 1d array
                Mu values for bodies [km^3/s^2]

            Return
            ------
            b, b1d: 1d arrays
                Barycenter vector and derivative. Units: b [km], b1d [km/s]
            k, k1d: scalars
                Scaling factor and derivative (instantaneous distance between primary and secondary). Units: k [km], k1d [km/s]
            C, C1d: 3x3 arrays
                Rotation matrix and derivative. Units: C [-], C1d [1/s]  

"""
def frame_rnbprpf_transform(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies):
    # Initialize acceleration and jerk arrays for only the primary and secondary bodies
    A = np.zeros((2, 3))

    # Calculate accelerations and jerks for primary and secondary bodies
    for idx, i in enumerate((id_primary, id_secondary)):
        for j in range(num_bodies):
            if j == i:
                continue  # Skip self-interaction
            
            R_vec = pos_bodies[j] - pos_bodies[i]
            norm_R = np.sqrt(np.dot(R_vec, R_vec))
            
            # Calculate acceleration and jerk (eq. 3.17)
            acc = (mu_bodies[j] / norm_R**3) * R_vec
            
            # Sum up acceleration and jerk for the primary and secondary bodies only
            A[idx] += acc # [km^3/s^2]

    # Extract primary and secondary positions, velocities, accelerations, and jerks
    Rp = pos_bodies[id_primary] # position of primary [km]
    Rs = pos_bodies[id_secondary] # position of secondary [km]
    Vp = vel_bodies[id_primary] # velocity of primary [km]
    Vs = vel_bodies[id_secondary] # velocity of secondary [km]
    Ap = A[0] # acceleration of primary [km]
    As = A[1] # acceleration of secondary [km]

    mu_p = mu_bodies[id_primary] # gravitational parameter of primary [km^3/s^2]
    mu_s = mu_bodies[id_secondary] # gravitational parameter of secondary [km^3/s^2]

    # Calculate relative vectors and barycenter (eq. 3.16)
    r = Rs - Rp # [km] 
    v = Vs - Vp # [km/s]
    a = As - Ap # [km/s^2]

    # Primaries barycenter (eq. 3.2)
    b = (mu_s * Rs + mu_p * Rp) / (mu_s + mu_p) # [km]
    b1d = (mu_s * Vs + mu_p * Vp) / (mu_s + mu_p) # [km/s]
    
    # Scaling factor (eqs. 3.3 and 3.12)
    k = np.sqrt(np.dot(r, r)) # [km]
    k1d = np.dot(r, v) / k # [km/s]

    # Direction cosines and rotation matrix
    crv = np.cross(r, v)
    cra = np.cross(r, a)
    
    # angular momentum and derivative (eq. 3.15)
    h = np.linalg.norm(crv) # [km^2/s]
    h1d = np.dot(crv, cra) / h # [km^2/s^2]

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

    return b, b1d, k, k1d, C, C1d



""" Transformation inertial -> synodic n-body RPF

            Parameters
            ----------
            time : scalar or 1d array
                Time vector [adim]
            state_inertial : 1d or nx6 array
                States in inertial frame [km, km/s]
            rnbprpf_dict : dictionary
                Contains parameters for n-body RPF

            Return
            ------
            1d array or nx6 array
                States in inertial frame [km, km/s]

"""
def inertialToSynodic(time, state_inertial, rnbprpf_dict):

    om0 = rnbprpf_dict["om0"]
    et0 = rnbprpf_dict["et0"]
    naif_id_bodies = rnbprpf_dict["naif_id_bodies"]
    mu_bodies = rnbprpf_dict["mu_bodies"]
    id_primary = rnbprpf_dict["id_primary"]
    id_secondary = rnbprpf_dict["id_secondary"]
    inertial_frame = rnbprpf_dict["inertial_frame"]
    abcorr_flag = rnbprpf_dict["abcorr_flag"]
    origin_frame = rnbprpf_dict["origin_frame"]
    
    time = np.atleast_1d(time)  # Ensures time is at least a 1D array
    state_inertial = np.atleast_2d(state_inertial)  # Ensures state is at least a 2D array

    # Constants and dimensions
    num_bodies = len(mu_bodies)
    num_time = len(time)

    # Define positions and velocities in inertial frame 
    pos_inertial = state_inertial[:, :3] # [km]
    vel_inertial = state_inertial[:, 3:6] # [km/s]
    
    TU = 1 / om0
    
    # epoch time
    et = et0 + time * TU 

    # retrieve the states of celestial bodies at all times in time using SPICE
    pos_bodies_all = np.zeros((num_time, num_bodies, 3))
    vel_bodies_all = np.zeros((num_time, num_bodies, 3))
    for jj in range(num_bodies):
        # Get state vectors from SPICE
        state_body, _ = spice.spkezr(str(naif_id_bodies[jj]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        pos_bodies_all[:, jj, :] = np.array([state[:3] for state in state_body])  # Position, [km]
        vel_bodies_all[:, jj, :] = np.array([state[3:6] for state in state_body])  # Velocity, [km/s]

    # Initialize states in synodic coordinates
    rho_synodic = np.zeros((num_time, 3))
    eta_synodic = np.zeros((num_time, 3))

    for i in range(num_time):
        # Get positions and velocities of bodies at each time
        pos_bodies = pos_bodies_all[i, :, :] # [km]
        vel_bodies = vel_bodies_all[i, :, :] # [km/s]
        
        # compute relevant quantities for rotation and translation
        b, b1d, k, k1d, C, C1d = frame_rnbprpf_transform(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies)

        # transform inertial to synodic coordinates (eq. 3.1a)
        rho_synodic[i, :] = (pos_inertial[i] - b) @ C / k # [adim]
        eta_synodic[i, :] = (vel_inertial[i] - b1d - (k * C1d + k1d * C) @ rho_synodic[i, :]) @ C / (k * om0) # [adim]

    return np.hstack((rho_synodic, eta_synodic)) # [adim, adim]



""" Transformation synodic n-body RPF -> inertial

            Parameters
            ----------
            time : scalar or 1d array
                Time vector [adim]
            state_synodic : 1d or nx6 array
                States in synodic n-body RPF [adim]
            rnbprpf_dict : dictionary
                Contains parameters for n-body RPF

            Return
            ------
            1d array or nx6 array
                States in inertial frame [km, km/s]

"""
def synodicToInertial(time, state_synodic, rnbprpf_dict):

    om0 = rnbprpf_dict["om0"]
    et0 = rnbprpf_dict["et0"]
    naif_id_bodies = rnbprpf_dict["naif_id_bodies"]
    mu_bodies = rnbprpf_dict["mu_bodies"]
    id_primary = rnbprpf_dict["id_primary"]
    id_secondary = rnbprpf_dict["id_secondary"]
    inertial_frame = rnbprpf_dict["inertial_frame"]
    abcorr_flag = rnbprpf_dict["abcorr_flag"]
    origin_frame = rnbprpf_dict["origin_frame"]
    
    time = np.atleast_1d(time)  # Ensures time is at least a 1D array
    state_synodic = np.atleast_2d(state_synodic)  # Ensures state is at least a 2D array
    
    # Constants and dimensions
    num_bodies = len(mu_bodies)
    num_time = len(time)

    # Define positions and velocities in synodic frame
    rho_synodic = state_synodic[:, :3]  # Initial positions [km]
    eta_synodic = state_synodic[:, 3:6]  # Initial velocities [km/s]
    
    TU = 1 / om0
    
    # epoch time
    et = et0 + time * TU # [s]

    # retrieve the states of celestial bodies at all times in time using SPICE
    pos_bodies_all = np.zeros((num_time, num_bodies, 3))
    vel_bodies_all = np.zeros((num_time, num_bodies, 3))
    for jj in range(num_bodies):
        # Get state vectors from SPICE
        state_body, _ = spice.spkezr(str(naif_id_bodies[jj]), et, inertial_frame, str(abcorr_flag), str(origin_frame))
        pos_bodies_all[:, jj, :] = np.array([state[:3] for state in state_body])  # Position [km]
        vel_bodies_all[:, jj, :] = np.array([state[3:6] for state in state_body])  # Velocity [km/s]

    # Initialize states in inertial coordinates
    pos_inertial = np.zeros((num_time, 3))
    vel_inertial = np.zeros((num_time, 3))
    
    for i in range(num_time):
        # Get positions and velocities of bodies at each time
        pos_bodies = pos_bodies_all[i, :, :] # [km]
        vel_bodies = vel_bodies_all[i, :, :] # [km/s]
        
        # compute relevant quantities for rotation and translation
        b, b1d, k, k1d, C, C1d = frame_rnbprpf_transform(id_primary, id_secondary, pos_bodies, vel_bodies, num_bodies, mu_bodies)

        # transform synodic to inertial coordinates (eq. 3.1a)
        pos_inertial[i, :] = b + k*rho_synodic[i,:] @ C.T # [km]
        vel_inertial[i, :] = b1d + ( (k1d*C+k*C1d) @ rho_synodic[i,:] + om0*k*C @ eta_synodic[i,:] ) # [km/s]

    return np.hstack((pos_inertial, vel_inertial)) # [km, km/s]


