import numpy as np
import copy
import spiceypy as spice
import init.load_kernels as krn
import frames.rnbp_rpf_transformations_nonuniform as rnbp
# import frames.rnbp_rpf_transformations_nonuniform as rnbp_nonuniform
# import frames.rnbp_rpf_transformations_uniform as rnbp_uniform


def get_transformation_matrix_synToIn(t):
    cost = np.cos(t)
    sint = np.sin(t)
    R = np.array([
        [cost, -sint, 0.0],
        [sint, cost, 0.0],
        [0.0, 0.0, 1.0]
        ])
    Rdot = np.array([
        [-sint, -cost, 0.0],
        [cost, -sint, 0.0],
        [0.0, 0.0, 0.0]
        ])

    return R, Rdot


def get_transformation_matrix_inToSyn(t):
    cost = np.cos(t)
    sint = np.sin(t)
    R = np.array([
        [cost, sint, 0.0],
        [-sint, cost, 0.0],
        [0.0, 0.0, 1.0]
        ])
    Rdot = np.array([
        [-sint, cost, 0.0],
        [-cost, -sint, 0.0],
        [0.0, 0.0, 0.0]
        ])

    return R, Rdot



# observer_primary_id: 0 (barycenter pf P1 and P2), 1 (origin at P1), 2 (origin at P2)
def synodicToInertial(t_syn, state_syn, mu_crtbp, observer_primary_id = 1):
    
    if observer_primary_id == 0:
        state_obs = np.zeros(6)
    elif observer_primary_id == 1:
        state_obs = np.concatenate((np.array(-mu_crtbp), np.zeros(5)))
    elif observer_primary_id == 2:
        state_obs = np.concatenate((np.array(1 - mu_crtbp), np.zeros(5)))
    else:
        raise Exception("observer_primary_id must be 0, 1 or 2") 
    
    t_syn = np.atleast_1d(t_syn)  # Ensures time is at least a 1D array
    state_syn = np.atleast_2d(state_syn)  # Ensures state is at least a 2D array
    
    num_elm = len(t_syn)
    
    # Initialize states in synodic coordinates
    r_inertial = np.zeros((num_time, 3))
    v_inertial = np.zeros((num_time, 3))

    for i in range(num_time):
        state_syn_translated = state_syn[i] - state_obs
        R, Rdot = get_transformation_matrix_synToIn(t_syn[i])
        r_inertial[i] = R @ state_syn_translated[0:3]
        v_inertial[i] = R @ state_syn_translated[3:6] + Rdot @ state_syn_translated[0:3]
        
    return np.hstack((r_inertial, v_inertial))
    


def inertialToSynodic(t_inertial, state_inertial, mu_crtbp, observer_primary_id = 1):
    
    if observer_primary_id == 0:
        state_obs = np.zeros(6)
    elif observer_primary_id == 1:
        state_obs = np.concatenate((np.array(-mu_crtbp), np.zeros(5)))
    elif observer_primary_id == 2:
        state_obs = np.concatenate((np.array(1 - mu_crtbp), np.zeros(5)))
    else:
        raise Exception("observer_primary_id must be 0, 1 or 2") 
    
    t_inertial = np.atleast_1d(t_inertial)  # Ensures time is at least a 1D array
    state_inertial = np.atleast_2d(state_inertial)  # Ensures state is at least a 2D array
    
    num_elm = len(t_inertial)
    
    # Initialize states in synodic coordinates
    r_synodic = np.zeros((num_time, 3))
    v_synodic = np.zeros((num_time, 3))

    for i in range(num_time):
        R, Rdot = get_transformation_matrix_inToSyn(t_inertial[i])
        r_synodic[i] = R @ state_inertial[i,0:3] + state_obs[0:3]
        v_synodic[i] = R @ state_inertial[i,3:6] + Rdot @ state_inertial[i,0:3] + state_obs[3:6]
        
    return np.hstack((r_synodic, v_synodic))



def synodicToInertialEphemeris(t_syn, state_syn, input_dict, two_body_flag = 0):
    
    input_dict_crtbp = copy.deepcopy(input_dict)
    
    if two_body_flag == 1 and len(input_dict["mu_bodies"]) > 2:
        id_primary = input_dict_crtbp["id_primary"]
        id_secondary = input_dict_crtbp["id_secondary"]
        
        mu_bodies = np.array([input_dict["mu_bodies"][id_primary], input_dict["mu_bodies"][id_secondary]])
        naif_id_bodies = np.array([input_dict["naif_id_bodies"][id_primary], input_dict["naif_id_bodies"][id_secondary]])
        
        input_dict_crtbp["mu_bodies"] = mu_bodies
        input_dict_crtbp["naif_id_bodies"] = naif_id_bodies

    state_inertial = rnbp.synodicToInertial(t_syn, state_syn, input_dict_crtbp)

    return state_inertial




def inertialEphemerisToSynodic(t_inertial, state_inertial, input_dict, two_body_flag = 0):
    
    input_dict_crtbp = copy.deepcopy(input_dict)
    
    if two_body_flag == 1 and len(input_dict["mu_bodies"]) > 2:
        id_primary = input_dict_crtbp["id_primary"]
        id_secondary = input_dict_crtbp["id_secondary"]
        
        mu_bodies = np.array([input_dict["mu_bodies"][id_primary], input_dict["mu_bodies"][id_secondary]])
        naif_id_bodies = np.array([input_dict["naif_id_bodies"][id_primary], input_dict["naif_id_bodies"][id_secondary]])
        
        input_dict_crtbp["mu_bodies"] = mu_bodies
        input_dict_crtbp["naif_id_bodies"] = naif_id_bodies

    state_synodic = rnbp.inertialToSynodic(t_inertial, state_inertial, input_dict_crtbp)

    return state_synodic
