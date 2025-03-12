import numpy as np
import copy
import spiceypy as spice
import load_kernels as krn
import rnbp_rpf_transformations_uniform as rnbp


def get_transformation_matrix_synToIn(t):
    cost = np.cos(t)
    sint = np.sin(t)
    R = np.array([
        [cost, -sint, 0.0],
        [sint, cost, 0.0],
        0.0, 0.0, 1.0]
        ])
    Rdot = np.array([
        [-sint, -cost, 0.0],
        [cost, -sint, 0.0],
        0.0, 0.0, 0.0]
        ])

    return R, Rdot


def get_transformation_matrix_inToSyn(t):
    cost = np.cos(t)
    sint = np.sin(t)
    R = np.array([
        [cost, sint, 0.0],
        [-sint, cost, 0.0],
        0.0, 0.0, 1.0]
        ])
    Rdot = np.array([
        [-sint, cost, 0.0],
        [-cost, -sint, 0.0],
        0.0, 0.0, 0.0]
        ])

    return R, Rdot



# observer_primary_id: 0 (barycenter pf P1 and P2), 1 (origin at P1), 2 (origin at P2)
def synodicToInertial(t_syn, state_syn, mu_crtbp, scaling_dict, observer_primary_id = 1):
    
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



def synodicToInertialEphemeris(t_syn, state_syn, input_dict, full_model = 0):
    
    input_dict_bcrfbp = copy.deepcopy(input_dict)
    
    if full_model == 1 and len(input_dict["mu_bodies"]) > 2:
        id_primary = input_dict_bcrfbp["id_primary"]
        id_secondary = input_dict_bcrfbp["id_secondary"]
        
        mu_bodies = np.array([input_dict["mu_bodies"][id_primary], input_dict["mu_bodies"][id_secondary], input_dict["mu_bodies"][id_tertiary]])
        naif_id_bodies = np.array([input_dict["naif_id_bodies"][id_primary], input_dict["naif_id_bodies"][id_secondary], input_dict["naif_id_bodies"][id_tertiary]])
        
        input_dict_bcrfbp["mu_bodies"] = mu_bodies
        input_dict_bcrfbp["naif_id_bodies"] = naif_id_bodies

    state_inertial = rnbp.synodicToInertial(t_syn, state_syn, input_dict_bcrfbp)

    return state_inertial




def inertialEphemerisToSynodic(t_inertial, state_inertial, input_dict, full_model = 0):
    
    input_dict_bcrfbp = copy.deepcopy(input_dict)
    
    if full_model == 1 and len(input_dict["mu_bodies"]) > 3:
        id_primary = input_dict_bcrfbp["id_primary"]
        id_secondary = input_dict_bcrfbp["id_secondary"]
        id_tertiary = input_dict_bcrfbp["id_tertiary"]
        
        mu_bodies = np.array([input_dict["mu_bodies"][id_primary], input_dict["mu_bodies"][id_secondary], input_dict["mu_bodies"][id_tertiary]])
        naif_id_bodies = np.array([input_dict["naif_id_bodies"][id_primary], input_dict["naif_id_bodies"][id_secondary], input_dict["naif_id_bodies"][id_tertiary]])
        
        input_dict_bcrfbp["mu_bodies"] = mu_bodies
        input_dict_bcrfbp["naif_id_bodies"] = naif_id_bodies

    state_synodic = rnbp.inertialToSynodic(t_inertial, state_inertial, input_dict_bcrfbp)

    return state_synodic





def get_transformation_matrix_emToSunB1(moon_angle, om_em):
    cost = np.cos(moon_angle)
    sint = np.sin(moon_angle)
    R = np.array([
        [cost, sint, 0.0],
        [-sint, cost, 0.0],
        0.0, 0.0, 1.0]
        ])
    Rdot = om_em * np.array([
        [-sint, cost, 0.0],
        [-cost, -sint, 0.0],
        0.0, 0.0, 0.0]
        ])

    return R, Rdot
    
    

def sunB1ToEarthMoon(t_syn, state_sb1, aux):
    mu_s = aux['param']['mu_s']
    mu = aux['param']['mu']
    om_sun = aux['param']['om_sun']
    sun_angle_t0 = aux['param']['sun_angle_t0']
    a_sun = aux['param']['a_sun']
    
    sun_angle = om_sun * t_syn + sun_angle_t0
    moon_angle = np.pi - sun_angle
    
    om_em_adim = np.abs(om_sun) / (1 - np.abs(om_sun))
    
    R, Rdot = get_transformation_matrix_emToSunB1(moon_angle, om_em_adim)
    
    pos_sb1 = state_sb1[:3]
    vel_sb1 = state_sb1[3:6]
    
    pos_em = a_sun * ( pos_sb1 - np.array([1-1/(mu_s + 1), 0, 0]) ) @ R.T
    vel_em = np.sqrt((mu_s+1)/a_sun) * ( vel_sb1 @ R + pos_sb1 @ Rdot.T )
    
    return np.concatenate((pos_em, vel_em))
    