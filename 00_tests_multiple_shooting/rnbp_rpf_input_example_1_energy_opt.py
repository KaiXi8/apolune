import numpy as np
import cppad_py
from scipy.integrate import odeint
import dynamics.rnbp_rpf_dynamics as rnbp_rpf
import propagation.propagator as propagation
import spiceypy as spice
import init.load_kernels as krn


def objective_fun(states, controls, p, aux):
    squared_norm_sum = np.sum(np.sum(controls**2, axis=1))
    return squared_norm_sum  
     

def event_constraints(x0, u0, xf, uf, p, aux):
    man_indices = aux['param']['man_index']
    n_event_con = aux['problem']['n_event_con']
    events = np.zeros(n_event_con, dtype=cppad_py.a_double)
    
    if man_indices[0] == 0:
        events[0:6] = x0 - np.concat((np.zeros(3), u0)) 
    else:
        events[0:6] = x0

    events[6:12] = xf 
    
    return events


def path_constraints(states, controls, p, auxdata):
    return np.empty(0)


def misc_constraints(states, controls, p, auxdata):
    return np.empty(0)


def initial_guess(auxdata):
    n_x = auxdata['problem']['n_x']
    n_u = auxdata['problem']['n_u']
    n_man = auxdata['param']['n_man']
    
    time = auxdata['param']['time_vec']
    x0 = auxdata['param']['x0']
    
    if auxdata['problem']['free_tf'] == 1:
        tf = auxdata['param']['tf']
        time = time * tf
        p_guess = np.array([tf])
    else:
        p_guess = np.empty(0)
    
    u_guess = np.zeros((n_man, n_u))
    x_guess = propagation.propagate(rnbp_rpf.dynamics, x0, u_guess, p_guess, time, auxdata)

    return x_guess, u_guess, p_guess


def settings():

    G = 6.67408e-20  # [km^3 kg^−1 s^−2]
    # AU = 149597870.693 # km
    AU = 1.495978706136889e+08 # [km] from SPICE
    
    inertial_frame = "J2000"
    abcorr_flag = "None"
    origin_frame = "SSB"
    
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
    
    krn.load_kernels()
    
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
    
    id_primary = id_ear[0]
    id_secondary = id_moo[0]
    
    naif_id_bodies = [id_mer[1], id_ven[1], id_ear[1], id_mar[1], id_jup[1], id_sat[1], id_ura[1], id_nep[1], id_plu[1], id_moo[1], id_sun[1]]
    mu_bodies = np.array([GM_mer, GM_ven, GM_ear, GM_mar, GM_jup, GM_sat, GM_ura, GM_nep, GM_plu, GM_moo, GM_sun])

    MU = mu_bodies[id_primary] + mu_bodies[id_secondary]
    
    if id_primary == id_ear[0] and id_secondary == id_moo[0]:
        LU = 384400.0 # [km]
    elif id_primary == id_sun[0] and id_secondary == id_ear[0]:
        LU = 1*AU 
    else:
        raise ValueError("Invalid primary and secondary body combination")
    
    
    TU = np.sqrt(LU**3 / MU) # scaling factor for time [s]
    VU = LU / TU # [km/s]
    om0 = 1/TU # constant [1/s] 
    
    # [km^3 s−2] to [adim]
    # mu_bodies = mu_bodies / VU**2 / LU 
    # mu_bodies = mu_bodies * TU**2 / LU**3
    
    et0 = spice.str2et('1 Jan 2017 00:00:00 TDB');
    
    n_x = 6
    n_u = 3
    N = 100
    Ns = N - 1
    
    free_tf = 1
    
    # node indices where maneuvers are applied; numpy array within [0, Ns]
    man_index = np.array([0, 30, 60, Ns]) # works with N = 100
    # man_index = np.array([0, 10, 20, 40, 50, 60, 70, 80, 90, Ns]) # works with N = 100
#     man_index = np.array([0, Ns]) # works with N = 100
    
    # initial and final boundary conditions
    x0 = np.array([8.2338046140454002e-01, 0, 1.3886061447073000e-02, 0, 1.2947638542136800e-01, 0]) # adim
    xf = np.array([1.1194488032482977e+00, 2.3502976908423845e-13, -1.1371675247773910e-02, 2.2969820490104098e-13, 1.7876223257953414e-01, -2.7620527393024119e-13]) # adim
    t0 = 0.0
    tf = 6.9083604301186052e-01 * 86400 / TU
    
    # bounds for states, controls, and dv per maneuver
    states_lower = -10.0 * np.ones(n_x)
    states_upper = 10.0 * np.ones(n_x)
    controls_lower = -10.0 * np.ones(n_u)
    controls_upper = 10.0 * np.ones(n_u)
    dv_max = 10.0
    
    # n_p: number of free parameters in the optimization
    # here: no free parameters
    n_man = len(man_index)
    n_p = 1
    
    tf_ind = 0
    
    # lower and upper bounds of the free variable vector p
#     p_lower = np.empty(0)
#     p_upper = np.empty(0)
    p_lower = np.array([0.01])
    p_upper = np.array([0.1])
    
    # lower and upper bounds of the event constraints
    event_constraints_lower = np.concatenate((x0, xf))
    event_constraints_upper = event_constraints_lower
    
    # lower and upper bounds of the remaining constraints
    # here: no additional constraints
    misc_constraints_lower = np.empty(0)
    misc_constraints_upper = np.empty(0)
    
    # allocate functions for objective, path constraints (can be None), event constraints (can be None), and misc constraints (can be None)
    objective_function = objective_fun
    path_constraints_function = None
    event_constraints_function = event_constraints
    misc_constraints_function = None
    
    # functions for dynamics and corresponding jacobian
    jacobian_x_function = rnbp_rpf.jacobian_x_standalone
    dynamics_function = rnbp_rpf.dynamics
    
    # number of path, event, and misc constraints
    n_path_con = 0
    n_event_con = 12
    n_misc_con = 0
    
    # tolerances for the integrator
    # we use different values for piecewise integration (from t_k to t_k+1) and full integration (from t0 to tf)
    ode_atol_piecewise = 1e-10
    ode_rtol_piecewise = 1e-10
    ode_atol = 1e-12
    ode_rtol = 1e-12
    
    # discretized time vector
    time = np.linspace(t0, tf, N)


    # -----
    # some internal definitions -> probably no need to change anything here for now
    
    x_len = N*n_x
    u_len = n_man*n_u
    t_len = 0
    p_len = n_p
    
    xu_len = x_len + u_len
    sol_len = xu_len + t_len + p_len
    x_ind = slice(0, x_len) 
    u_ind = slice(x_len, xu_len)
    p_ind = slice(xu_len, xu_len+p_len)
    tf_ind_sol = xu_len + tf_ind
    
    stm_x_len = n_x*n_x
    stm_t_len = n_x * free_tf
    
    V_len = n_x + stm_x_len + stm_t_len
        
    x_ind_stm = slice(0, n_x)
    stm_x_ind = slice(n_x, stm_x_len+n_x)
    stm_t_ind = slice(stm_x_len+n_x, stm_x_len+n_x+stm_t_len)
    
    V0 = np.zeros(V_len);
    V0[stm_x_ind] = np.identity(n_x).flatten()
     
    x_stm = np.zeros((N,n_x));
    stm_x = np.zeros((Ns,n_x*n_x));
    stm_t = np.zeros((Ns,stm_t_len));
    
    if free_tf == 1:
        time = np.linspace(0.0, 1.0, N)
    else:
        time = np.linspace(t0, tf, N)
    
    if man_index[0] == 0:
        man_index_defects = man_index[1:]
        n_man_defects = n_man - 1
    else:
        man_index_defects = man_index
        n_man_defects = n_man
    
    n_con = n_path_con + n_event_con + n_misc_con
    
    scaling_dict = {'LU': LU, 'VU': VU, 'TU': TU}
    lengths_dict = {'x_len': x_len, 'u_len': u_len, 'xu_len': xu_len, 'p_len': p_len, 't_len': t_len, 'sol_len': sol_len, 'stm_x_len': stm_x_len, 'stm_t_len': stm_t_len, 'V_len': V_len}
    indices_dict = {'x_ind': x_ind, 'u_ind': u_ind, 'p_ind': p_ind, 'tf_ind': tf_ind, 'tf_ind_sol': tf_ind_sol, 'x_ind_stm': x_ind_stm, 'stm_x_ind': stm_x_ind, 'stm_t_ind': stm_t_ind}
    problem_dict = {'n_x': n_x, 'n_u': n_u, 'n_p': n_p, 'N': N, 'Ns': Ns, 'free_tf': free_tf, 'n_path_con': n_path_con, 'n_event_con': n_event_con, \
        'n_misc_con': n_misc_con, 'n_con': n_con, 'dynamics': dynamics_function, 'objective': objective_function, 'jacobian_x': jacobian_x_function}
    constraints_dict = {'path_constraints': path_constraints_function, 'event_constraints': event_constraints_function, 'misc_constraints': misc_constraints_function}
    param_dict = {'naif_id_bodies': naif_id_bodies, 'mu_bodies': mu_bodies, 'id_primary': id_primary, 'id_secondary': id_secondary, \
        'om0': om0, 'et0': et0, 'inertial_frame': inertial_frame, 'abcorr_flag': abcorr_flag, 'origin_frame': origin_frame, 't0': t0, 'tf': tf, 'x0': x0, 'xf': xf, 'time_vec': time, 'n_man': n_man, 'man_index': man_index, \
        'n_man_defects': n_man_defects, 'man_index_defects': man_index_defects, 'dv_max': dv_max, \
        'ode_atol': ode_atol, 'ode_rtol': ode_rtol, 'ode_atol_piecewise': ode_atol_piecewise, 'ode_rtol_piecewise': ode_rtol_piecewise}
    boundaries_dict = {'states_lower': states_lower, 'states_upper': states_upper, 'controls_lower': controls_lower, 'controls_upper': controls_upper, \
        'p_lower': p_lower, 'p_upper': p_upper, 'event_constraints_lower': event_constraints_lower, 'event_constraints_upper': event_constraints_upper, \
        'misc_constraints_lower': misc_constraints_lower, 'misc_constraints_upper': misc_constraints_upper}
    discretization_dict = {'V0': V0, 'x': x_stm, 'stm_x': stm_x, 'stm_t': stm_t}
    
    auxdata = {'problem': problem_dict, 'lengths': lengths_dict, 'indices': indices_dict, 'param': param_dict, 'discretization': discretization_dict, 'boundaries': boundaries_dict, \
        'constraints': constraints_dict, 'scaling': scaling_dict}
    
    # generate initial guess 
    x_guess, u_guess, p_guess = initial_guess(auxdata)
    initial_guess_dict = {'state': x_guess, 'control': u_guess, 'p': p_guess}
    auxdata['initial_guess'] = initial_guess_dict
    
    return auxdata
    
