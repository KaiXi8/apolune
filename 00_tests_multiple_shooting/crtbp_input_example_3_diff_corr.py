import numpy as np
import cppad_py
from scipy.integrate import odeint
import dynamics.crtbp_dynamics as crtbp
import propagation.propagator as propagation



def objective_fun(states, controls, p, aux):
    return np.zeros(1, dtype=cppad_py.a_double) 
     

def event_constraints(x0, u0, xf, uf, p, aux):
    man_indices = aux['param']['man_index']
    n_event_con = aux['problem']['n_event_con']
    events = np.zeros(n_event_con, dtype=cppad_py.a_double)
    
    events[0] = x0[1]
    events[1] = x0[3]
    events[2] = x0[5]

    events[3] = xf[1]
    events[4] = xf[3]
    events[5] = xf[5]
    
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
#     x_guess = propagation.propagate(crtbp.eqm_6_synodic, x0, p_guess, time, auxdata)
    x_guess = propagation.propagate_odeint(crtbp.dynamics, x0, u_guess, p_guess, time, auxdata)

    return x_guess, u_guess, p_guess



def find_T_crossing(ode_fun, x0, u, p, tspan, auxdata, plane='xz', direction=0, isterminal=1):

    time, state, time_event, state_stm_event, ind_event = propagation.propagate_event(ode_fun, x0, u, p, tspan, auxdata, plane=plane, direction=direction, isterminal=isterminal)

    return time, state, time_event, state_stm_event, ind_event



def settings():

    mu_earth_moon = 1.21506683e-2
    LU = 3.84405000e5 # km
    TU = 4.34811305 # days
    VU = 1.02323281 # km/s
    
    mu = mu_earth_moon
    
    pos_primary1 = np.array([-mu, 0, 0])
    pos_primary2 = np.array([1 - mu, 0, 0])
    
    n_x = 6
    n_u = 0
    N = 10
    Ns = N - 1
    
    free_tf = 1
    
    # node indices where maneuvers are applied; numpy array within [0, Ns]
    man_index = np.empty(0) # works with N = 100
    # man_index = np.array([0, 10, 20, 40, 50, 60, 70, 80, 90, Ns]) # works with N = 100
#     man_index = np.array([0, Ns]) # works with N = 100
    
    # initial and final boundary conditions
    x0 = np.array([0.824624564360438, 0, 0.0421950697023608, 0, 0.150473452492088, 0]) # adim
    xf = np.array([0.824624564360438, 0, 0.0421950697023608, 0, 0.150473452492088, 0]) # adim
    t0 = 0.0
#     tf = 1.3772
    tf = 1.4
 
    
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
    p_lower = np.array([1.0])
    p_upper = np.array([1.5])
    
    # lower and upper bounds of the event constraints
    event_constraints_lower = np.zeros(6)
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
    jacobian_x_function = crtbp.jacobian_x_numba
    dynamics_function = crtbp.dynamics_numba
    
    # number of path, event, and misc constraints
    n_path_con = 0
    n_event_con = 6
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
    
    man_index_defects = man_index
    n_man_defects = n_man
    
    n_con = n_path_con + n_event_con + n_misc_con
    
    scaling_dict = {'LU': LU, 'VU': VU, 'TU': TU}
    lengths_dict = {'x_len': x_len, 'u_len': u_len, 'xu_len': xu_len, 'p_len': p_len, 't_len': t_len, 'sol_len': sol_len, 'stm_x_len': stm_x_len, 'stm_t_len': stm_t_len, 'V_len': V_len}
    indices_dict = {'x_ind': x_ind, 'u_ind': u_ind, 'p_ind': p_ind, 'tf_ind': tf_ind, 'tf_ind_sol': tf_ind_sol, 'x_ind_stm': x_ind_stm, 'stm_x_ind': stm_x_ind, 'stm_t_ind': stm_t_ind}
    problem_dict = {'n_x': n_x, 'n_u': n_u, 'n_p': n_p, 'N': N, 'Ns': Ns, 'free_tf': free_tf, 'n_path_con': n_path_con, 'n_event_con': n_event_con, \
        'n_misc_con': n_misc_con, 'n_con': n_con, 'dynamics': dynamics_function, 'objective': objective_function, 'jacobian_x': jacobian_x_function}
    constraints_dict = {'path_constraints': path_constraints_function, 'event_constraints': event_constraints_function, 'misc_constraints': misc_constraints_function}
    param_dict = {'mu': mu, 't0': t0, 'tf': tf, 'x0': x0, 'xf': xf, 'time_vec': time, 'n_man': n_man, 'man_index': man_index, \
        'n_man_defects': n_man_defects, 'man_index_defects': man_index_defects, 'dv_max': dv_max, \
        'ode_atol': ode_atol, 'ode_rtol': ode_rtol, 'ode_atol_piecewise': ode_atol_piecewise, 'ode_rtol_piecewise': ode_rtol_piecewise,
                 'constant_names':['mu']}
    boundaries_dict = {'states_lower': states_lower, 'states_upper': states_upper, 'controls_lower': controls_lower, 'controls_upper': controls_upper, \
        'p_lower': p_lower, 'p_upper': p_upper, 'event_constraints_lower': event_constraints_lower, 'event_constraints_upper': event_constraints_upper, \
        'misc_constraints_lower': misc_constraints_lower, 'misc_constraints_upper': misc_constraints_upper}
    discretization_dict = {'V0': V0, 'x': x_stm, 'stm_x': stm_x, 'stm_t': stm_t}
    
    auxdata = {'problem': problem_dict, 'lengths': lengths_dict, 'indices': indices_dict, 'param': param_dict, 'discretization': discretization_dict, 'boundaries': boundaries_dict, \
        'constraints': constraints_dict, 'scaling': scaling_dict}
    
    
    tspan = np.array([t0, tf])
    plane = 'xz'
    direction = -1
    isterminal = 1
    _, _, time_events, _, _ = find_T_crossing(crtbp.dynamics, x0, 0.0, 0.0, tspan, auxdata, plane=plane, direction=direction, isterminal=isterminal)
    tf12_crossing = time_events[0]
    
    # generate initial guess 
    x_guess, u_guess, p_guess = initial_guess(auxdata)
    initial_guess_dict = {'state': x_guess, 'control': u_guess, 'p': p_guess}
    auxdata['initial_guess'] = initial_guess_dict
    
    return auxdata
    
