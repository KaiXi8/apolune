import numpy as np
import cppad_py
from scipy.integrate import odeint
import dynamics.bcrfbp_dynamics as bcrfbp
import propagation.propagator as propagation



def objective_fun(states, controls, p, aux):
    obj = np.sum(p[:-1])
    return obj  
        

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
    
    n_man = auxdata['param']['n_man']
    misc_con = np.zeros(n_man, dtype=cppad_py.a_double)
    
    for i in range(n_man):
        u = controls[i]
        misc_con[i] = p[i]**2 - (u[0]**2 + u[1]**2 + u[2]**2)
    
    return misc_con



def initial_guess(auxdata):
    n_x = auxdata['problem']['n_x']
    n_u = auxdata['problem']['n_u']
    n_man = auxdata['param']['n_man']
    
    time = auxdata['param']['time_vec']
    x0 = auxdata['param']['x0']
    
    if auxdata['problem']['free_tf'] == 1:
        tf = auxdata['param']['tf']
        time = time * tf
        p_guess = np.concatenate((1e-1 * np.ones(n_man), np.array([tf])))
    else:
        p_guess = 1e-1 * np.ones(n_man)
    
    u_guess = np.zeros((n_man, n_u))
    x_guess = propagation.propagate(bcrfbp.dynamics, x0, u_guess, p_guess, time, auxdata)

    return x_guess, u_guess, p_guess


def settings():

    mu = 1.21506683e-2 # adim
    mu_s = 3.28900541e5 # adim
    a_sun = 3.88811143e2 # adim
    om_sun = -9.25195985e-1
    sun_angle_t0 = 0.0
    
    LU = 3.84405000e5 # km
#     TU = 4.34811305*86400 # seconds
    TU = 4.34811305*86400 # days
    VU = 1.02323281 # km/s
    
    pos_primary1 = np.array([-mu, 0, 0])
    pos_primary2 = np.array([1 - mu, 0, 0])
    
    n_x = 6
    n_u = 3
    N = 100
    Ns = N - 1
    
    free_tf = 1
    
    # node indices where maneuvers are applied; numpy array within [0, Ns]
    man_index = np.array([0, 30, 60, Ns]) # works with N = 100
#     man_index = np.array([0, 10, 20, 40, 50, 60, 70, 80, 90, Ns]) # works with N = 100
#     man_index = np.array([0, Ns]) # works with N = 100
    
    # initial and final boundary conditions
    x0 = np.array([0.870183, -0.059444, 0, -0.010471, -0.175136, 0]) # adim
    xf = np.array([1.11559, -0.056398, 0, -0.008555, 0.157211, 0]) # adim
    t0 = 0.0
#     tf = 12.34 / TU * 1.91 # time of flight
    tf = 12.34 / TU * 1.95 # time of flight
#     tf = 12.34 / TU * 2.1 # time of flight
    
    # bounds for states, controls, and dv per maneuver
    states_lower = -10.0 * np.ones(n_x)
    states_upper = 10.0 * np.ones(n_x)
    controls_lower = -10.0 * np.ones(n_u)
    controls_upper = 10.0 * np.ones(n_u)
    dv_max = 10.0
    
    # n_p: number of free parameters in the optimization
    # here: we use slack variables to solve the fuel-optimal problem with smooth derivatives
    n_man = len(man_index)
    n_p = n_man + free_tf
    
    tf_ind = n_man
    
    # lower and upper bounds of the free variable vector p
    # p_lower = np.zeros(n_man)
#     p_upper = 
    p_lower = np.concatenate((np.zeros(n_man), np.array([4.0])))
    p_upper = np.concatenate((dv_max * np.ones(n_man), np.array([7.0])))
    
    # lower and upper bounds of the event constraints
    event_constraints_lower = np.concatenate((x0, xf))
    event_constraints_upper = event_constraints_lower
    
    # lower and upper bounds of the remaining constraints
    # here: we need constraints on the slack variables to ensure that the obtained solution is the fuel-optimal solution
    misc_constraints_lower = np.zeros(n_man)
    misc_constraints_upper = dv_max * np.ones(n_man)
    
    # allocate functions for objective, path constraints (can be None), event constraints (can be None), and misc constraints (can be None)
    objective_function = objective_fun
    path_constraints_function = None
    event_constraints_function = event_constraints
    misc_constraints_function = misc_constraints
    
    # functions for dynamics and corresponding jacobian
    jacobian_x_function = bcrfbp.jacobian_x_numba
    dynamics_function = bcrfbp.dynamics_numba
    
    # number of path, event, and misc constraints
    n_path_con = 0
    n_event_con = 12
    n_misc_con = n_man
    
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
    
    stm_x_len = n_x * n_x
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
    indices_dict = {'x_ind': x_ind, 'u_ind': u_ind, 'p_ind': p_ind, 'x_ind_stm': x_ind_stm, 'stm_x_ind': stm_x_ind, 'tf_ind': tf_ind, 'tf_ind_sol': tf_ind_sol, 'stm_t_ind': stm_t_ind}
    problem_dict = {'n_x': n_x, 'n_u': n_u, 'n_p': n_p, 'N': N, 'Ns': Ns, 'free_tf': free_tf, 'n_path_con': n_path_con, 'n_event_con': n_event_con, \
        'n_misc_con': n_misc_con, 'n_con': n_con, 'dynamics': dynamics_function, 'objective': objective_function, 'jacobian_x': jacobian_x_function}
    constraints_dict = {'path_constraints': path_constraints_function, 'event_constraints': event_constraints_function, 'misc_constraints': misc_constraints_function}
    param_dict = {'mu': mu, 'mu_s': mu_s, 'om_sun': om_sun, 'sun_angle_t0': sun_angle_t0, 'a_sun': a_sun, 't0': t0, 'tf': tf, 'x0': x0, 'xf': xf, 'time_vec': time, 'n_man': n_man, 'man_index': man_index, \
        'n_man_defects': n_man_defects, 'man_index_defects': man_index_defects, 'dv_max': dv_max, \
        'ode_atol': ode_atol, 'ode_rtol': ode_rtol, 'ode_atol_piecewise': ode_atol_piecewise, 'ode_rtol_piecewise': ode_rtol_piecewise, 'constant_names':['mu', 'mu_s', 'om_sun', 'sun_angle_t0', 'a_sun']}
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
    
