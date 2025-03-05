import numpy as np
from scipy.integrate import odeint
import cvxpy as cvx


def compute_foh(t, uk, uk1, tk, tk1):
    dt = tk1 - tk;
    foh_k = (tk1-t)/dt;
    foh_k1 = (t-tk)/dt;
    
    return uk*foh_k + uk1*foh_k1, foh_k, foh_k1
    
    

def dynamics_foh(t, states, uk, uk1, tk, tk1, p, auxdata):
    dynamics_fun = auxdata['problem']['dynamics']
    
    u, _, _ = compute_foh(t, uk, uk1, tk, tk1)

    dx = dynamics_fun(t, states, u, p, auxdata);
    
    return dx



def dynamics_free_tf(tau, states, controls, p, auxdata):
    dynamics_fun = auxdata['problem']['dynamics']
    
    t0 = auxdata['param']['t0']   
    tf_ind = auxdata['indices']['tf_ind']
    tf = p[tf_ind]
    
    t = tau * (tf - t0) + t0
    dt = tf - t0
    
    return dt * dynamics_fun(t, states, controls, p, auxdata)
    

    
def dynamics_ltv_zoh(t, state_stm, control, p, auxdata):
    
    dynamics_fun = auxdata['problem']['dynamics']
    jacobian_x_fun = auxdata['problem']['jacobian_x']
    jacobian_u_fun = auxdata['problem']['jacobian_u']
        
    n_x = auxdata['problem']['n_x']
    n_u = auxdata['problem']['n_u']
    x_ind = slice(0, n_x)
    stm_x_ind = auxdata['indices']['stm_x_ind']
#     stm_uk_ind = auxdata['indices']['stm_uk_ind']   
    
    x = state_stm[x_ind];
    stm_x = state_stm[stm_x_ind].reshape((n_x, n_x))
#     stm_uk = state_stm[stm_uk_ind].reshape((n_x, n_u))
    
    x_dot = dynamics_fun(t, x, control, p, auxdata)
    x_dot_free = dynamics_fun(t, x, np.zeros(n_u), p, auxdata)
    
    jac_x = jacobian_x_fun(t, x, control, p, auxdata)
    jac_u = jacobian_u_fun(t, x, control, p, auxdata)
    const_part = x_dot_free - jac_x @ x
    
    stm_x_inv = np.linalg.inv(stm_x)
    
    stm_dot_x = jac_x.dot(stm_x) 
    stm_dot_uk = stm_x_inv @ jac_u
    const_part_dot = stm_x_inv @ (x_dot_free - jac_x @ x)
    
    return np.concatenate((x_dot, stm_dot_x.flatten(), stm_dot_uk.flatten(), const_part_dot)) 



def dynamics_ltv_foh(t, state_stm, uk, uk1, tk, tk1, p, auxdata):
    
    dynamics_fun = auxdata['problem']['dynamics']
    jacobian_x_fun = auxdata['problem']['jacobian_x']
    jacobian_u_fun = auxdata['problem']['jacobian_u']
        
    n_x = auxdata['problem']['n_x']
    n_u = auxdata['problem']['n_u']
    x_ind = slice(0, n_x)
    stm_x_ind = auxdata['indices']['stm_x_ind']
#     stm_uk_ind = auxdata['indices']['stm_uk_ind']   

    control, foh_k, foh_k1 = compute_foh(t, uk, uk1, tk, tk1)
    
    x = state_stm[x_ind]
    stm_x = state_stm[stm_x_ind].reshape((n_x, n_x))
#     stm_uk = state_stm[stm_uk_ind].reshape((n_x, n_u))
    
    x_dot = dynamics_fun(t, x, control, p, auxdata)
    x_dot_free = dynamics_fun(t, x, np.zeros(n_u), p, auxdata)
    
    jac_x = jacobian_x_fun(t, x, control, p, auxdata)
    jac_u = jacobian_u_fun(t, x, control, p, auxdata)
    const_part = x_dot_free - jac_x @ x
    
    stm_x_inv = np.linalg.inv(stm_x)
    
    stm_dot_x = jac_x.dot(stm_x) 
    stm_dot_uk = stm_x_inv @ jac_u * foh_k
    stm_dot_uk1 = stm_x_inv @ jac_u * foh_k1
    const_part_dot = stm_x_inv @ const_part
    
    return np.concatenate((x_dot, stm_dot_x.flatten(), stm_dot_uk.flatten(), stm_dot_uk1.flatten(), const_part_dot)) 



def dynamics_ltv_foh_free_tf(tau, state_stm, uk, uk1, tk, tk1, p, auxdata):
    
    dynamics_fun = auxdata['problem']['dynamics']
    jacobian_x_fun = auxdata['problem']['jacobian_x']
    jacobian_u_fun = auxdata['problem']['jacobian_u']
        
    n_x = auxdata['problem']['n_x']
    x_ind = slice(0, n_x)
    stm_x_ind = auxdata['indices']['stm_x_ind']
#     stm_t_ind = auxdata['indices']['stm_t_ind']
#     stm_const_ind = auxdata['indices']['stm_const_ind']
    
    t0 = auxdata['param']['t0']   
    tf_ind = auxdata['indices']['tf_ind']
    tf = p[tf_ind]
    
    t = tau * (tf - t0) + t0
    dt = tf - t0
    
    control, foh_k, foh_k1 = compute_foh(t, uk, uk1, tk, tk1)
    
    x = state_stm[x_ind];
    stm_x = state_stm[stm_x_ind].reshape((n_x, n_x))
    
    x_dot = dynamics_fun(t, x, control, p, auxdata)
    x_dot_free = dynamics_fun(t, x, np.zeros(n_u), p, auxdata)
    
    jac_x = jacobian_x_fun(t, x, control, p, auxdata)
    jac_u = jacobian_u_fun(t, x, control, p, auxdata)
    
    jac_x_dt = dt * jac_x
    jac_u_dt = dt * jac_u
    const_part_dt = -jac_x_dt @ x - jac_u_dt @ control
    
    x_dot_dt = dt * x_dot
    stm_dot_x_dt = jac_x_dt.dot(stm_x) 
    stm_dot_uk_dt = stm_x_inv @ jac_u_dt * foh_k
    stm_dot_uk1_dt = stm_x_inv @ jac_u_dt * foh_k1
    const_part_dot_dt = stm_x_inv @ const_part_dt
    
    return np.concatenate((x_dot_dt, stm_dot_x_dt.flatten(), stm_dot_uk_dt.flatten(), stm_dot_uk1_dt.flatten(), const_part_dot_dt)) 
    


# def dynamics_stm_foh(t, state_stm, uk, uk1, tk, tk1, p, auxdata):
#     
#     dynamics_fun = auxdata['problem']['dynamics']
#     jacobian_x_fun = auxdata['problem']['jacobian_x']
#     jacobian_u_fun = auxdata['problem']['jacobian_u']
#         
#     n_x = auxdata['problem']['n_x']
#     n_u = auxdata['problem']['n_u']
#     x_ind = slice(0, n_x)
#     stm_x_ind = auxdata['indices']['stm_x_ind']
#     stm_uk_ind = auxdata['indices']['stm_uk_ind']   
#     stm_uk1_ind = auxdata['indices']['stm_uk1_ind']
#     
#     u, foh_k, foh_k1 = compute_foh(t, uk, uk1, tk, tk1)
#     
#     x = state_stm[x_ind];
#     stm = state_stm[stm_x_ind].reshape((n_x, n_x))
#     stm_uk = state_stm[stm_uk_ind].reshape((n_x, n_u))
#     stm_uk1 = state_stm[stm_uk1_ind].reshape((n_x, n_u))
#     
#     x_dot = dynamics_fun(t, x, u, p, auxdata)
#     jac_x = jacobian_x_fun(t, x, u, p, auxdata)
#     jac_u = jacobian_u_fun(t, x, u, p, auxdata)
#     stm_dot_x = jac_x.dot(stm) 
#     stm_dot_uk = jac_x.dot(stm_uk) + jac_u*foh_k
#     stm_dot_uk1 = jac_x.dot(stm_uk1) + jac_u*foh_k1
#     
#     return np.concatenate((x_dot, stm_dot_x.flatten(), stm_dot_uk.flatten(), stm_dot_uk1.flatten())) 
# 
# 
# 
# 
# 
# def dynamics_ltv_free_tf(tau, state_stm, controls, p, auxdata):
#     
#     dynamics_fun = auxdata['problem']['dynamics']
#     jacobian_x_fun = auxdata['problem']['jacobian_x']
#         
#     n_x = auxdata['problem']['n_x']
#     x_ind = slice(0, n_x)
#     stm_x_ind = auxdata['indices']['stm_x_ind']
#     stm_t_ind = auxdata['indices']['stm_t_ind']
#     stm_const_ind = auxdata['indices']['stm_const_ind']
#     
#     t0 = auxdata['param']['t0']   
#     tf_ind = auxdata['indices']['tf_ind']
#     tf = p[tf_ind]
#     
#     x = state_stm[x_ind];
#     stm = state_stm[stm_x_ind].reshape((n_x, n_x))
#     stm_t = state_stm[stm_t_ind]
#     const_part = state_ltv[stm_const_ind]
#     
#     t = tau * (tf - t0) + t0
#     dt = tf - t0
#     
#     f = dynamics_fun(t, x, controls, p, auxdata)
#     
#     stm_inv = np.linalg.inv(stm)
#     
#     x_dot = dt * f
#     jac_x = jacobian_x_fun(t, x, controls, p, auxdata)
#     stm_dot_x = dt * jac_x.dot(stm) 
#     stm_dot_t = stm_inv @ f
#     const_part_dot = stm_inv @ (- dt * jac_x @ x)
# 
#     return np.concatenate((x_dot, stm_dot_x.flatten(), stm_dot_t, const_part_dot)) 


def integrate_piecewise(dynamics_fun, tvec, x, u, p, aux):

    n_x = aux['problem']['n_x']
    N = aux['problem']['N']
    Ns = aux['problem']['Ns']
    
    ode_rtol = aux['param']['ode_rtol_piecewise']
    ode_atol = aux['param']['ode_atol_piecewise']
        
    state_int_piecewise = np.zeros((Ns, n_x))
    
    for i in range(0, Ns):
        tk = tvec[i]
        tk1 = tvec[i+1]
        tspan = np.array([tk, tk1])
        
        if auxdata['discretization']['foh_flag'] == 1:
            uk = u[i]
            uk1 = u[i+1]
            tmp = odeint(dynamics_fun, x[i], tspan, args=(uk, uk1, tk, tk1, p, auxdata,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
        else:
            tmp = odeint(dynamics_fun, x[i], tspan, args=(u[i], p, auxdata,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
        
        state_int_piecewise[i] = tmp[-1]
    
    return state_int_piecewise


def integrate_piecewise_ltv(dynamics_fun, discretizationData, tvec, x, u, p, auxdata):

    n_x = auxdata['problem']['n_x']
    N = auxdata['problem']['N']
    Ns = auxdata['problem']['Ns']
    x_ind = slice(0, n_x)
    stm_x_ind = auxdata['indices']['stm_x_ind']
    stm_uk_ind = auxdata['indices']['stm_uk_ind']
    stm_uk1_ind = auxdata['indices']['stm_uk1_ind']
    stm_t_ind = auxdata['indices']['stm_t_ind']
    stm_const_ind = auxdata['indices']['stm_const_ind']
    
    ode_rtol = auxdata['param']['ode_rtol_piecewise']
    ode_atol = auxdata['param']['ode_atol_piecewise']
    
    V0 = auxdata['discretization']['V0']
    discretizationData['x'][0] = x[0]
    
    state_prop_piecewise = np.zeros((Ns, n_x))
    
    for i in range(0, Ns):
        tk = tvec[i]
        tk1 = tvec[i+1]
        tspan = np.array([tk, tk1])
        
        V0[x_ind] = x[i]
        
        if auxdata['discretization']['foh_flag'] == 1:
            uk = u[i]
            uk1 = u[i+1]
            V = odeint(dynamics_fun, V0, tspan, args=(uk, uk1, tk, tk1, p, auxdata,), tfirst=True, rtol=ode_rtol, atol=ode_atol)[-1]
        else:
            V = odeint(dynamics_fun, V0, tspan, args=(u[i], p, auxdata,), tfirst=True, rtol=ode_rtol, atol=ode_atol)[-1]
            
        

        V = odeint(dynamics_fun, V0, tspan, args=(u, p, auxdata,), tfirst=True, rtol=ode_rtol, atol=ode_atol)[-1]
        
        stm = V[stm_x_ind].reshape((n_x, n_x))
        
        discretizationData['x'][i+1] = V[x_ind]
        discretizationData['stm_x'][i] = V[stm_x_ind]
        discretizationData['stm_uk'][i] = (stm @ V[stm_uk_ind].reshape((n_x, n_u))).flatten()
        discretizationData['stm_uk1'][i] = (stm @ V[stm_uk1_ind].reshape((n_x, n_u))).flatten()
        discretizationData['stm_const'][i] = stm @ V[stm_const_ind]
        
        if auxdata['problem']['free_tf'] == 1:
            discretizationData['stm_t'][i] = (stm @ V[stm_t_ind]).flatten()
    
    return discretizationData



def compute_defects(tvec, x, u, p, auxdata):

    n_x = auxdata['problem']['n_x']
    
    # if auxdata['problem']['free_tf'] == 1:
#         dynamics_fun = dynamics_free_tf
#     else:
#         dynamics_fun = auxdata['problem']['dynamics']
        
    if auxdata['discretization']['foh_flag'] == 1:
        dynamics_fun = dynamics_foh
    else:
        dynamics_fun = auxdata['problem']['dynamics']
        
    state_int_piecewise = integrate_piecewise(dynamics_fun, tvec, x, u, p, auxdata)
            
    defects_matrix = state_int_piecewise - x[1:]
    
    return defects_matrix



def calc_nonlinear_cost(tvec, state, control, p, auxdata):
    n_x = auxdata['problem']['n_x']
    n_u = auxdata['problem']['n_u']
    Tmax = auxdata['param']['Tmax']

    defects_matrix = compute_defects(tvec, state, control, p, auxdata)
    
    violation_dynamics = np.linalg.norm(defects_matrix.flatten(), 1)

    x0 = auxdata['param']['x0']

    dx0 = state[0] - x0
    violation_x0 = np.linalg.norm(dx0, 1)
    
    xf = auxdata['param']['xf']
    dxf = state[-1] - xf
    violation_xf = np.linalg.norm(dxf, 1)
    
    u_mag = control[:, n_u - 1]
    z = x[:, n_x - 1]
    umag_violations = np.maximum(0, u_mag - Tmax * np.exp(-z))
    
    
    
    
    
    dv_max = auxdata['param']['dv_max']
    dv_violations = np.maximum(0, np.linalg.norm(control, axis=1) - dv_max)

    # Total violation
    dv_violations_total = np.sum(dv_violations)
    
    total_con_violation = violation_dynamics + violation_x0 + violation_xf + dv_violations_total
    
    con_violation_concat = np.concatenate((defects_matrix.flatten(), dx0, dxf, dv_violations))
    max_con_violation = np.linalg.norm(con_violation_concat, np.inf)
    
    return total_con_violation, max_con_violation



def calc_linear_cost(tvec, state, control, p, defect_constraints, auxdata):
    
    violations_array = np.array(defect_constraints)
    violation_dynamics = np.linalg.norm(violations_array.flatten(), 1)

    x0 = auxdata['param']['x0']

    dx0 = state[0] - x0
    violation_x0 = np.linalg.norm(dx0, 1)
    
    xf = auxdata['param']['xf']
    dxf = state[-1] - xf
    violation_xf = np.linalg.norm(dxf, 1)
    
    dv_max = auxdata['param']['dv_max']
    dv_violations = np.maximum(0, np.linalg.norm(control, axis=1) - dv_max)

    # Total violation
    dv_violations_total = np.sum(dv_violations)
    
    total_con_violation = violation_dynamics + violation_x0 + violation_xf + dv_violations_total
    
    con_violation_concat = np.concatenate((violations_array.flatten(), dx0, dxf, dv_violations))
    max_con_violation = np.linalg.norm(con_violation_concat, np.inf)
    
    return total_con_violation, max_con_violation



def calc_linear_cost_v02(tvec, state, control, p, discretizationData, auxdata):

    n_x = auxdata['problem']['n_x']
    N = auxdata['problem']['N']
    n_man = auxdata['param']['n_man']
    man_index = auxdata['param']['man_index']

    stm_x_array = discretizationData['stm_x']
    stm_t_array = discretizationData['stm_t']
    stm_const_array = discretizationData['stm_const']
    
    state_k1_lin = np.zeros((N-1, n_x))
    
    for k in range(N-1):
        state_k1_lin[k] = stm_x_array[k].reshape((n_x, n_x)) @ state[k] + stm_t_array[k] * p + stm_const_array[k]
        
    for i in range(0, n_man):
        man_index = man_index[i]
        if man_index > 0:
            con_index = man_index - 1
            state_k1_lin[con_index,3:6] += control_cvx[i]   
    
    # defect constraints
    defects_matrix = state_k1_lin - state[1:]

    x0 = auxdata['param']['x0']
    if man_index[0] == 0:
        dx0 = state[0] - np.concat((np.zeros(3), control[0])) - x0
    else:
        dx0 = state[0] - x0
    violation_x0 = np.linalg.norm(dx0, 1)
    
    xf = auxdata['param']['xf']
    dxf = state[-1] - xf
    violation_xf = np.linalg.norm(dxf, 1)
    
    total_con_violation = violation_dynamics + violation_x0 + violation_xf
    
    con_violation_concat = np.concatenate((defects_matrix.flatten(), np.array([violation_x0]), np.array([violation_xf])))
    max_con_violation = np.linalg.norm(con_violation_concat, np.inf)
    
    return total_con_violation, max_con_violation



def check_feasibility(nonlin_max_con_viol, lin_max_con_viol, tol):
    feasible_flag = 0
    
    if (nonlin_max_con_viol <= tol) and (lin_max_con_viol <= tol):
        feasible_flag = 1

    return feasible_flag



def check_optimality(delta_cost, tol):
    optimal_flag = 0
    
    if delta_cost <= tol:
        optimal_flag = 1

    return optimal_flag



def check_convergence(delta_state_norm, feasible_flag, optimal_flag, tol):
    converged_flag = 0
    
    if feasible_flag == 1 and optimal_flag == 1:
        converged_flag = 1
    elif delta_state_norm <= tol:
        converged_flag = 2

    return converged_flag



def update_tr_data(tr_dict):
    
    if tr_dict['rho'] > tr_dict['rho1'] and tr_dict['rho_old'] >= tr_dict['rho0']: # Case 1: Both accepted
    
        tr_dict['beta'] = tr_dict['beta'] * tr_dict['delta']
        if tr_dict['beta'] > tr_dict['beta_max']:
            tr_dict['beta'] = tr_dict['beta_max']
        
        tr_dict['alpha'] = tr_dict['alpha'] / tr_dict['delta']
        if tr_dict['alpha'] < tr_dict['alpha_min']:
            tr_dict['alpha'] = tr_dict['alpha_min']
        
    elif tr_dict['rho'] >= tr_dict['rho0'] and tr_dict['rho_old'] < tr_dict['rho0']:
    
        tr_dict['beta'] = tr_dict['beta'] / tr_dict['delta']
        if tr_dict['beta'] < tr_dict['beta_min']:
            tr_dict['beta'] = tr_dict['beta_min']
        
        tr_dict['alpha'] = tr_dict['alpha'] * tr_dict['delta']
        if tr_dict['alpha'] > tr_dict['alpha_max']:
            tr_dict['alpha'] = tr_dict['alpha_max']

    elif tr_dict['rho'] < tr_dict['rho0'] and tr_dict['rho_old'] < tr_dict['rho0']: # Case 4: Both rejected
        
        tr_dict['alpha'] = tr_dict['alpha'] * tr_dict['delta']
        if tr_dict['alpha'] > tr_dict['alpha_max']:
            tr_dict['alpha'] = tr_dict['alpha_max']

