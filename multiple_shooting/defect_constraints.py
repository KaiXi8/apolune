import numpy as np
import cppad_py
from scipy.integrate import odeint


def dynamics_free_tf(tau, states, controls, p, auxdata):
    dynamics_fun = auxdata['problem']['dynamics']
    
    t0 = auxdata['param']['t0']   
    tf_ind = auxdata['indices']['tf_ind']
    tf = p[tf_ind]
    
    t = tau * (tf - t0) + t0
    dt = tf - t0
    
    return dt * dynamics_fun(t, states, controls, p, auxdata)
    

def dynamics_stm(t, state_stm, controls, p, auxdata):
    
    dynamics_fun = auxdata['problem']['dynamics']
    jacobian_x_fun = auxdata['problem']['jacobian_x']
        
    n_x = auxdata['problem']['n_x']
    x_ind = slice(0, n_x)
    stm_x_ind = auxdata['indices']['stm_x_ind']
    
    x = state_stm[x_ind];
    stm = state_stm[stm_x_ind].reshape((n_x, n_x))
    
    x_dot = dynamics_fun(t, x, controls, p, auxdata)
    jac_x = jacobian_x_fun(t, x, controls, p, auxdata)
    stm_dot_x = jac_x.dot(stm) 

    return np.concatenate((x_dot, stm_dot_x.flatten())) 


def dynamics_stm_free_tf(tau, state_stm, controls, p, auxdata):
    
    dynamics_fun = auxdata['problem']['dynamics']
    jacobian_x_fun = auxdata['problem']['jacobian_x']
        
    n_x = auxdata['problem']['n_x']
    x_ind = slice(0, n_x)
    stm_x_ind = auxdata['indices']['stm_x_ind']
    stm_t_ind = auxdata['indices']['stm_t_ind']
    
    t0 = auxdata['param']['t0']   
    tf_ind = auxdata['indices']['tf_ind']
    tf = p[tf_ind]
    
    x = state_stm[x_ind];
    stm = state_stm[stm_x_ind].reshape((n_x, n_x))
    stm_t = state_stm[stm_t_ind]
    
    t = tau * (tf - t0) + t0
    dt = tf - t0
    
    f = dynamics_fun(t, x, controls, p, auxdata)
    
    x_dot = dt * f
    jac_x = jacobian_x_fun(t, x, controls, p, auxdata)
    stm_dot_x = dt * jac_x.dot(stm) 
    stm_dot_t = f + dt * jac_x.dot(stm_t) 

    return np.concatenate((x_dot, stm_dot_x.flatten(), stm_dot_t.flatten())) 


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
        tmp = odeint(dynamics_fun, x[i], tspan, args=(u, p, aux,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
        state_int_piecewise[i] = tmp[-1]
    
    return state_int_piecewise


def integrate_piecewise_stm(dynamics_fun, discretizationData, tvec, x, u, p, aux):

    n_x = aux['problem']['n_x']
    N = aux['problem']['N']
    Ns = aux['problem']['Ns']
    x_ind = slice(0, n_x)
    stm_x_ind = aux['indices']['stm_x_ind']
    stm_t_ind = aux['indices']['stm_t_ind']
    
    ode_rtol = aux['param']['ode_rtol_piecewise']
    ode_atol = aux['param']['ode_atol_piecewise']
    
    V0 = aux['discretization']['V0']
    discretizationData['x'][0] = x[0]
    
    state_prop_piecewise = np.zeros((Ns, n_x), dtype=cppad_py.a_double)
    
    for i in range(0, Ns):
        tk = tvec[i]
        tk1 = tvec[i+1]
        tspan = np.array([tk, tk1])
        
        V0[x_ind] = x[i]

        V = odeint(dynamics_fun, V0, tspan, args=(u, p, aux,), tfirst=True, rtol=ode_rtol, atol=ode_atol)[-1]
            
        discretizationData['x'][i+1] = V[x_ind]
        discretizationData['stm_x'][i] = V[stm_x_ind]
        discretizationData['stm_t'][i] = V[stm_t_ind]  
    
    return discretizationData



def compute_defects(tvec, x, u, p, auxdata):

    n_x = auxdata['problem']['n_x']
    
    if auxdata['problem']['free_tf'] == 1:
        dynamics_fun = dynamics_free_tf
    else:
        dynamics_fun = auxdata['problem']['dynamics']
        
    state_int_piecewise = integrate_piecewise(dynamics_fun, tvec, x, u, p, auxdata)
    
    n_man = auxdata['param']['n_man']
    man_index_list = auxdata['param']['man_index']

    for i in range(0, n_man):
        man_index = man_index_list[i]
        if man_index > 0:
            con_index = man_index - 1
            state_int_piecewise[con_index,3:6] += u[i] 
            
    defects_matrix = state_int_piecewise - x[1:]
    
    return defects_matrix



def compute_jacobian_defects_data(jac_dict_defects, tvec, x, u, p, auxdata):
           
    if auxdata['problem']['free_tf'] == 1:
        dynamics_fun = dynamics_stm_free_tf
    else:
        dynamics_fun = dynamics_stm        
           
           
    discretizationData = integrate_piecewise_stm(dynamics_fun, auxdata['discretization'], tvec, x, u, p, auxdata)
    
    jac_data = update_jacobian_defects_data(jac_dict_defects, jac_dict_defects['data'], discretizationData['stm_x'], discretizationData['stm_t'])
    
    return jac_data



def compute_jacobian_sparsity_patt_defects(N, n_x, n_u, auxdata):
    """
    Determines the sparsity pattern for the Jacobian of the defect constraints,
    identifying nonzero row and column indices, and providing data indices
    for later updates. Uses pre-allocated numpy arrays for efficiency.
    
    Parameters:
    - N: Number of nodes (or shooting points)
    - n_x: Dimension of each state
    - n_u: Dimension of each control
    
    Returns:
    - row_indices: Pre-allocated numpy array of row indices for nonzero entries
    - col_indices: Pre-allocated numpy array of column indices for nonzero entries
    - phi_indices: Pre-allocated numpy array of indices for phi_x updates
    - minus_I_indices: Pre-allocated numpy array of indices for -I updates
    - psi_uk_indices: Pre-allocated numpy array of indices for psi_uk updates
    - psi_uk1_indices: Pre-allocated numpy array of indices for psi_uk1 updates
    """
    
    n_man_defects = auxdata['param']['n_man_defects']
    man_index_defects = auxdata['param']['man_index_defects']
    man_index = auxdata['param']['man_index']
    x_len = auxdata['lengths']['x_len']

    stm_x_len = auxdata['lengths']['stm_x_len']
    stm_t_len = auxdata['lengths']['stm_t_len']
    
    tf_ind_sol = auxdata['indices']['tf_ind_sol']
    
    # Calculate the number of nonzero entries per interval and overall
    entries_per_interval_state = stm_x_len + n_x + stm_t_len
    total_entries = (N - 1) * entries_per_interval_state + n_u * n_man_defects

    # Initialize arrays for row and column indices of nonzero elements
    data = np.zeros(total_entries, dtype=float)
    row_indices = np.zeros(total_entries, dtype=int)
    col_indices = np.zeros(total_entries, dtype=int)
    
    # Initialize arrays for indices in the data array for each block
    phi_indices = np.zeros((N - 1) * n_x * n_x, dtype=int)
    minus_I_indices = np.zeros((N - 1) * n_x, dtype=int)
    dv_indices = np.zeros(n_u * n_man_defects, dtype=int)
    stm_t_indices = np.zeros((N - 1) * stm_t_len, dtype=int)
    
    data_index = 0
    phi_index = 0
    minus_I_index = 0
    dv_index = 0
    stm_t_index = 0

    for k in range(N - 1):
        # Row start index for this defect constraint block
        row_start = k * n_x
        
        # Column indices for x_k, x_{k+1}, u_k, and u_{k+1} in the full Jacobian
        xk_col_start = k * n_x
        xk1_col_start = (k + 1) * n_x

        # Nonzero entries for phi_x (STM with respect to x_k)
        for i in range(n_x):
            for j in range(n_x):
                row_indices[data_index] = row_start + i
                col_indices[data_index] = xk_col_start + j
                phi_indices[phi_index] = data_index
                data_index += 1
                phi_index += 1

        # Nonzero entries for -I with respect to x_{k+1}
        for i in range(n_x):
            row_indices[data_index] = row_start + i
            col_indices[data_index] = xk1_col_start + i
            minus_I_indices[minus_I_index] = data_index
            data[data_index] = -1
            data_index += 1
            minus_I_index += 1
            
        if stm_t_len > 0:
            # Nonzero entries for tf block with respect to tf
            for i in range(n_x):
                row_indices[data_index] = row_start + i
                col_indices[data_index] = tf_ind_sol
                stm_t_indices[stm_t_index] = data_index
                data_index += 1
                stm_t_index += 1


    dv_col_index = x_len  
    for man_node in man_index:
        if man_node == 0:
            dv_col_index += n_u
            continue
        
        row_start = (man_node - 1) * n_x + 3
        
        for i in range(0, n_u):
            row_indices[data_index] = row_start + i
            col_indices[data_index] = dv_col_index + i
            dv_indices[dv_index] = data_index
            data[data_index] = 1
            data_index += 1
            dv_index += 1
        
        dv_col_index += n_u


    jac_dict_defects = {}
    jac_dict_defects['data'] = data
    jac_dict_defects['nnz_rows'] = row_indices
    jac_dict_defects['nnz_cols'] = col_indices
    jac_dict_defects['stm_x_ind'] = phi_indices
    jac_dict_defects['stm_t_ind'] = stm_t_indices
    jac_dict_defects['dv_indices'] = dv_indices
    jac_dict_defects['stm_identity_ind'] = minus_I_indices
    
    return jac_dict_defects
        


def update_jacobian_defects_data(jac_dict, data, stm_x_flattened, stm_t_flattened):
    """
    Updates the data array with the values from phi_x, psi_uk, and psi_uk1
    at the specified indices.

    Parameters:
    - data: The data array to be updated with STM values (1D numpy array).
    - stm_x_flattened: Flattened state transition matrices for states, shape (N-1, n_x^2).
    - stm_x_data_indices: Indices in the data array for stm_x_flattened entries.
    """
    
    stm_x_data_indices = jac_dict['stm_x_ind']
    stm_t_data_indices = jac_dict['stm_t_ind']
    
    # Update data for stm_x_flattened (STM for states x)
    for k in range(stm_x_flattened.shape[0]):
        data[stm_x_data_indices[k * stm_x_flattened.shape[1] : (k + 1) * stm_x_flattened.shape[1]]] = stm_x_flattened[k]

    if stm_t_flattened.shape[1] > 0:
        # Update data for stm_t_flattened (STM for tf)
        for k in range(stm_t_flattened.shape[0]):
            data[stm_t_data_indices[k * stm_t_flattened.shape[1] : (k + 1) * stm_t_flattened.shape[1]]] = stm_t_flattened[k]

    return data
