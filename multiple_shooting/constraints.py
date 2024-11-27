import numpy as np
import cppad_py
from multiple_shooting.utils import reshapeToNdim
import multiple_shooting.defect_constraints as defect_constraints


def compute_jacobian_mapping_function_constraints(tvec, auxdata):
    sol_len = auxdata['lengths']['sol_len']
    
    ind = np.ones(sol_len, dtype=float)
    ind_ = cppad_py.independent(ind)
    
    dep_ = compute_path_event_misc_constraints(ind_, tvec, auxdata)
    mapping_function = cppad_py.d_fun(ind_, dep_)
    mapping_function.optimize()
    
    return mapping_function



def compute_jacobian_sparsity_patt_constraints(tvec, auxdata):

    n_path_con = auxdata['problem']['n_path_con']
    n_event_con = auxdata['problem']['n_event_con']
    n_misc_con = auxdata['problem']['n_misc_con']
    
    num_constraints = n_path_con + n_event_con + n_misc_con
    
    if num_constraints == 0:
        return np.empty(0)

    mapping_function = compute_jacobian_mapping_function_constraints(tvec, auxdata)
    
    # Jacobian number of columns
    n_col = mapping_function.size_domain()
    
    # Set up of the identity matrix pattern
    pattern_in = cppad_py.sparse_rc()
    pattern_in.resize(n_col, n_col, n_col)
    for i in range(n_col):
        pattern_in.put(i, i, i)
    
    # Sets up of the pattern, just a sparsity
    # pattern that will hold the Jacobian sparsity pattern
    jac_sp_patt = cppad_py.sparse_rc()
    
    # Computation of the jacobian sparsity pattern
    mapping_function.for_jac_sparsity(pattern_in, jac_sp_patt)
    
    # Computation of all possibly non-zero entries in Jacobian
    jac_data = cppad_py.sparse_rcv()
    jac_data.pat(jac_sp_patt)
    
    # Work space used to save time for multiple calls
    work = cppad_py.sparse_jac_work()
    
    jac_dict_con = {}
    jac_dict_con['cppad_data'] = jac_data
    jac_dict_con['nnz_rows'] = jac_data.row()
    jac_dict_con['nnz_cols'] = jac_data.col()
    jac_dict_con['cppad_sparsity_pattern'] = jac_sp_patt
    jac_dict_con['cppad_work'] = work
    jac_dict_con['cppad_mapping_function'] = mapping_function
    
    return jac_dict_con



def compute_jacobian_constraints_data(sol, jac_dict_con, auxdata):

    n_path_con = auxdata['problem']['n_path_con']
    n_event_con = auxdata['problem']['n_event_con']
    n_misc_con = auxdata['problem']['n_misc_con']
    
    num_constraints = n_path_con + n_event_con + n_misc_con
    
    if num_constraints == 0:
        return np.empty(0)

    mapping_function = jac_dict_con['cppad_mapping_function']
    mapping_function.sparse_jac_for(jac_dict_con['cppad_data'], sol, jac_dict_con['cppad_sparsity_pattern'], jac_dict_con['cppad_work'])

    return jac_dict_con['cppad_data'].val()



def compute_jacobian_sparsity_pattern(tvec, auxdata):

    n_x = auxdata['problem']['n_x']
    n_u = auxdata['problem']['n_u']
    N = auxdata['problem']['N']
    Ns = auxdata['problem']['Ns']
    
    jac_dict_defects = defect_constraints.compute_jacobian_sparsity_patt_defects(N, n_x, n_u, auxdata)
    defects_rows = jac_dict_defects['nnz_rows']
    defects_cols = jac_dict_defects['nnz_cols']
    num_rows_defects = n_x*Ns
    
    if auxdata['problem']['n_con'] > 0:
        jac_dict_constraints = compute_jacobian_sparsity_patt_constraints(tvec, auxdata)
        constraints_rows = jac_dict_constraints['nnz_rows']
        constraints_cols = jac_dict_constraints['nnz_cols']
        
        nnz_rows = np.concatenate((defects_rows, constraints_rows + num_rows_defects))
        nnz_cols = np.concatenate((defects_cols, constraints_cols))
    else:
        nnz_rows = defects_rows
        nnz_cols = defects_cols
    
    return nnz_rows, nnz_cols


def compute_jacobian(tvec, sol, jac_dict_defects, jac_dict_con, auxdata):

    x, u, p = reshapeToNdim(sol, auxdata)
    
    jac_data_defects = defect_constraints.compute_jacobian_defects_data(jac_dict_defects, tvec, x, u, p, auxdata) 
    jac_data_constraints = compute_jacobian_constraints_data(sol, jac_dict_con, auxdata)
    
    return np.concatenate((jac_data_defects, jac_data_constraints))
    

def build_decision_variable_vector_boundaries(aux):

    states_lower = aux['boundaries']['states_lower']
    states_upper = aux['boundaries']['states_upper']
    
    N = aux['problem']['N']
    n_man = aux['param']['n_man']
    n_p = aux['problem']['n_p']
    
    # States boundaries initialization
    states_low = np.hstack([states_lower] * N)
    states_upp = np.hstack([states_upper] * N)
    
    # Controls boundaries initialization
    if n_man > 0:
        controls_lower = aux['boundaries']['controls_lower']
        controls_upper = aux['boundaries']['controls_upper']
        controls_low = np.hstack([controls_lower] * n_man)
        controls_upp = np.hstack([controls_upper] * n_man)
    else:
        controls_low = np.empty(0)
        controls_upp = np.empty(0)
    
    if n_p > 0:
        p_low = aux['boundaries']['p_lower']
        p_upp = aux['boundaries']['p_upper']
    else:
        p_low = np.empty(0)
        p_upp = np.empty(0)

    # Concatenation of the states, controls boundaries
    low = np.concatenate((states_low, controls_low, p_low))
    upp = np.concatenate((states_upp, controls_upp, p_upp))

    return low, upp



def build_constraints_boundaries(aux):

    n_x = aux['problem']['n_x']
    N = aux['problem']['N']
    n_path_con = aux['problem']['n_path_con']
    n_event_con = aux['problem']['n_event_con']
    n_misc_con = aux['problem']['n_misc_con']
    
    defects_low = np.zeros(n_x * (N - 1))
    defects_upp = np.zeros(n_x * (N - 1))
    
    if n_path_con != 0:
        path_constraints_lower = aux['boundaries']['path_constraints_lower']
        path_constraints_upper = aux['boundaries']['path_constraints_upper']
        path_low = np.hstack([path_constraints_lower] * N)
        path_upp = np.hstack([path_constraints_upper] * N)
    else:
        path_low = np.empty(0)
        path_upp = np.empty(0)
        
    if n_event_con != 0:
        event_low = aux['boundaries']['event_constraints_lower']
        event_upp = aux['boundaries']['event_constraints_upper']
    else:
        event_low = np.empty(0)
        event_upp = np.empty(0)
     
    if n_misc_con != 0:
        misc_low = aux['boundaries']['misc_constraints_lower']
        misc_upp = aux['boundaries']['misc_constraints_upper']
    else:
        misc_low = np.empty(0)
        misc_upp = np.empty(0)

    low = np.concatenate((defects_low, path_low, event_low, misc_low))
    upp = np.concatenate((defects_upp, path_upp, event_upp, misc_upp))

    return low, upp



def compute_constraints(sol, tvec, aux):

    x, u, p = reshapeToNdim(sol, aux)
    
    defects_matrix = defect_constraints.compute_defects(tvec, x, u, p, aux)
    defects = defects_matrix.flatten()
    
    other_con = compute_path_event_misc_constraints(sol, tvec, aux)
        
    con = np.concatenate((defects, other_con))

    return con



def compute_path_event_misc_constraints(sol, tvec, aux):

    n_path_con = aux['problem']['n_path_con']
    n_event_con = aux['problem']['n_event_con']
    n_misc_con = aux['problem']['n_misc_con']

    x, u, p = reshapeToNdim(sol, aux)
        
    if n_path_con != 0:
        path_constraints = aux['constraints']['path_constraints']
        path_con = path_constraints(x, u, p, aux)
        path = path_con.flatten()
    else:
        path = np.empty(0)
        
    if n_event_con != 0:
        event_constraints = aux['constraints']['event_constraints']
        if aux['problem']['n_u'] == 0:
            event_con = event_constraints(x[0], 0.0, x[-1], 0.0, p, aux)
        else:
            event_con = event_constraints(x[0], u[0], x[-1], u[-1], p, aux)
        event = event_con.flatten()
    else:
        event = np.empty(0)
     
    if n_misc_con != 0:
        misc_constraints = aux['constraints']['misc_constraints']
        misc_con = misc_constraints(x, u, p, aux)
        misc = misc_con.flatten()
    else:
        misc = np.empty(0)

    con = np.concatenate((path, event, misc), dtype=cppad_py.a_double)
    
    return con
    
    