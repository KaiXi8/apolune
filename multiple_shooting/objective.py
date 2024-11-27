import numpy as np
import cppad_py
from multiple_shooting.utils import reshapeToNdim

def compute_objective(sol, aux):
    
    objective_cost = 0.0
    
    x, u, p = reshapeToNdim(sol, aux)

    objective_fun = aux['problem']['objective']
    objective_cost = objective_fun(x, u, p, aux)
        
    return objective_cost


def compute_objective_mapping_function(auxdata):
        sol_len = auxdata['lengths']['sol_len']
        
        ind = np.zeros(sol_len, dtype=float)
        ind_ = cppad_py.independent(ind)

#         dep_ = np.array([compute_objective(ind_, auxdata)])

        # seems to work with energy optimal
#         dep_ = np.array(compute_objective(ind_, auxdata))
        
        
        dep_ = np.array([compute_objective(ind_, auxdata)])
        
#         print(dep_.shape)
#         tmp = compute_objective(ind, auxdata)
#         tmp_arr = np.array([compute_objective(ind, auxdata)])
#         print(tmp.shape)
#         print(tmp_arr.shape)
        
        mapping_function = cppad_py.d_fun(ind_, dep_)

        return mapping_function


def compute_objective_gradient(sol, mapping_function):
        grad = mapping_function.jacobian(sol)[0]

        return grad

