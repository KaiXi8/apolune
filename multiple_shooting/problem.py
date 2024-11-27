import numpy as np
import cyipopt
import multiple_shooting.objective as obj
import multiple_shooting.defect_constraints as defect_constraints
import multiple_shooting.constraints as constraints

class high_thrust(cyipopt.Problem):

    def __init__(self, aux):
    
        self.auxdata = aux
        self.time_vec = aux['param']['time_vec']
        self.N = aux['problem']['N']
        self.Ns = aux['problem']['Ns']
        self.n_x = aux['problem']['n_x']
        self.n_u = aux['problem']['n_u']
        
        self.objective_mapping_function = obj.compute_objective_mapping_function(self.auxdata)
        self.jac_dict_defects = defect_constraints.compute_jacobian_sparsity_patt_defects(self.N, self.n_x, self.n_u, self.auxdata)
        self.jac_dict_con = constraints.compute_jacobian_sparsity_patt_constraints(self.time_vec, self.auxdata)
        self.nnz_rows, self.nnz_cols = constraints.compute_jacobian_sparsity_pattern(self.time_vec, self.auxdata)
      

    def objective(self, x):
        objective_cost = obj.compute_objective(x, self.auxdata)
        return objective_cost
    
    def gradient(self, x):
        grad_obj = obj.compute_objective_gradient(x, self.objective_mapping_function)
        return grad_obj
    
    def constraints(self, x):
        con = constraints.compute_constraints(x, self.time_vec, self.auxdata)
        return con
    
    def jacobian(self, x):
        jac_data = constraints.compute_jacobian(self.time_vec, x, self.jac_dict_defects, self.jac_dict_con, self.auxdata)
        return jac_data
    
    def jacobianstructure(self):    
        return self.nnz_rows, self.nnz_cols

    def get_decision_vector_boundaries(self):
        lb, ub = constraints.build_decision_variable_vector_boundaries(self.auxdata)
        return lb, ub
    
    def get_constraints_boundaries(self):
        cl, cu = constraints.build_constraints_boundaries(self.auxdata)
        return cl, cu

    def get_num_constraints(self):
        cl, _ = constraints.build_constraints_boundaries(self.auxdata)
        return len(cl)



class differential_correction(cyipopt.Problem):

    def __init__(self, aux):
    
        self.auxdata = aux
        self.time_vec = aux['param']['time_vec']
        self.N = aux['problem']['N']
        self.Ns = aux['problem']['Ns']
        self.n_x = aux['problem']['n_x']
        self.n_u = aux['problem']['n_u']
        
        self.jac_dict_defects = defect_constraints.compute_jacobian_sparsity_patt_defects(self.N, self.n_x, self.n_u, self.auxdata)
        self.jac_dict_con = constraints.compute_jacobian_sparsity_patt_constraints(self.time_vec, self.auxdata)
        self.nnz_rows, self.nnz_cols = constraints.compute_jacobian_sparsity_pattern(self.time_vec, self.auxdata)
      

    def objective(self, x):
        return 0.0
    
    def gradient(self, x):
        return np.zeros(len(x))
    
    def constraints(self, x):
        con = constraints.compute_constraints(x, self.time_vec, self.auxdata)
        return con
    
    def jacobian(self, x):
        jac_data = constraints.compute_jacobian(self.time_vec, x, self.jac_dict_defects, self.jac_dict_con, self.auxdata)
        return jac_data
    
    def jacobianstructure(self):    
        return self.nnz_rows, self.nnz_cols

    def get_decision_vector_boundaries(self):
        lb, ub = constraints.build_decision_variable_vector_boundaries(self.auxdata)
        return lb, ub
    
    def get_constraints_boundaries(self):
        cl, cu = constraints.build_constraints_boundaries(self.auxdata)
        return cl, cu

    def get_num_constraints(self):
        cl, _ = constraints.build_constraints_boundaries(self.auxdata)
        return len(cl)