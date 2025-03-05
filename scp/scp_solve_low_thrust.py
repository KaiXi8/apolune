import numpy as np
from copy import deepcopy
import cvxpy as cvx
import time as tm
import scp.scp_core_low_thrust as scp_core


def solve(guess_dict, scp_data, tr_dict, auxdata):
    model = auxdata['problem']['model']  
    
    N = auxdata['problem']['N']   
    n_x = auxdata['problem']['n_x']   
    n_u = auxdata['problem']['n_u']   
    n_p = auxdata['problem']['n_p']   
    
    x0 = auxdata['param']['x0']   
    xf = auxdata['param']['xf']   
    
    states_lower = auxdata['boundaries']['states_lower']   
    states_upper = auxdata['boundaries']['states_upper']   
    controls_lower = auxdata['boundaries']['controls_lower']   
    controls_upper = auxdata['boundaries']['controls_upper']   
    
    
    verbose_solver = scp_data['verbose_solver'] 
    
    feasibility_tol = scp_data['feasibility_tol'] 
    optimality_tol = scp_data['optimality_tol'] 
    step_tol = scp_data['step_tol'] 
    
    max_iterations = scp_data['max_iterations'] 
    factor_nonlin = scp_data['factor_nonlin'] 
    factor_lin = scp_data['factor_lin'] 
    
    objective_old = scp_data['objective_old'] 
    
    converged_flag = 0
    feasibleflag = 0
    optimalflag = 0
    
    solution_data = {}
    tmp_solution = {}
    
    solution_data['state'] = deepcopy(guess_dict['state'])
    solution_data['control'] = deepcopy(guess_dict['control'])
    solution_data['p'] = deepcopy(guess_dict['p'])
    solution_data['time'] = deepcopy(guess_dict['time'])
    
    tmp_solution['time'] = deepcopy(guess_dict['time'])
    
    if model == 1 or model == 2: # crtbp or bcrfbp
        Tmax = auxdata['param']['Tmax']   
    else:
        Tmax_vec = scp_core.get_Tmax(solution_data['time'], auxdata)
    
    nonlin_con_violation_guess, nonlin_max_con_violation_guess = scp_core.calc_nonlinear_cost(guess_dict['time'], guess_dict['state'], guess_dict['control'], guess_dict['p'], auxdata)
    nonlin_cost_old = factor_nonlin * nonlin_con_violation_guess
    # nonlin_cost_old = 100
    
    
    iterations = 0
    
    # number of constraints
    num_defect_con = (N-1)
    num_initial_bound_con = n_x
    num_final_bound_con = n_x
    num_lowerb_con = N * (n_x + n_u) + n_p
    num_upperb_con = N * (n_x + n_u) + n_p
    num_tr_con = 1
    
#     num_con = num_defect_con + num_initial_bound_con + num_final_bound_con + num_lowerb_con + num_upperb_con + num_tr_con
#     ind_tr_con = num_con - num_tr_con - 1
    
    start_time = tm.perf_counter()
    while (iterations < max_iterations) and converged_flag <= 0:
        
        state_cvx = cvx.Variable((N, n_x))  
        control_cvx = cvx.Variable((N, n_u)) 
        p_cvx = cvx.Variable(1)  
        virtual_control_dynamics = cvx.Variable((N-1, n_x))  
        virtual_control_ineq = cvx.Variable(N, nonneg=True) 
    #     state_k1_lin = cvx.Variable((N-1, n_x)) 
    #     tr_radius_cvx = cvx.Parameter(nonneg=True)
    #     tr_radius_cvx.value = tr_dict['radius']
        
        constraints = []
        
#         if auxdata['problem']['free_tf'] == 1:
        if auxdata['discretization']['foh_flag'] == 1:
#             dynamics_ltv_fun = scp_core.dynamics_ltv_free_tf
            dynamics_ltv_fun = scp_core.dynamics_ltv_foh
        else:
            dynamics_ltv_fun = scp_core.dynamics_ltv_zoh        
                   
        discretizationData = scp_core.integrate_piecewise_ltv(dynamics_ltv_fun, auxdata['discretization'], solution_data['time'], solution_data['state'], solution_data['control'], solution_data['p'], auxdata)
        stm_x_array = discretizationData['stm_x']
        stm_uk_array = discretizationData['stm_uk']
        stm_uk1_array = discretizationData['stm_uk1']
        stm_t_array = discretizationData['stm_t']
        stm_const_array = discretizationData['stm_const']
        
        state_k1_lin = []
        for k in range(N-1):
    #         state_k1_lin[k] = stm_x_array[k].reshape((n_x, n_x)) @ state_cvx[k] + stm_t_array[k] * p_cvx + stm_const_array[k]
            state_k1_lin.append(
                stm_x_array[k].reshape((n_x, n_x)) @ state_cvx[k] + stm_uk_array[k].reshape((n_x, n_u)) @ control_cvx[k] 
                + stm_uk1_array[k].reshape((n_x, n_u)) @ control_cvx[k+1] + stm_const_array[k]
                )
        
        # defect constraints
        # state_k1_lin - state_cvx[1:] == 0  
        for k in range(N-1):
            defect_con = state_k1_lin[k] - state_cvx[k+1] + virtual_control_dynamics[k] == 0
            constraints.append(defect_con)
        
        # initial boundary conditions x[0] == x0
        x0_con = state_cvx[0] == x0
        constraints.append(x0_con)
        
        # final boundary conditions x[N-1] == xf
        xf_con = state_cvx[-1, :6] == xf
        constraints.append(xf_con)
    
        # bounds
        constraints.append(states_lower[np.newaxis, :] <= state_cvx)
        constraints.append(state_cvx <= states_upper[np.newaxis, :])
        constraints.append(controls_lower[np.newaxis, :] <= control_cvx)
        constraints.append(control_cvx <= controls_upper[np.newaxis, :])
    
        # trust region constraint
        constraints.append(cvx.norm(state_cvx - solution_data['state'], 1) <= tr_dict['radius'])
        
        # second-order cone constraint on controls
        for k in range(N):
            constraints.append(cvx.norm(control_cvx[k,:3], 2) <= control_cvx[k,3])
        
        # linearized thrust magnitude constraints
        # rnbp_rpf with nonuniform time: store LUs and VUs for every point in time (assuming a fixed tf)
        for k in range(N):
            z_k = state_cvx[k,-1]
            z_old_k = solution_data['state'][k,-1]
            if model > 2: # rnbp_rpf
                Tmax = Tmax_vec[k]
            constraints.append( control_cvx[k,3] - Tmax * np.exp(-z_old_k) * (1 - z_k + z_old_k) - virtual_control_ineq[k] <= 0 )
        
        # compute objective function
        obj = - state_cvx[-1,-1]
        
        objective = cvx.Minimize(obj + factor_lin * cvx.norm(virtual_control_dynamics, 1) + factor_lin * cvx.sum(cvx.pos(virtual_control_ineq)))
    
    
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.CLARABEL, verbose=verbose_solver)
    
        iterations += 1
        
        tmp_solution['state'] = deepcopy(state_cvx.value)
        tmp_solution['control'] = deepcopy(control_cvx.value)
        tmp_solution['p'] = deepcopy(p_cvx.value)
    
        # compute nonlinear and linear constraint violations
        nonlin_con_violation, nonlin_max_con_violation = scp_core.calc_nonlinear_cost(solution_data['time'], state_cvx.value, control_cvx.value, p_cvx.value, auxdata)
        lin_con_violation, lin_max_con_violation = scp_core.calc_linear_cost(solution_data['time'], state_cvx.value, control_cvx.value, p_cvx.value, virtual_control_dynamics.value, virtual_control_ineq.value, auxdata)
    
        nonlin_cost = factor_nonlin * nonlin_con_violation
        lin_cost = factor_lin * lin_con_violation
    
        predicted_decrease = nonlin_cost_old - lin_cost
        actual_decrease = nonlin_cost_old - nonlin_cost
#         print(f'obj_old: {objective_old}')
#         print(f'obj_val: {objective.value}')
        delta_cost = np.abs((objective_old - objective.value) / objective_old)
        
        delta_state_norm = np.linalg.norm(tmp_solution['state'].flatten() - solution_data['state'].flatten()) / np.linalg.norm(solution_data['state'].flatten())
        
        # check feasibility, optimality, and if converged
        feasible_flag = scp_core.check_feasibility(nonlin_max_con_violation, lin_max_con_violation, feasibility_tol)
        # print(f'homot_param: {auxdata["homot_param"]}')
        optimal_flag = scp_core.check_optimality(delta_cost, optimality_tol)
        converged_flag = scp_core.check_convergence(delta_state_norm, feasible_flag, optimal_flag, step_tol)
        
        if predicted_decrease == 0: # converged if 0; also to avoid division by 0
            converged_flag = 1
        
        if tr_dict['radius'] <= tr_dict['radius_min']:
            converged_flag = 2
        
        if converged_flag > 0: # converged
            solution_data = deepcopy(tmp_solution)
            print("Finished with converged_flag = ", converged_flag)
            break 
        else: # not converged
            tr_dict['rho'] = actual_decrease / predicted_decrease
            
            if iterations > 1:
    #             scp_core.update_tr_data(tr_dict)
                test = 1
            
            tr_dict['rho_old'] = deepcopy(tr_dict['rho'])
            
            if tr_dict['rho'] < tr_dict['rho0']: # reject solution
                print('Solution rejected.')
                tr_dict['radius'] = tr_dict['radius'] / tr_dict['alpha']
                print('Reducing trust region size. Solving again with radius = ', tr_dict['radius'])
            else: # accept solution
                print('Solution accepted.')
                solution_data = deepcopy(tmp_solution)
                nonlin_cost_old = deepcopy(nonlin_cost)
                objective_old = deepcopy(objective.value)
                if tr_dict['rho'] < tr_dict['rho1']: # decrease trust region size
                    tr_dict['radius'] = tr_dict['radius'] / tr_dict['alpha']
                    print('Decreasing radius to = ', tr_dict['radius'])
                elif tr_dict['rho'] >= tr_dict['rho2']: # increase trust region size
                    tr_dict['radius'] = tr_dict['radius'] * tr_dict['beta']
                    if tr_dict['radius'] > tr_dict['radius_max']:
                        tr_dict['radius'] = deepcopy(tr_dict['radius_max'])
                        print('Trust region radius reached maximum value; not increasing.')
                    else:
                        print('Increasing radius to = ', tr_dict['radius'])
    
        
        header = '{:<9}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}'
        print(header.format('iter', 'max_viol', 'dJ', 'c_lin', 'c_nlin', 
            'lin_cost', 'nonlin_cost', 'J', 'dX', 'TR'))
        
        row_format = '{:<9d}  {:>13g}  {:>13g}  {:>13g}  {:>13g}  {:>13g}  {:>13g}  {:>13g}  {:>13g}  {:>13g}'
        print(row_format.format(iterations, nonlin_max_con_violation, delta_cost, lin_max_con_violation, 
            lin_max_con_violation, lin_cost, nonlin_cost, objective.value, delta_state_norm, tr_dict['radius']))   
            
    end_time = tm.perf_counter()
    print("total cpu time: ", end_time - start_time)
    print("cpu time per iteration: ", (end_time - start_time) / iterations)
    
    solution_data['converged_flag'] = converged_flag
    solution_data['objective_new'] = objective.value
    return solution_data

