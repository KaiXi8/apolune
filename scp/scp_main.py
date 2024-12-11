import sys
import os

# Construct the full path to the directory containing the package
project_path = '/Users/hofmannc/git/apolune'

# Add the directory to sys.path
sys.path.append(project_path)


import numpy as np
from copy import deepcopy
from scipy.integrate import odeint
import dynamics.crtbp_dynamics as crtbp
import dynamics_coeff.dynamics_general as dyn_coeff
import propagation.propagator as propagation
import cvxpy as cvx
import matplotlib.pyplot as plt
import spiceypy as spice
import init.load_kernels as krn
import dynamics_coeff.rnbp_rpf_utils as rnbp_utils
import time as tm
import scp_core

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
    x_guess = propagation.propagate(crtbp.dynamics, x0, u_guess, p_guess, time, auxdata)

    return x_guess, u_guess, p_guess



mu_earth_moon = 1.21506683e-2
LU = 3.84405000e5 # km
TU = 4.34811305 # days
VU = 1.02323281 # km/s

# crtbp
mu = mu_earth_moon

# bcrfbp
mu_sun = 3.28900541e5 # adim
a_sun = 3.88811143e2 # adim
om_sun = -9.25195985e-1
sun_angle_t0 = 0.0

# rnbp_rpf
krn.load_kernels()

G = 6.67408e-20  # [km^3 kg^−1 s^−2]
# AU = 149597870.693 # km
AU = 1.495978706136889e+08 # [km] from SPICE

observer_id = 0

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

id_bodies = np.array([id_mer[0], id_ven[0], id_ear[0], id_mar[0], id_jup[0], id_sat[0], id_ura[0], id_nep[0], id_plu[0], id_moo[0], id_sun[0]])
naif_id_bodies = np.array([id_mer[1], id_ven[1], id_ear[1], id_mar[1], id_jup[1], id_sat[1], id_ura[1], id_nep[1], id_plu[1], id_moo[1], id_sun[1]])
mu_bodies = np.array([GM_mer, GM_ven, GM_ear, GM_mar, GM_jup, GM_sat, GM_ura, GM_nep, GM_plu, GM_moo, GM_sun])

MU = mu_bodies[id_primary] + mu_bodies[id_secondary]

epoch_t0 = spice.str2et('23 September 2022 00:00:00 TDB')
reference_frame = "j2000"
reference_frame_encoded = rnbp_utils.frame_encoder("J2000")
mu_p = mu_bodies[id_primary]
mu_s = mu_bodies[id_secondary]
naif_id_p = naif_id_bodies[id_primary]
naif_id_s = naif_id_bodies[id_secondary]

ode_rtol = 1e-12
ode_atol = 1e-12
tau_0 = 0.0
tau_f = 100.0
t0 = 0.0
n_points = 5000
tau_vec_input = np.linspace(tau_0, tau_f, n_points)
tau_vec, t_vec = rnbp_utils.compute_time(tau_vec_input, t0, epoch_t0, mu_p, mu_s, naif_id_p, naif_id_s, reference_frame, ode_rtol = ode_rtol, ode_atol = ode_atol)


n_x = 6
n_u = 3
N = 100
Ns = N - 1

# 0: fixed final time, 1: free final time
# NOTE: i only implemented the case for free_tf = 0 so far
free_tf = 0

# model = 1 # crtbp
# model = 2 # bcrfbp
model = 3 # rnbp_rpf


# node indices where maneuvers are applied; numpy array within [0, Ns]
man_index = np.array([0, 30, 60, Ns])

# initial and final boundary conditions
x0 = np.array([8.2338046140454002e-01, 0, 1.3886061447073000e-02, 0, 1.2947638542136800e-01, 0]) # adim
xf = np.array([1.1194488032482977e+00, 2.3502976908423845e-13, -1.1371675247773910e-02, 2.2969820490104098e-13, 1.7876223257953414e-01, -2.7620527393024119e-13]) # adim
t0 = 0.0
tf = 6.9083604301186052e-01

# bounds for states, controls, and dv per maneuver
states_lower = -10.0 * np.ones(n_x)
states_upper = 10.0 * np.ones(n_x)
controls_lower = -10.0 * np.ones(n_u)
controls_upper = 10.0 * np.ones(n_u)
dv_max = 0.5

# n_p: number of free parameters in the optimization
# here: no free parameters
n_man = len(man_index)
n_p = 0

tf_ind = 0

# lower and upper bounds of the free variable vector p
p_lower = np.empty(0)
p_upper = np.empty(0)

# functions for dynamics and corresponding jacobian
jacobian_x_function = dyn_coeff.jacobian
dynamics_function = dyn_coeff.dynamics

# tolerances for the integrator
# we use different values for piecewise integration (from t_k to t_k+1) and full integration (from t0 to tf)
ode_atol_piecewise = 1e-10
ode_rtol_piecewise = 1e-10
ode_atol = 1e-12
ode_rtol = 1e-12

# discretized time vector
time = np.linspace(t0, tf, N)

# trust region parameters
tr_dict = {}
tr_dict['radius'] = 100.0
tr_dict['rho'] = 100.0
tr_dict['rho0'] = 0.01
tr_dict['rho1'] = 0.2
tr_dict['rho2'] = 0.85
tr_dict['alpha'] = 1.5
tr_dict['beta'] = 1.5
tr_dict['delta'] = 1.0
tr_dict['alpha_min'] = 1.01
tr_dict['alpha_max'] = 4.0
tr_dict['beta_min'] = 1.01
tr_dict['beta_max'] = 4.0
tr_dict['radius_min'] = 1e-7
tr_dict['radius_max'] = 500.0


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
stm_const_len = n_x

V_len = n_x + stm_x_len + stm_t_len + stm_const_len
    
x_ind_stm = slice(0, n_x)
stm_x_ind = slice(n_x, stm_x_len+n_x)
stm_t_ind = slice(stm_x_len+n_x, stm_x_len+n_x+stm_t_len)
stm_const_ind = slice(stm_x_len+n_x+stm_t_len, stm_x_len+n_x+stm_t_len+stm_const_len)

V0 = np.zeros(V_len);
V0[stm_x_ind] = np.identity(n_x).flatten()
 
x_stm = np.zeros((N,n_x));
stm_x = np.zeros((Ns,n_x*n_x));
stm_t = np.zeros((Ns,stm_t_len));
stm_const = np.zeros((Ns,stm_const_len));

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


scaling_dict = {'LU': LU, 'VU': VU, 'TU': TU}
lengths_dict = {'x_len': x_len, 'u_len': u_len, 'xu_len': xu_len, 'p_len': p_len, 't_len': t_len, 'sol_len': sol_len, 'stm_x_len': stm_x_len, 'stm_t_len': stm_t_len, 'stm_const_len': stm_const_len, 'V_len': V_len}
indices_dict = {'x_ind': x_ind, 'u_ind': u_ind, 'p_ind': p_ind, 'tf_ind': tf_ind, 'tf_ind_sol': tf_ind_sol, 'x_ind_stm': x_ind_stm, 'stm_x_ind': stm_x_ind, 'stm_t_ind': stm_t_ind, 'stm_const_ind': stm_const_ind}
problem_dict = {'n_x': n_x, 'n_u': n_u, 'n_p': n_p, 'N': N, 'Ns': Ns, 'free_tf': free_tf, \
    'dynamics': dynamics_function, 'jacobian_x': jacobian_x_function}
param_dict = {'mu': mu, 't0': t0, 'tf': tf, 'x0': x0, 'xf': xf, 'time_vec': time, 'n_man': n_man, 'man_index': man_index, \
    'n_man_defects': n_man_defects, 'man_index_defects': man_index_defects, 'dv_max': dv_max, \
    'ode_atol': ode_atol, 'ode_rtol': ode_rtol, 'ode_atol_piecewise': ode_atol_piecewise, 'ode_rtol_piecewise': ode_rtol_piecewise}
boundaries_dict = {'states_lower': states_lower, 'states_upper': states_upper, 'controls_lower': controls_lower, 'controls_upper': controls_upper, \
    'p_lower': p_lower, 'p_upper': p_upper}
discretization_dict = {'V0': V0, 'x': x_stm, 'stm_x': stm_x, 'stm_t': stm_t, 'stm_const': stm_const}

auxdata = {'problem': problem_dict, 'lengths': lengths_dict, 'indices': indices_dict, 'param': param_dict, 'discretization': discretization_dict, 'boundaries': boundaries_dict, \
    'scaling': scaling_dict}

# generate initial guess 
state_guess, control_guess, p_guess = initial_guess(auxdata)

# for bcrfbp
auxdata["model"] = model
auxdata["mu_crtbp"] = mu
auxdata["mu_sun"] = mu_sun
auxdata["om_sun"] = om_sun
auxdata["sun_angle_t0"] = sun_angle_t0
auxdata["a_sun"] = a_sun

# for rnbp_rpf
auxdata["id_primary"] = id_primary
auxdata["id_secondary"] = id_secondary
auxdata["mu_bodies"] = mu_bodies
auxdata["naif_id_bodies"] = naif_id_bodies
auxdata["observer_id"] = observer_id
auxdata["reference_frame"] = reference_frame # needed if no numba / jit is used
# auxdata["reference_frame"] = reference_frame_encoded # needed for numba and jit
auxdata["epoch_t0"] = epoch_t0
auxdata["tau_vec"] = tau_vec
auxdata["t_vec"] = t_vec


verbose_solver = False

feasibility_tol = 1e-7
optimality_tol = 1e-5
step_tol = 1e-8

max_iterations = 100
factor_nonlin = 10.0
factor_lin = 10.0

objective_old = 1e-3

converged_flag = 0
feasibleflag = 0
optimalflag = 0

solution_data = {}
tmp_solution = {}

solution_data['state'] = deepcopy(state_guess)
solution_data['control'] = deepcopy(control_guess)
solution_data['p'] = deepcopy(p_guess)
solution_data['time'] = deepcopy(time)

tmp_solution['time'] = deepcopy(time)

nonlin_con_violation_guess, nonlin_max_con_violation_guess = scp_core.calc_nonlinear_cost(time, state_guess, control_guess, p_guess, auxdata)
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

num_con = num_defect_con + num_initial_bound_con + num_final_bound_con + num_lowerb_con + num_upperb_con + num_tr_con
ind_tr_con = num_con - num_tr_con - 1

start_time = tm.perf_counter()
while (iterations < max_iterations) and converged_flag <= 0:
    
    state_cvx = cvx.Variable((N, n_x))  
    control_cvx = cvx.Variable((n_man, n_u)) 
    p_cvx = cvx.Variable(1)  
    virtual_control_cvx = cvx.Variable((N-1, n_x))  
#     state_k1_lin = cvx.Variable((N-1, n_x)) 
#     tr_radius_cvx = cvx.Parameter(nonneg=True)
#     tr_radius_cvx.value = tr_dict['radius']
    
    constraints = []
    
    if auxdata['problem']['free_tf'] == 1:
        dynamics_ltv_fun = scp_core.dynamics_ltv_free_tf
    else:
        dynamics_ltv_fun = scp_core.dynamics_ltv        
               
    discretizationData = scp_core.integrate_piecewise_ltv(dynamics_ltv_fun, auxdata['discretization'], solution_data['time'], solution_data['state'], solution_data['control'], solution_data['p'], auxdata)
    stm_x_array = discretizationData['stm_x']
    stm_t_array = discretizationData['stm_t']
    stm_const_array = discretizationData['stm_const']
    
    state_k1_lin = []
    for k in range(N-1):
#         state_k1_lin[k] = stm_x_array[k].reshape((n_x, n_x)) @ state_cvx[k] + stm_t_array[k] * p_cvx + stm_const_array[k]
        state_k1_lin.append(stm_x_array[k].reshape((n_x, n_x)) @ state_cvx[k] + stm_const_array[k])
        
    for i in range(0, n_man):
        man_ind = man_index[i]
        if man_ind > 0:
            con_index = man_ind - 1
            dv = cvx.hstack([0.0, 0.0, 0.0, control_cvx[i]])
            state_k1_lin[con_index] = stm_x_array[con_index].reshape((n_x, n_x)) @ state_cvx[con_index] + stm_const_array[con_index] + dv
    
    # defect constraints
    # state_k1_lin - state_cvx[1:] == 0  
    for k in range(N-1):
        defect_con = state_k1_lin[k] - state_cvx[k+1] + virtual_control_cvx[k] == 0
        constraints.append(defect_con)
    
    # initial boundary conditions x[0] == x0
    if man_index[0] == 0:
        x0_con = state_cvx[0] - cvx.hstack([0.0, 0.0, 0.0, control_cvx[0]]) == x0
    else:
        x0_con = state_cvx[0] == x0
    
    constraints.append(x0_con)
    
    # final boundary conditions x[N-1] == xf
    xf_con = state_cvx[-1] == xf
    constraints.append(xf_con)

    # bounds
    constraints.append(states_lower[np.newaxis, :] <= state_cvx)
    constraints.append(state_cvx <= states_upper[np.newaxis, :])
    constraints.append(controls_lower[np.newaxis, :] <= control_cvx)
    constraints.append(control_cvx <= controls_upper[np.newaxis, :])

    # trust region constraint
    constraints.append(cvx.norm(state_cvx - solution_data['state'], 1) <= tr_dict['radius'])
    
    # constraint on maximum control magnitude per maneuver
    for k in range(n_man):
        constraints.append(cvx.norm(control_cvx[k], 2) <= dv_max)
    
    # compute objective function
    obj = 0.0
    for k in range(n_man):
        obj += cvx.sum_squares(control_cvx[k])
    
    objective = cvx.Minimize(obj + factor_lin * cvx.norm(virtual_control_cvx, 1))


    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.CLARABEL, verbose=verbose_solver)

    iterations += 1
    
    tmp_solution['state'] = deepcopy(state_cvx.value)
    tmp_solution['control'] = deepcopy(control_cvx.value)
    tmp_solution['p'] = deepcopy(p_cvx.value)

    # compute nonlinear and linear constraint violations
    nonlin_con_violation, nonlin_max_con_violation = scp_core.calc_nonlinear_cost(solution_data['time'], state_cvx.value, control_cvx.value, p_cvx.value, auxdata)
    lin_con_violation, lin_max_con_violation = scp_core.calc_linear_cost(solution_data['time'], state_cvx.value, control_cvx.value, p_cvx.value, virtual_control_cvx.value, auxdata)

    nonlin_cost = factor_nonlin * nonlin_con_violation
    lin_cost = factor_lin * lin_con_violation

    predicted_decrease = nonlin_cost_old - lin_cost
    actual_decrease = nonlin_cost_old - nonlin_cost
    delta_cost = np.abs((objective_old - objective.value) / objective_old)
    
    delta_state_norm = np.linalg.norm(tmp_solution['state'].flatten() - solution_data['state'].flatten()) / np.linalg.norm(solution_data['state'].flatten())
    
    # check feasibility, optimality, and if converged
    feasible_flag = scp_core.check_feasibility(nonlin_max_con_violation, lin_max_con_violation, feasibility_tol)
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


state_array = np.array(state_cvx.value)
control_array = np.array(control_cvx.value)
dv_array = np.linalg.norm(control_array, axis=1)
print("control_cvx.value: ", control_array)
print("dv's: ", dv_array)
print("total dv: ", np.sum(dv_array))


plt.figure()
ax = plt.axes(projection ='3d')
ax.plot(state_array[:,0], state_array[:,1], state_array[:,2], label='transfer')
ax.plot(state_guess[:,0], state_guess[:,1], state_guess[:,2], label='guess')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()

fig, ax = plt.subplots()
ax.plot(state_array[:,0], state_array[:,1], label='transfer')
ax.plot(state_guess[:,0], state_guess[:,1], label='guess')
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.legend()
plt.show()

    
