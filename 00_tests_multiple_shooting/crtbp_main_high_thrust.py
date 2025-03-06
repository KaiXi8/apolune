import sys
import os

# Construct the full path to the directory containing the package
project_path = '/Users/hofmannc/git/apolune'

# Add the directory to sys.path
sys.path.append(project_path)

import numpy as np
import cppad_py
from scipy.integrate import odeint
import cyipopt
import matplotlib.pyplot as plt
from multiple_shooting.utils import reshapeToNdim, reshapeTo1d
import dynamics.crtbp_dynamics as crtbp
import propagation.propagator as propagation
import multiple_shooting.problem as ms_problem

# import user input data
import crtbp_input_example_1_energy_opt as energy_opt
import crtbp_input_example_2_fuel_opt as fuel_opt


# get user input data
# auxdata = fuel_opt.settings()
auxdata = energy_opt.settings()

N = auxdata['problem']['N']
TU = auxdata['scaling']['TU']
t0 = auxdata['param']['t0']
tf = auxdata['param']['tf']
x0 = auxdata['param']['x0']
xf = auxdata['param']['xf']
time = auxdata['param']['time_vec']

# get initial guess
x_guess = auxdata['initial_guess']['state']
u_guess = auxdata['initial_guess']['control']
p_guess = auxdata['initial_guess']['p']


# build initial guess vector
sol_guess = reshapeTo1d(x_guess, u_guess, p_guess)

# initialize problem instance
prob = ms_problem.high_thrust(auxdata)

# create NLP problem
nlp = cyipopt.Problem(
   n = auxdata['lengths']['sol_len'],
   m = prob.get_num_constraints(),
   problem_obj = prob,
   lb = prob.get_decision_vector_boundaries()[0],
   ub = prob.get_decision_vector_boundaries()[1],
   cl = prob.get_constraints_boundaries()[0],
   cu = prob.get_constraints_boundaries()[1],
)

# set solver settings
nlp.add_option('linear_solver', 'ma57')
nlp.add_option('hsllib', 'libcoinhsl.dylib')
nlp.add_option('print_level', 5)
# nlp.add_option('derivative_test', 'first-order')
nlp.add_option('max_iter', 5000)
nlp.add_option('constr_viol_tol', 1e-7)
nlp.add_option('tol', 1e-7)
nlp.add_option('acceptable_tol', 1e-6)
nlp.add_option('nlp_scaling_method', 'none')

# Solve the problem
solution, info = nlp.solve(sol_guess)

x_sol, u_sol, p_sol = reshapeToNdim(solution, auxdata)



# print solution
x0_sol = x_sol[0]
print("optimized x0: ", x0_sol)
print("controls: ", u_sol)
print("dv's: ", np.linalg.norm(u_sol, axis=1))
print("total dv: ", np.sum(np.linalg.norm(u_sol, axis=1)))
print("p_sol: ", p_sol)
print("time[-1]: ", time[-1])

if auxdata['problem']['free_tf'] == 1:
    tf_sol = p_sol[-1]
    time = time * tf_sol

# propagate initial state using the optimized controls
x_prop = propagation.propagate_high_thrust(x0_sol, u_sol, p_sol, time, auxdata)

# check difference between propagated state and optimized state
dxf = x_sol - x_prop
dxf_max = np.max(np.abs(dxf))
print("max. difference between propagated and optimized final state: ", dxf_max)


plt.figure()
ax = plt.axes(projection ='3d')
ax.plot(x_sol[:,0], x_sol[:,1], x_sol[:,2], label='transfer')
ax.plot(x_guess[:,0], x_guess[:,1], x_guess[:,2], label='guess')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()

fig, ax = plt.subplots()
ax.plot(x_sol[:,0], x_sol[:,1], label='transfer')
ax.plot(x_guess[:,0], x_guess[:,1], label='guess')
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.legend()
plt.show()


# period_x0 = 2.8 # approx.period
# period_xf = 3.4 # approx.period
# time_x0 = np.linspace(t0, period_x0, N)
# time_xf = np.linspace(t0, period_xf, N)
# x0_prop = propagation.propagate(crtbp.eqm_6_synodic, x0, p_sol, time_x0, auxdata)
# xf_prop = propagation.propagate(crtbp.eqm_6_synodic, xf, p_sol, time_xf, auxdata)
# 
# plt.figure()
# ax = plt.axes(projection ='3d')
# ax.plot(x0_prop[:,0], x0_prop[:,1], x0_prop[:,2], label='x0 halo')
# ax.plot(xf_prop[:,0], xf_prop[:,1], xf_prop[:,2], label='xf halo')
# ax.plot(x_sol[:,0], x_sol[:,1], x_sol[:,2], label='transfer')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.legend()
# 
# fig, ax = plt.subplots()
# ax.plot(x0_prop[:,0], x0_prop[:,1], label='x0 halo')
# ax.plot(xf_prop[:,0], xf_prop[:,1], label='xf halo')
# ax.plot(x_sol[:,0], x_sol[:,1], label='transfer')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# plt.legend()
# 
# plt.show()
