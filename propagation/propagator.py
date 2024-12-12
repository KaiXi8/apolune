import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from propagation.ode_event import plane_crossing_event
import numba

def propagate_high_thrust(x0_cart, controls, p_vec, time, auxdata):
    N = auxdata['problem']['N']
    Ns = auxdata['problem']['Ns']
    n_x = auxdata['problem']['n_x']
    dynamics_fun = auxdata['problem']['dynamics']
    man_indices = auxdata['param']['man_index']
    
    ode_rtol = auxdata['param']['ode_rtol_piecewise']
    ode_atol = auxdata['param']['ode_atol_piecewise']

    use_numba = dynamics_fun.__name__.endswith("_numba")
    if use_numba:
        params = [auxdata['param'][name] for name in auxdata['param']['constant_names']]
    state_prop = np.zeros((N, n_x))
    state_prop[0] = x0_cart
    for i in range(0, Ns):
        tspan = np.array([time[i], time[i+1]])
        if use_numba:
            tmp = odeint(dynamics_fun, state_prop[i], tspan, args=(controls, p_vec, *params), tfirst=True, rtol=ode_rtol, atol=ode_atol)
        else:
            tmp = odeint(dynamics_fun, state_prop[i], tspan, args=(controls, p_vec, auxdata), tfirst=True, rtol=ode_rtol, atol=ode_atol)
        state_prop[i+1] = tmp[-1]
        
        if i+1 in man_indices:
            man_ind = np.where(man_indices == i+1)[0][0]
            state_prop[i+1,3:6] += controls[man_ind] 
    
    return state_prop
    

def propagate(ode_fun, state0, control, p, tvec, auxdata):

    ode_rtol = auxdata['param']['ode_rtol']
    ode_atol = auxdata['param']['ode_atol']
    use_numba = ode_fun.__name__.endswith("_numba")
    if use_numba:
        params = [auxdata['param'][name] for name in auxdata['param']['constant_names']]
    
    if len(tvec) > 2:
        if not use_numba:
            sol = odeint(ode_fun, state0, tvec, args=(control, p, auxdata,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
        else:
            sol = odeint(ode_fun, state0, tvec, args=(control, p, *params,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
    else:
        tspan = np.array([tvec[0], tvec[-1]])
        if not use_numba:
            sol = solve_ivp(ode_fun, tspan, state0, args=(control, p, auxdata,), rtol=ode_rtol, atol=ode_atol)
        else:
            sol = solve_ivp(ode_fun, tspan, state0, args=(control, p, *params,), rtol=ode_rtol, atol=ode_atol)
    
    
    return sol

def propagate_numba(ode_fun, state0, control, p, tvec, auxdata):

    ode_rtol = auxdata['param']['ode_rtol']
    ode_atol = auxdata['param']['ode_atol']
    use_numba = ode_fun.__name__.endswith("_numba")
    if use_numba:
        params = [auxdata['param'][name] for name in auxdata['param']['constant_names']]
    
    
    if len(tvec) > 2:
        if use_numba:
            sol = odeint(ode_fun, state0, tvec, args=(control, p, *params,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
        else:
            sol = odeint(ode_fun, state0, tvec, args=(control, p, auxdata,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
    else:
        tspan = np.array([tvec[0], tvec[-1]])
        if use_numba:
            sol = solve_ivp(ode_fun, tspan, state0, args=(control, p, *params,), rtol=ode_rtol, atol=ode_atol)
        else:
            sol = solve_ivp(ode_fun, tspan, state0, args=(control, p, auxdata,), rtol=ode_rtol, atol=ode_atol)            

    
    return sol


def propagate_odeint(ode_fun, state0, control, p, tvec, auxdata):

    ode_rtol = auxdata['param']['ode_rtol']
    ode_atol = auxdata['param']['ode_atol']
    use_numba = ode_fun.__name__.endswith("_numba")
    if use_numba:
        params = [auxdata['param'][name] for name in auxdata['param']['constant_names']]
        sol = odeint(ode_fun, state0, tvec, args=(control, p, *params,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
    else:
        sol = odeint(ode_fun, state0, tvec, args=(control, p, auxdata,), tfirst=True, rtol=ode_rtol, atol=ode_atol)
    return sol



def propagate_ivp(ode_fun, state0, control, p, tvec, auxdata):

    ode_rtol = auxdata['param']['ode_rtol']
    ode_atol = auxdata['param']['ode_atol']
    
    
    
    if len(tvec) > 2:
        teval = tvec
    else:
        teval = None

    tspan = np.array([tvec[0], tvec[-1]])
    use_numba = ode_fun.__name__.endswith("_numba")
    if use_numba:
        params = [auxdata['param'][name] for name in auxdata['param']['constant_names']]
        sol = solve_ivp(ode_fun, tspan, state0, t_eval=teval, args=(control, p, *params,), rtol=ode_rtol, atol=ode_atol)
    else:
        sol = solve_ivp(ode_fun, tspan, state0, t_eval=teval, args=(control, p, auxdata,), rtol=ode_rtol, atol=ode_atol)
    
    return sol



# Define function to propagate with event detection
def propagate_event(ode_fun, x0, control, p, tspan, auxdata, plane='xz', direction=0, isterminal=1):
    """
    Propagate the state in the CR3BP system with an event function for plane crossing.

    Parameters:
    - ode_fun : function : The CR3BP differential equations function
    - x0_stm : array : Initial state and STM (state transition matrix) vector
    - tspan : tuple : Integration time span (start, end)
    - mu_cr3bp : float : CR3BP system parameter
    - ode_rtol : float : Relative tolerance for the solver
    - ode_atol : float : Absolute tolerance for the solver
    - plane : str : Plane to detect crossing ('yz', 'xz', 'xy')
    - direction : int : Direction of crossing (-1, 0, 1)
    - isterminal : int : Termination flag (1 if event terminates the integration)

    Returns:
    - result : Bunch object with integration results and events.
    """
    
    plane_event = plane_crossing_event(plane=plane, direction=direction, isterminal=isterminal)
    
    ode_rtol = auxdata['param']['ode_rtol']
    ode_atol = auxdata['param']['ode_atol']

    # Run the solver with the event function
    use_numba = ode_fun.__name__.endswith("_numba")
    if use_numba:
        params = [auxdata['param'][name] for name in auxdata['param']['constant_names']]
        result = solve_ivp(
            ode_fun, tspan, x0, args=(control, p, *params,),
            rtol=ode_rtol, atol=ode_atol, events=plane_event
        )
    else:
        result = solve_ivp(
            ode_fun, tspan, x0, args=(control, p, auxdata,),
            rtol=ode_rtol, atol=ode_atol, events=plane_event
        )
    
    # Unpack result, including event times and states
    time = result.t
    state = result.y
    # time_event = result.t_events if result.t_events else []
#     state_event = result.y_events if result.y_events else []
    time_event = result.t_events[0] if result.t_events else []
    state_event = result.y_events[0] if result.y_events else []
    ind_event = 0 if time_event else None  # Index of event occurrence, None if no event

    return time, state, time_event, state_event, ind_event
