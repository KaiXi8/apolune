import numpy as np

def xisval(t, x, crit_val):
    """
    This function returns when x = crit_val, for use as an "events" function
    - t: time
    - x: state
    """

    if t < 0.01:  # Ignore the initial crossing time, for Poincaré maps
        value = 1.0
    else:
        value = x[0] - crit_val

    return value

def velocity_planar(mu, y, C):
    """
    This function returns the resultant velocity along the line x = 1 - mu, for a given Jacobi constant C
    Inputs:
    - mu:
    - y:
    - C:
    """

    v_sq = ((1. - mu)**2.) + y**2. + 2.*((1.-mu)/np.sqrt(y**2. + 1)) + 2.*(mu/np.sqrt(y**2.)) - C

    flag = 'inside'
    if v_sq >= 0.:
        v = np.sqrt(v_sq)
    else:
        v = 0.
        flag = 'outside'  # Reject un-physical samples requiring v^2 < 0 for same Jacobi constant

    return v, flag

def n_crossing_integrator(mu, ode_func, events_func, ICs, n_max, tf):
    """ Integrate the state to cross surface of section n_max times """

    # Integrator function:
    fun1 = lambda t, x: ode_func(t, x, mu)
    # Set a stopping condition function:
    terminal_func = lambda t, x: events_func(t, x, 1. - mu)  # This uses xisval as "events_func"
    terminal_func.terminal = True
    terminal_func.direction = +1 # Positive xdot

    # Time vector:
    time_vec = np.linspace(0., tf, 500)

    count = 0
    break_idx = n_max
    event_states = np.zeros((n_max, 6))
    t_events = np.zeros((n_max,))
    while count < n_max:
        # Integrate:
        sol = solve_ivp(fun1, [time_vec[0], time_vec[-1]], ICs, method='DOP853', t_eval=time_vec,
                                   events=terminal_func, dense_output=True, rtol=1e-13, atol=1e-14)
        # state_nondim = np.transpose(sol.y)  # m x 6
        check_time = sol.t_events[0]
        if check_time.shape[0] == 0:  # Check if integration failure
            break_idx = count
            str_print = 'In n_crossing_integrator, stopping condition not met at count ' + str(count)
            print(str_print)
            break
        else:
            final_time = sol.t_events[0][0]
            final_state = np.array(sol.y_events[0][0])
            # Save the data:
            event_states[count, :] = final_state
            t_events[count] = final_time
            # Update:
            ICs = final_state
            count += 1

    # Trim excess zeros if there was a break
    if (break_idx < n_max) and (break_idx != 0):
        event_states = event_states[0:break_idx, :]
        t_events = t_events[0:break_idx]

    return t_events, event_states
   