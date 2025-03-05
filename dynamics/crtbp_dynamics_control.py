import numpy as np


def u_bar_first_partials(state, mu_cr3bp):

    r_13 = ((state[0] + mu_cr3bp) * (state[0] + mu_cr3bp) +
            state[1] * state[1] + state[2] * state[2]) ** (-1.5)  # 1/r1^3
    r_23 = (((state[0] - 1.0 + mu_cr3bp) * (state[0] - 1.0 + mu_cr3bp)) +
            state[1] * state[1] + state[2] * state[2]) ** (-1.5)  # 1/r2^3

    u_bar = np.empty(3)  # preallocate array to store the partials

    u_bar[0] = mu_cr3bp * (state[0] - 1.0 + mu_cr3bp) * r_23 + \
        (1.0 - mu_cr3bp) * (state[0] + mu_cr3bp) * r_13 - state[0]
    u_bar[1] = mu_cr3bp * state[1] * r_23 + (1.0 - mu_cr3bp) * state[1] * r_13 - state[1]
    u_bar[2] = mu_cr3bp * state[2] * r_23 + (1.0 - mu_cr3bp) * state[2] * r_13

    return u_bar


def u_bar_second_partials(state, mu_cr3bp):

    r_12 = (state[0] + mu_cr3bp) * (state[0] + mu_cr3bp) + state[1] * state[1] + \
        state[2] * state[2]  # r1^2
    r_22 = ((state[0] - 1.0 + mu_cr3bp) * (state[0] - 1.0 + mu_cr3bp)) + state[1] * state[1] + \
        state[2] * state[2]  # r2^2

    r_13 = r_12 ** (-1.5)  # 1/r1^3
    r_23 = r_22 ** (-1.5)  # 1/r2^3
    r_15 = r_12 ** (-2.5)  # 1/r1^5
    r_25 = r_22 ** (-2.5)  # 1/r2^5

    u_bar2 = np.empty((3, 3))

    u_bar2[0, 0] = mu_cr3bp * r_23 + (1.0 - mu_cr3bp) * r_13 - \
        3.0 * mu_cr3bp * (state[0] - 1.0 + mu_cr3bp) * (state[0] - 1.0 + mu_cr3bp) * r_25 - \
        3.0 * (state[0] + mu_cr3bp) * (state[0] + mu_cr3bp) * (1.0 - mu_cr3bp) * r_15 - 1.0

    u_bar2[0, 1] = 3.0 * state[1] * (mu_cr3bp + state[0]) * (mu_cr3bp - 1.0) * r_15 - \
        3.0 * mu_cr3bp * state[1] * (state[0] - 1.0 + mu_cr3bp) * r_25

    u_bar2[0, 2] = 3.0 * state[2] * (mu_cr3bp + state[0]) * (mu_cr3bp - 1.0) * r_15 - \
        3.0 * mu_cr3bp * state[2] * (state[0] - 1.0 + mu_cr3bp) * r_25

    u_bar2[1, 0] = u_bar2[0, 1]

    u_bar2[1, 1] = mu_cr3bp * r_23 - (mu_cr3bp - 1.0) * r_13 + \
        3.0 * state[1] * state[1] * (mu_cr3bp - 1.0) * r_15 - \
        3.0 * mu_cr3bp * state[1] ** 2 * r_25 - 1.0

    u_bar2[1, 2] = 3.0 * state[1] * state[2] * (mu_cr3bp - 1.0) * r_15 - \
        3.0 * mu_cr3bp * state[1] * state[2] * r_25

    u_bar2[2, 0] = u_bar2[0, 2]
    u_bar2[2, 1] = u_bar2[1, 2]

    u_bar2[2, 2] = mu_cr3bp * r_23 - (mu_cr3bp - 1.0) * r_13 + \
        3.0 * state[2] ** 2 * (mu_cr3bp - 1.0) * r_15 - 3.0 * mu_cr3bp * state[2] ** 2 * r_25

    return u_bar2


def dynamics(t, state, control, p, auxdata):
    
    mu_cr3bp = auxdata['param']['mu']
    g0 = auxdata['param']['g0']
    Isp = auxdata['param']['Isp']

    u_bar = u_bar_first_partials(state, mu_cr3bp)

    dot_state = np.empty(7)  # preallocate array for state derivatives
    dot_state[0] = state[3]
    dot_state[1] = state[4]
    dot_state[2] = state[5]
    dot_state[3] = 2.0 * state[4] - u_bar[0] + control[0]
    dot_state[4] = - 2.0 * state[3] - u_bar[1] + control[1]
    dot_state[5] = - u_bar[2] + control[2]
    dot_state[6] = - control[3] / (g0*Isp)

    return dot_state


def dynamics_free(t, state, p, auxdata):
    
    mu_cr3bp = auxdata['param']['mu']
    g0 = auxdata['param']['g0']
    Isp = auxdata['param']['Isp']

    u_bar = u_bar_first_partials(state, mu_cr3bp)

    dot_state = np.empty(7)  # preallocate array for state derivatives
    dot_state[0] = state[3]
    dot_state[1] = state[4]
    dot_state[2] = state[5]
    dot_state[3] = 2.0 * state[4] - u_bar[0]
    dot_state[4] = - 2.0 * state[3] - u_bar[1]
    dot_state[5] = - u_bar[2]
    dot_state[6] = 0.0

    return dot_state
    
    
def jacobian_x(t, states, controls, p, auxdata):
    mu_cr3bp = auxdata['param']['mu']

    # first-order ODEs for STM
    x_stm = np.zeros((7, 7))
    x_stm[0:3, 3:6] = np.eye(3)
    x_stm[3:6, 0:3] = - u_bar_second_partials(states, mu_cr3bp)
    x_stm[3, 4] = 2.0
    x_stm[4, 3] = - 2.0
    
    return x_stm
    

def jacobian_u(t, states, controls, p, auxdata):
    g0 = auxdata['param']['g0']
    Isp = auxdata['param']['Isp']

    # first-order ODEs for STM
    u_stm = np.zeros((7, 4))
    u_stm[3:6, 0:3] = np.eye(3)
    u_stm[-1, -1] = -1/(g0*Isp)
    
    return u_stm


def dynamics_stm(t, state, control, p, auxdata):
    state_dot = dynamics(t, state, control, p, auxdata)
    jac_x = jacobian_x(t, state, control, p, auxdata)
    
    return np.concatenate((state_dot, jac_x.flatten()))


def u_bar(state, mu_var):
    """Augmented potential (from Koon et al. 2011, chapter 2). """
    x_var, y_var, z_var = state
    mu1 = 1 - mu_var
    mu2 = mu_var
    # where r1 and r2 are expressed in rotating coordinates
    r1_var = ((x_var+mu2)**2 + y_var**2 + z_var**2)**(1/2)
    r2_var = ((x_var-mu1)**2 + y_var**2 + z_var**2)**(1/2)
    aug_pot = -1/2*(x_var**2+y_var**2) - mu1/r1_var - mu2/r2_var - 1/2*mu1*mu2
    return aug_pot

def jacobi(state, mu_var):
    """Computes the jacobi constant at a given state in the CRTBP system with the mass ratio mu.
    See equation 2.3.14 of Koon et al. 2011. """
    if len(state) == 3:  # if only position is given, velocity is supposed null
        jacobi_cst = -2*u_bar(state, mu_var)
    elif len(state) == 6:
        jacobi_cst = -2*u_bar(state[0:3], mu_var) - (state[3]**2 + state[4]**2 + state[5]**2)
    else:
        raise Exception('State dimension wrong. State dimensions must be 3 or 6')
    return jacobi_cst