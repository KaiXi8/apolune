import numpy as np
import matplotlib.pyplot as plt

def threebody(t, x, mu):
    """
    The ODE function for the orbit in the CR3BP

    Inputs:
    - t: The time
    - x: A numpy array of 6 elements, the orbit states x, y, z, x_dot, y_dot, z_dot
    - mu: The only parameter of importance in the CR3BP
    """

    r = np.linalg.norm(x[0:3])

    xdot = np.empty((6,))
    xdot[0:3] = x[3:6]

    r1 = np.sqrt((x[0] + mu)**2. + x[1]**2. + x[2]**2.) 
    r2 = np.sqrt((x[0] - 1. + mu)**2. + x[1]**2. + x[2]**2.)

    xdot[3] = 2.*x[4] + x[0] - ((1. - mu)*(x[0] + mu)/(r1**3.)) - (mu*(x[0] - 1. + mu)/(r2**3.))
    xdot[4] = -2.*x[3] + x[1] - ((1. - mu)*x[1]/(r1**3.)) - (mu*x[1]/(r2**3.))
    xdot[5] = -((1. - mu)*x[2]/(r1**3.)) - (mu*x[2]/(r2**3.))

    return xdot
    
def U_hessian(mu, X):
    """
    Compute U_xx, U_xy, U_yy, Uyz, Uzz
    Inputs:
    - mu: ...
    - X: State
    """
    r1 = np.sqrt((X[0] + mu) ** 2. + X[1] ** 2. + X[2] ** 2.)
    r2 = np.sqrt((X[0] - 1. + mu) ** 2. + X[1] ** 2. + X[2] ** 2.)
    x = X[0]
    y = X[1]
    z = X[2]
    Uxx = 1. - ((1. - mu)/(r1**3.)) - (mu/(r2**3.)) + (3.*(1. - mu)*((x + mu)**2.)/(r1**5.)) + \
          (3.*mu*((x - 1. + mu)**2.)/(r2**5.))
    Uxy = (3.*(1. - mu)*(x + mu)*y/(r1**5.)) + (3.*mu*(x - 1. + mu)*y/(r2**5.))
    Uyy = 1. - ((1. - mu)/(r1**3.)) - (mu/(r2**3.)) + (3.*(1. - mu)*(y**2.)/(r1**5.)) + (3.*mu*(y**2.)/(r2**5.))
    Uxz = 3.*((1. - mu)*(x + mu)*z/(r1**5.)) + 3.*(mu*(x - 1. + mu)*z/(r2**5.))
    Uyz = 3.*((1. - mu)*y*z/(r1**5.)) + 3*(mu*y*z/(r2**5.))
    Uzz = -((1. - mu)/(r1**3.)) - (mu/(r2**3.)) + 3.*((1. - mu)*(z**2.)/(r1**5.)) + 3.*mu*((z**2.)/(r2**5.))
    Uxx_mat = np.array([[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]])
    
    return Uxx_mat

def A_mat_CR3BP(mu, X):
    """
    Compute the general plant matrix for CR3BP
    Inputs:
    - mu: mass-ratio
    - X: State
    """
    U_mat = U_hessian(mu, X)    
    A_mat = np.zeros((6, 6))
    A_mat[0:3, 3:6] = np.identity(3)
    A_mat[3:6, 0:3] = U_mat
    A_mat[3, 4] = 2.
    A_mat[4, 3] = -2.

    return A_mat

def ode_STM_cr3bp(t, x, mu):
    '''
    The ODE function for the orbit and STM components, combined

    Inputs:
    - t: The time
    - x: A numpy array of 42 elements. The first six are the orbit states, the remainder are STM elements
    - mu: mass-ratio
    '''

    # Build the A matrix for phi_dot = A*phi:
    A_local = A_mat_CR3BP(mu, x[0:6])

    x_dot = np.zeros((42,))

    # Orbital Dynamics:
    x_dot[0:6] = threebody(t, x[0:6], mu)

    # State Transition Matrix Dynamics:
    phi = np.reshape(x[6:42], (6, 6), order='F')
    phi_dot = np.dot(A_local, phi)  # np.matmul(A_local, phi)
    phi_dot_reshape = np.reshape(phi_dot, 36, order='F')
    x_dot[6:42] = phi_dot_reshape

    return x_dot

def jacobi(states, mu):
    """
    Compute the Jacobi constant for an array of states
    Args:
        states (array(n, 6)): Array of n states, each with 6 elements
        mu (float): mass-ratio

    Returns:
        C (array(n, )): Array of Jacobi constants
    """
    
    states = np.transpose(np.asarray(states, dtype=float))
    
    if np.shape(states) == (6,): # Only 1 state case
        x = states[0]; y = states[1]; z = states[2]
        x_dot = states[3]; y_dot = states[4]; z_dot = states[5]
    elif len(states) == 4: # Planar case
        x = states[0]; y = states[1]; z = 0
        x_dot = states[2]; y_dot = states[3]; z_dot = 0
    else:
        x = states[:, 0]; y = states[:, 1]; z = states[:, 2]
        x_dot = states[:, 3]; y_dot = states[:, 4]; z_dot = states[:, 5]

    r = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    d = np.sqrt((x + mu)**2 + y**2 + z**2)
    U = (x**2 + y**2) / 2 + mu / r + (1 - mu) / d
    v = np.sqrt(x_dot**2 + y_dot**2 + z_dot**2)
    C = 2 * U - v**2
    
    return C


def fourbody_ES(t, x, mu):
    """ The ODE function for the orbit in the BCR4BP in the Earth-Sun centered frame
    
    Inputs:
    - t: the time
    - x: A numpy array of 4 elements, the orbit states x, y, x_dot, y_dot
    - mu: The CR3BP mass-ratio parameter.
    """
    xdot = threebody_orbit(x, mu) 

    LU = 1.49598E+8 # Earth-Sun distance, or 1 AU, [km]
    TU = 5022635 # seconds
    # tau = # Time non-dim scaling factor

    mu_m = 4.9028695E+3 # Gravitational parameter of the Moon [km^3 s^-2]
    R_m = 384400 / LU # Orbital radius of the Moon about the Earth [-]
    moon_earth_orbit_seconds = 2360591.78
    w_m = 2*math.pi/moon_earth_orbit_seconds # Angular velocity of the Moon about the Earth

    r_m = [1-mu + R_m * np.cos(w_m * t),
            R_m * np.cos(w_m * t)] 

    r = x[0:2] - r_m # Vector from S/C to the Moon
    
    a2b = -mu_m * r / np.norm(r) ** 3

    a2b = a2b * TU**2/LU # Non-dimensionalize acceleration

    xdot[2:4] += a2b

    return xdot

def threebody_orbit(t, x, mu):
    """
    The ODE function for the orbit in the CR3BP

    Inputs:
    - t: The time
    - x: A numpy array of 4 elements, the orbit states x, y, x_dot, y_dot
    - mu: The only parameter of importance in the CR3BP
    """
    r = np.linalg.norm(x[0:2])

    xdot = np.empty((4,))
    xdot[0:2] = x[2:5]

    r1 = np.sqrt((x[0] + mu)**2. + x[1]**2.)
    r2 = np.sqrt((x[0] - 1. + mu)**2. + x[1]**2.)

    xdot[2] = 2.*x[3] + x[0] - ((1. - mu)*(x[0] + mu)/(r1**3.)) - (mu*(x[0] - 1. + mu)/(r2**3.))
    xdot[3] = -2.*x[2] + x[1] - ((1. - mu)*x[1]/(r1**3.)) - (mu*x[1]/(r2**3.))
    #xdot[5] = -((1. - mu)*x[2]/(r1**3.)) - (mu*x[2]/(r2**3.))

    return xdot
