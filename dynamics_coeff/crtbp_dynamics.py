import numpy as np
from numba import jit
# import dynamics_coeff.dynamics_utils as dyn_utils
import dynamics_coeff.dynamics_utils_jit as dyn_utils


@jit('float64[::1](float64)', nopython=True, nogil=True, fastmath=True)
def get_mu_bodies(mu):
    return np.array([1-mu, mu])
    

@jit('float64[:,::1](float64)', nopython=True, nogil=True, fastmath=True)
def get_body_positions(mu):
    rho_p = np.array([-mu, 0.0, 0.0])
    rho_s = np.array([1-mu, 0.0, 0.0])
    
    pos = np.empty((2,3), dtype=np.float64) 
    pos[0,:3] = rho_p
    pos[1,:3] = rho_s

    return pos
#     return np.vstack((rho_p, rho_s))


@jit('float64(float64[::1], float64)', nopython=True, nogil=True, fastmath=True)
def omega(rho, mu):
    x, y = rho[:2]
    rho_p = get_body_positions(mu)
    mu_adim = get_mu_bodies(mu)
    
    om1 = dyn_utils.compute_omega(rho, rho_p, mu_adim)
    om2 = (x**2 + y**2) / 2
    
    return om1 + om2


@jit('float64[::1](float64[::1], float64)', nopython=True, nogil=True, fastmath=True)
def grad_omega(rho, mu):
    x, y = rho[:2]
    rho_p = get_body_positions(mu)
    mu_adim = get_mu_bodies(mu)
    
    grad_om = dyn_utils.compute_grad_omega(rho, rho_p, mu_adim)
    
    return grad_om


@jit('float64[:,::1](float64[::1], float64)', nopython=True, nogil=True, fastmath=True)
def jac_grad_omega(rho, mu):
    rho_p = get_body_positions(mu)
    mu_adim = get_mu_bodies(mu)

    return dyn_utils.compute_jac_grad_omega(rho, rho_p, mu_adim)

 
@jit('float64[::1]()', nopython=True, nogil=True, fastmath=True)
def compute_coeff():
    b = np.zeros(13)
    b[4] = 2.0
    b[6] = 1.0
    b[9] = 1.0
    b[12] = 1.0
    
    return b
    