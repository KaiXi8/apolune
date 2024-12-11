import numpy as np
import dynamics_coeff.dynamics_utils as dyn_utils

def get_mu_bodies(mu):
    return np.array([1-mu, mu])
    

def get_body_positions(mu):
    rho_p = np.array([-mu, 0.0, 0.0])
    rho_s = np.array([1-mu, 0.0, 0.0])

    return np.vstack((rho_p, rho_s))


def omega(rho, mu):
    x, y = rho[:2]
    rho_p = get_body_positions(mu)
    mu_adim = get_mu_bodies(mu)
    
    om1 = dyn_utils.compute_omega(rho, rho_p, mu_adim)
    om2 = (x**2 + y**2) / 2
    
    return om1 + om2



def grad_omega(rho, mu):
    x, y = rho[:2]
    rho_p = get_body_positions(mu)
    mu_adim = get_mu_bodies(mu)
    
    grad_om = dyn_utils.compute_grad_omega(rho, rho_p, mu_adim)
    
    return grad_om



def jac_grad_omega(rho, mu):
    rho_p = get_body_positions(mu)
    mu_adim = get_mu_bodies(mu)

    return dyn_utils.compute_jac_grad_omega(rho, rho_p, mu_adim)

 

def compute_coeff():
    b = np.zeros(13)
    b[4] = 2.0
    b[6] = 1.0
    b[9] = 1.0
    b[12] = 1.0
    
    return b
    