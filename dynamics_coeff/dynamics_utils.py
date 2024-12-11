import numpy as np

def compute_omega(rho, rho_p, mu_bodies_adim):
    # Synodic relative position of spacecraft wrt celestial bodies
    diff_rho = rho - rho_p
    
    diff_rho = np.atleast_1d(diff_rho)
    mu_bodies_adim = np.atleast_1d(mu_bodies_adim)

    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) # [adim]
    
    # calculate omega
    om = (mu_bodies_adim / norm_rho) 

    return om.sum()



def compute_grad_omega(rho, rho_p, mu_bodies_adim):
    # Synodic relative position of spacecraft wrt celestial bodies
    diff_rho = rho - rho_p
    
    diff_rho = np.atleast_1d(diff_rho)
    mu_bodies_adim = np.atleast_1d(mu_bodies_adim)
    
    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) # [adim]

    # Gradient of potential energy 
    factor = (mu_bodies_adim / norm_rho**3)[:, np.newaxis] # [adim]
    grad_om = - factor * diff_rho # [adim]
    
    # Sum along the rows to get the final gradient
    grad = grad_om.sum(axis=0) # [adim]

    return grad 



def compute_jac_grad_omega(rho, rho_p, mu_bodies_adim):
    # Synodic relative position of spacecraft wrt celestial bodies
    diff_rho = rho - rho_p 
    
    diff_rho = np.atleast_1d(diff_rho)
    mu_bodies_adim = np.atleast_1d(mu_bodies_adim)
    
    norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) 

    I3 = np.identity(3)
    
    num_bodies = len(mu_bodies_adim)
    
    # Term 1 of gradient of potential energy
    grad_om_term1 = - np.sum(mu_bodies_adim / norm_rho**3) * I3

    # Term 2 of gradient of potential energy
    om_mat = (mu_bodies_adim / norm_rho**5)[:, np.newaxis] * diff_rho
    grad_om_term2 = np.zeros((3, 3))
    for jj in range(num_bodies):
        grad_om_term2 += 3 * np.outer(om_mat[jj, :], diff_rho[jj, :])

    return grad_om_term1 + grad_om_term2


