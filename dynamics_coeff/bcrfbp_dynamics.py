import numpy as np
import dynamics_coeff.crtbp_dynamics as dyn_crtbp
import dynamics_coeff.dynamics_utils as dyn_utils


def get_mu_bodies(mu, mu_4):
    return np.array([1-mu, mu, mu_4])
    

def get_body_positions(tau, mu, a_sun, om_sun, sun_angle_t0):
    rho_ps = dyn_crtbp.get_body_positions(mu)
    
    rho_sun = compute_sun_position(tau, a_sun, om_sun, sun_angle_t0)

    return np.vstack((rho_ps, rho_sun))
    
    
    
def compute_sun_angle(tau, om_sun, sun_angle_t0):
    return om_sun * tau + sun_angle_t0



def compute_sun_position(tau, a_sun, om_sun, sun_angle_t0):
    sun_angle = compute_sun_angle(tau, om_sun, sun_angle_t0)
    rho_sun = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0])
    return rho_sun


def omega_bcrfbp(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0):

    rho_p = get_body_positions(tau, mu_crtbp, a_sun, om_sun, sun_angle_t0)
    mu_adim = get_mu_bodies(mu_crtbp, mu_sun)
    
    return dyn_utils.compute_omega(rho, rho_p, mu_adim)



def grad_omega(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0):

    rho_p = get_body_positions(tau, mu_crtbp, a_sun, om_sun, sun_angle_t0)
    mu_adim = get_mu_bodies(mu_crtbp, mu_sun)
    
    return dyn_utils.compute_grad_omega(rho, rho_p, mu_adim)



def jac_grad_omega(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0):

    rho_p = get_body_positions(tau, mu_crtbp, a_sun, om_sun, sun_angle_t0)
    mu_adim = get_mu_bodies(mu_crtbp, mu_sun)
    
    return dyn_utils.compute_jac_grad_omega(rho, rho_p, mu_adim)




def compute_coeff(tau, mu_sun, a_sun, om_sun, sun_angle_t0):
    b = np.zeros(13)
    
    sun_angle = compute_sun_angle(tau, om_sun, sun_angle_t0)
    tmp = - mu_sun / a_sun**2
    b[0] = tmp * np.cos(sun_angle)
    b[1] = tmp * np.sin(sun_angle)
    b[4] = 2.0
    b[6] = 1.0
    b[9] = 1.0
    b[12] = 1.0
    
    return b