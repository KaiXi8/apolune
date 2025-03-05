import numpy as np
from numba import jit, prange
import dynamics_coeff.crtbp_dynamics as crtbp
import dynamics_coeff.bcrfbp_dynamics as bcrfbp
import dynamics_coeff.rnbp_rpf_dynamics_nonuniform as rnbp_rpf
from dynamics_coeff.homotopy import eval_homotopy_at_point


def dynamics(tau, state, control, p, auxdata):
    rho = state[:3]
    eta = state[3:6]
    
    model = auxdata["model"]
    
    if model == 1: # crtbp
        mu_crtbp = auxdata["mu_crtbp"]

        b = crtbp.compute_coeff()
        grad_om_adim = crtbp.grad_omega(rho, mu_crtbp)
    elif model == 2: # bcrfbp
        mu_crtbp = auxdata["mu_crtbp"]
        mu_sun = auxdata["mu_sun"]
        a_sun = auxdata["a_sun"]
        om_sun = auxdata["om_sun"]
        sun_angle_t0 = auxdata["sun_angle_t0"]
        
        b = bcrfbp.compute_coeff(tau, mu_sun, a_sun, om_sun, sun_angle_t0)
        grad_om_adim = bcrfbp.grad_omega(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0)
    elif model == 3 or model == 4: #3 is rpnbp, 4 uses manually sent in coefficeints
        id_primary = auxdata["id_primary"]
        id_secondary = auxdata["id_secondary"]
        mu_bodies_dim = auxdata["mu_bodies"]
        naif_id_bodies = auxdata["naif_id_bodies"]
        observer_id = auxdata["observer_id"]
        reference_frame = auxdata["reference_frame"]
        epoch_t0 = auxdata["epoch_t0"]
        tau_vec = auxdata["tau_vec"]
        t_vec = auxdata["t_vec"]
        if model == 3:
            b, grad_om_adim = rnbp_rpf.compute_coeff_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec)
        else:
            b = np.zeros(13)
            for i1 in range(len(auxdata['coeff_3bp'])):
                b[i1] = eval_homotopy_at_point( auxdata['sel_homotopy'], auxdata['homot_param'], auxdata["tau_vec"], tau, auxdata['coeff_3bp'][i1], auxdata['f_precomputed'][i1] )
            grad_om_adim = rnbp_rpf.compute_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec) #compute grad without computing b1,...
        
    g0 = auxdata["g0"]
    Isp = auxdata["Isp"]
    
    return dynamics_coeff(state, control, b, grad_om_adim, g0, Isp)


def dynamics_coeff(state, control, b, grad_om_adim, g0, Isp):
    rho = state[:3]
    eta = state[3:6]
    
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = b
    
    rho_dot_matrix = np.array([
        [b4, b5, 0.0],
        [-b5, b4, b6],
        [0.0, -b6, b4]
        ])
    rho_matrix = np.array([
        [b7, b9, b8],
        [-b9, b10, b11],
        [b8, -b11, b12]
        ])

    drho = eta
    deta = b[:3] + rho_dot_matrix @ eta + rho_matrix @ rho + b13 * grad_om_adim + control[:3]
    dz = - control[3] / (g0*Isp)
    
#     return np.hstack((drho, deta, dz))
    return np.concatenate((drho, deta, [dz]))



def jacobian_coeff(b, jac_grad_om):

    I3 = np.identity(3)
    Z3 = np.zeros((3, 3))
    
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = b
    
    rho_dot_matrix = np.array([
        [b4, b5, 0.0],
        [-b5, b4, b6],
        [0.0, -b6, b4]
        ])
    rho_matrix = np.array([
        [b7, b9, b8],
        [-b9, b10, b11],
        [b8, -b11, b12]
        ])

    A11 = Z3  # drhodot / drho
    A12 = I3  # drhodot / deta
    A21 = rho_matrix + b13 * jac_grad_om # detadot / drho
    A22 = rho_dot_matrix # detadot / deta

    zero_col = np.zeros((3, 1))  # 3x1 column of zeros
    zero_row = np.zeros((1, 7))  # 1x7 row of zeros
    
#     return np.block([[A11, A12], [A21, A22]])
    return np.block([
        [A11, A12, zero_col],  # First 3 rows
        [A21, A22, zero_col],  # Next 3 rows
        [zero_row]             # Last row
        ])



def jacobian_x(tau, state, control, p, auxdata):
    rho = state[:3]
    eta = state[3:6]
    
    model = auxdata["model"]
    
    if model == 1: # crtbp
        mu_crtbp = auxdata["mu_crtbp"]

        b = crtbp.compute_coeff()
        jac_grad_om_adim = crtbp.jac_grad_omega(rho, mu_crtbp)
    elif model == 2: # bcrfbp
        mu_crtbp = auxdata["mu_crtbp"]
        mu_sun = auxdata["mu_sun"]
        a_sun = auxdata["a_sun"]
        om_sun = auxdata["om_sun"]
        sun_angle_t0 = auxdata["sun_angle_t0"]
        
        b = bcrfbp.compute_coeff(tau, mu_sun, a_sun, om_sun, sun_angle_t0)
        jac_grad_om_adim = bcrfbp.jac_grad_omega(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0)
    elif model == 3 or model == 4:
        id_primary = auxdata["id_primary"]
        id_secondary = auxdata["id_secondary"]
        mu_bodies_dim = auxdata["mu_bodies"]
        naif_id_bodies = auxdata["naif_id_bodies"]
        observer_id = auxdata["observer_id"]
        reference_frame = auxdata["reference_frame"]
        epoch_t0 = auxdata["epoch_t0"]
        tau_vec = auxdata["tau_vec"]
        t_vec = auxdata["t_vec"]
        if model == 3:
            b, jac_grad_om_adim = rnbp_rpf.compute_coeff_jac_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec)
        else:
            b = np.zeros(13)
            for i1 in range(len(auxdata['coeff_3bp'])):
                b[i1] = eval_homotopy_at_point( auxdata['sel_homotopy'], auxdata['homot_param'], auxdata["tau_vec"], tau, auxdata['coeff_3bp'][i1], auxdata['f_precomputed'][i1] )
            jac_grad_om_adim = rnbp_rpf.compute_jac_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec) #compute grad without computing b1,...
    
    return jacobian_coeff(b, jac_grad_om_adim)


def jacobian_u(tau, state, control, p, auxdata):
#     g0 = auxdata['param']['g0']
#     Isp = auxdata['param']['Isp']
    
    g0 = auxdata['g0']
    Isp = auxdata['Isp']

    # first-order ODEs for STM
    u_stm = np.zeros((7, 4))
    u_stm[3:6, 0:3] = np.eye(3)
    u_stm[-1, -1] = -1/(g0*Isp)
    
    return u_stm
