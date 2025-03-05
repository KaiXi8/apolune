import numpy as np
from numba import jit, prange
import dynamics_coeff.crtbp_dynamics as crtbp
import dynamics_coeff.bcrfbp_dynamics as bcrfbp
import dynamics_coeff.rnbp_rpf_dynamics_nonuniform_jit as rnbp_rpf
from dynamics_coeff.homotopy import eval_homotopy_at_point

# @jit('float64(float64[::1], float64[:,::1], float64[::1])', nopython=True, nogil=True, fastmath=True)
# def compute_omega(rho, rho_p, mu_bodies_adim):
#     # Synodic relative position of spacecraft wrt celestial bodies
#     diff_rho = rho - rho_p
#     
#     diff_rho = np.atleast_1d(diff_rho)
#     mu_bodies_adim = np.atleast_1d(mu_bodies_adim)
# 
#     norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) # [adim]
#     
#     # calculate omega
#     om = (mu_bodies_adim / norm_rho) 
# 
#     return om.sum()
# 
# 
# 
# @jit('float64[::1](float64[::1], float64[:,::1], float64[::1])', nopython=True, nogil=True, fastmath=True)
# def compute_grad_omega(rho, rho_p, mu_bodies_adim):
#     # Synodic relative position of spacecraft wrt celestial bodies
#     diff_rho = rho - rho_p
#     
#     diff_rho = np.atleast_1d(diff_rho)
#     mu_bodies_adim = np.atleast_1d(mu_bodies_adim)
#     
#     norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) # [adim]
# 
#     # Gradient of potential energy 
#     factor = (mu_bodies_adim / norm_rho**3)[:, np.newaxis] # [adim]
#     grad_om = - factor * diff_rho # [adim]
#     
#     # Sum along the rows to get the final gradient
#     grad = grad_om.sum(axis=0) # [adim]
# 
#     return grad 
# 
# 
# 
# @jit('float64[:,::1](float64[::1], float64[:,::1], float64[::1])', nopython=True, nogil=True, fastmath=True)
# def compute_jac_grad_omega(rho, rho_p, mu_bodies_adim):
#     # Synodic relative position of spacecraft wrt celestial bodies
#     diff_rho = rho - rho_p 
#     
#     diff_rho = np.atleast_1d(diff_rho)
#     mu_bodies_adim = np.atleast_1d(mu_bodies_adim)
#     
#     norm_rho = np.sqrt(np.sum(diff_rho ** 2, axis=1)) 
# 
#     I3 = np.identity(3)
#     
#     num_bodies = len(mu_bodies_adim)
#     
#     # Term 1 of gradient of potential energy
#     grad_om_term1 = - np.sum(mu_bodies_adim / norm_rho**3) * I3
# 
#     # Term 2 of gradient of potential energy
#     om_mat = (mu_bodies_adim / norm_rho**5)[:, np.newaxis] * diff_rho
#     grad_om_term2 = np.zeros((3, 3))
#     for jj in range(num_bodies):
#         grad_om_term2 += 3 * np.outer(om_mat[jj, :], diff_rho[jj, :])
# 
#     return grad_om_term1 + grad_om_term2



def dynamics(tau, state, control, p, auxdata):
    rho = state[:3]
    eta = state[3:6]
    
    model = auxdata["model"]
    
    if model == 1: # crtbp
        mu_crtbp = auxdata["mu_crtbp"]

        b = crtbp.compute_coeff()
        grad_om_adim = crtbp.grad_omega(rho, mu_crtbp)
        
        g0 = auxdata["g0"]
        Isp = auxdata["Isp"]
    elif model == 2: # bcrfbp
        mu_crtbp = auxdata["mu_crtbp"]
        mu_sun = auxdata["mu_sun"]
        a_sun = auxdata["a_sun"]
        om_sun = auxdata["om_sun"]
        sun_angle_t0 = auxdata["sun_angle_t0"]
        
        b = bcrfbp.compute_coeff(tau, mu_sun, a_sun, om_sun, sun_angle_t0)
        grad_om_adim = bcrfbp.grad_omega(tau, rho, mu_crtbp, mu_sun, a_sun, om_sun, sun_angle_t0)
        
        g0 = auxdata["g0"]
        Isp = auxdata["Isp"]
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
        
        g0_dim = auxdata["g0_dim"]
        Isp_dim = auxdata["Isp_dim"]
        
        mu_p = auxdata["mu_p"]
        mu_s = auxdata["mu_s"]
        l_vec = auxdata["l_vec"]
        g0, Isp = compute_g0_Isp_adim(g0_dim, Isp_dim, tau, tau_vec, l_vec, mu_p, mu_s)
        
        
        if model == 3:
            b, grad_om_adim = rnbp_rpf.compute_coeff_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec)
        else:
           #  b = np.zeros(13)
#             for i1 in range(len(auxdata['coeff_3bp'])):
#                  b[i1] = eval_homotopy_at_point( auxdata['sel_homotopy'], auxdata['homot_param'], auxdata["tau_vec"], tau, auxdata['coeff_3bp'][i1], auxdata['f_precomputed'][i1] )
            b = eval_homotopy_at_point( auxdata['sel_homotopy'], auxdata['homot_param'], auxdata["tau_vec"], tau, auxdata['coeff_3bp'], auxdata['f_precomputed'] )
            grad_om_adim = rnbp_rpf.compute_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec) #compute grad without computing b1,...
            if auxdata['homot_param'] < 1: #mix grads for homotopy
                grad_om_adim3bp = crtbp.grad_omega(rho, auxdata['mu_crtbp'])
                grad_om_adim = auxdata['homot_param'] * grad_om_adim + (1 - auxdata['homot_param']) * grad_om_adim3bp
    
#     g0 = auxdata["g0"]
#     Isp = auxdata["Isp"]
    
    return dynamics_coeff(state, control, b, grad_om_adim, g0, Isp)


# @jit
@jit('float64[::1](float64[::1], float64[::1], float64[::1], float64[::1], float64, float64)', nopython=True, nogil=True, fastmath=True)
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
    
    dx = np.empty(7, dtype=np.float64) 
    dx[:3] = drho
    dx[3:6] = deta
    dx[6] = dz
    
    return dx
    
#     return np.hstack((drho, deta, dz))
#     return np.concatenate((drho, deta, [dz]))


@jit('float64[:,::1](float64[::1], float64[:,::1])', nopython=True, nogil=True, fastmath=True)
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
    
    jac = np.zeros((7,7), dtype=np.float64)
    jac[:3, :3] = A11
    jac[:3, 3:6] = A12
    jac[3:6, :3] = A21  # Bottom-left block
    jac[3:6, 3:6] = A22  # Bottom-right block
    
    return jac
    
#     return np.block([[A11, A12], [A21, A22]])
    # return np.block([
#         [A11, A12, zero_col],  # First 3 rows
#         [A21, A22, zero_col],  # Next 3 rows
#         [zero_row]             # Last row
#     ])



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
           #  b = np.zeros(13)
#             for i1 in range(len(auxdata['coeff_3bp'])):
#                 b[i1] = eval_homotopy_at_point( auxdata['sel_homotopy'], auxdata['homot_param'], auxdata["tau_vec"], tau, auxdata['coeff_3bp'][i1], auxdata['f_precomputed'][i1] )
            b = eval_homotopy_at_point( auxdata['sel_homotopy'], auxdata['homot_param'], auxdata["tau_vec"], tau, auxdata['coeff_3bp'], auxdata['f_precomputed'] )
            jac_grad_om_adim = rnbp_rpf.compute_jac_grad(tau, state, id_primary, id_secondary, mu_bodies_dim, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec) #compute grad without computing b1,...
            if auxdata['homot_param'] < 1: #mix grads for homotopy
                jac_grad_om_adim3bp = crtbp.jac_grad_omega(rho, auxdata['mu_crtbp'])
                jac_grad_om_adim = auxdata['homot_param'] * jac_grad_om_adim + (1 - auxdata['homot_param']) * jac_grad_om_adim3bp
    
    return jacobian_coeff(b, jac_grad_om_adim)



def jacobian_u(tau, state, control, p, auxdata):
#     g0 = auxdata['param']['g0']
#     Isp = auxdata['param']['Isp']

    model = auxdata["model"]
    
    if model == 1 or model == 2: # crtbp or bcrfbp
#     if model == 1 or model == 2 or model == 3: # crtbp or bcrfbp
        g0 = auxdata['g0']
        Isp = auxdata['Isp']
    else: # rnbp_rpf
        g0_dim = auxdata["g0_dim"]
        Isp_dim = auxdata["Isp_dim"]
        mu_p = auxdata["mu_p"]
        mu_s = auxdata["mu_s"]
        tau_vec = auxdata["tau_vec"]
        l_vec = auxdata["l_vec"]
        g0, Isp = compute_g0_Isp_adim(g0_dim, Isp_dim, tau, tau_vec, l_vec, mu_p, mu_s)
        
    return jacobian_u_coeff(g0, Isp)

    # first-order ODEs for STM
    # u_stm = np.zeros((7, 4))
#     u_stm[3:6, 0:3] = np.eye(3)
#     u_stm[-1, -1] = -1/(g0*Isp)
#     
#     return u_stm
    

@jit('Tuple((float64, float64))(float64, float64, float64, float64[::1], float64[::1], float64, float64)', nopython=True, nogil=True, fastmath=True)
def compute_g0_Isp_adim(g0_dim, Isp_dim, tau, tau_vec, l_vec, mu_p, mu_s):
    LU = np.interp(tau, tau_vec, l_vec) # [km]
    TU = np.sqrt(LU**3 / (mu_p + mu_s)) # [s]
    VU = LU / TU # [km/s]
    ACU = VU / TU # Acceleration unit [km/s^2]
    
    g0 = g0_dim / ACU
    Isp = Isp_dim / TU
    
    return g0, Isp


@jit('float64[:,::1](float64, float64)', nopython=True, nogil=True, fastmath=True)
def jacobian_u_coeff(g0, Isp):
    
    # first-order ODEs for STM
    u_stm = np.zeros((7, 4))
    u_stm[3:6, 0:3] = np.eye(3)
    u_stm[-1, -1] = -1/(g0*Isp)
    
    return u_stm