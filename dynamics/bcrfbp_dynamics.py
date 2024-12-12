import numpy as np
import numba

def dynamics(t, state, control, p, aux):
    
    mu_s = aux['param']['mu_s']
    mu = aux['param']['mu']
    om_sun = aux['param']['om_sun']
    sun_angle_t0 = aux['param']['sun_angle_t0']
    a_sun = aux['param']['a_sun']

    x, y, z, xdot, ydot, zdot = state

    dx = np.zeros(6)

    sun_angle = om_sun * t + sun_angle_t0
    rs = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0])

    # Com_sunpute the relative positions
    re_sc = np.array([x + mu, y, z])
    rm_sc = np.array([x - 1 + mu, y, z])
    rs_sc = np.array([x - rs[0], y - rs[1], z - rs[2]])

    # Com_sunpute the distances cubed
    re_sc3 = np.dot(re_sc, re_sc)**1.5
    rm_sc3 = np.dot(rm_sc, rm_sc)**1.5
    rs_sc3 = np.dot(rs_sc, rs_sc)**1.5
    
    a_sun3 = a_sun**3

    # Compute the accelerations
    xddot = (2 * ydot + x 
             - (1 - mu) * (x + mu) / re_sc3 
             - mu * (x - 1 + mu) / rm_sc3 
             - mu_s * (x - rs[0]) / rs_sc3 
             - mu_s * rs[0] / a_sun3)

    yddot = (- 2*xdot + y
             - (1 - mu) * y / re_sc3 
             - mu * y / rm_sc3 
             - mu_s * (y - rs[1]) / rs_sc3 
             - mu_s * rs[1] / a_sun3)

    zddot = (- (1 - mu) * z / re_sc3 
             - mu * z / rm_sc3 
             - mu_s * (z - rs[2]) / rs_sc3 
             - mu_s * rs[2] / a_sun3)

    # Fill in the derivatives
    dx[0] = xdot
    dx[1] = ydot
    dx[2] = zdot
    dx[3] = xddot
    dx[4] = yddot
    dx[5] = zddot

    return dx

@numba.njit
def dynamics_numba(t, state, control, p, mu_s, mu, om_sun, sun_angle_t0, a_sun):
    x, y, z, xdot, ydot, zdot = state

    dx = np.zeros(6)

    sun_angle = om_sun * t + sun_angle_t0
    rs = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0.0])


    
    # Com_sunpute the relative positions
    re_sc = np.array([x + mu, y, z])
    rm_sc = np.array([x - 1.0 + mu, y, z])
    rs_sc = np.array([x - rs[0], y - rs[1], z - rs[2]])

    # Com_sunpute the distances cubed
    re_sc3 = np.dot(re_sc, re_sc)**1.5
    rm_sc3 = np.dot(rm_sc, rm_sc)**1.5
    rs_sc3 = np.dot(rs_sc, rs_sc)**1.5
    
    a_sun3 = a_sun**3

    # Compute the accelerations
    xddot = (2 * ydot + x 
             - (1 - mu) * (x + mu) / re_sc3 
             - mu * (x - 1 + mu) / rm_sc3 
             - mu_s * (x - rs[0]) / rs_sc3 
             - mu_s * rs[0] / a_sun3)

    yddot = (- 2*xdot + y
             - (1 - mu) * y / re_sc3 
             - mu * y / rm_sc3 
             - mu_s * (y - rs[1]) / rs_sc3 
             - mu_s * rs[1] / a_sun3)

    zddot = (- (1 - mu) * z / re_sc3 
             - mu * z / rm_sc3 
             - mu_s * (z - rs[2]) / rs_sc3 
             - mu_s * rs[2] / a_sun3)

    # Fill in the derivatives
    dx[0] = xdot
    dx[1] = ydot
    dx[2] = zdot
    dx[3] = xddot
    dx[4] = yddot
    dx[5] = zddot

    return dx
   

def jacobian_x(t, state, control, p, aux):
    
    mu_s = aux['param']['mu_s']
    mu = aux['param']['mu']
    om_sun = aux['param']['om_sun']
    sun_angle_t0 = aux['param']['sun_angle_t0']
    a_sun = aux['param']['a_sun']

    x, y, z, xdot, ydot, zdot = state

    sun_angle = om_sun * t + sun_angle_t0
    rs = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0])
    rs0, rs1, rs2 = rs
    
    # Com_sunpute the relative positions
    re_sc = np.array([x + mu, y, z])
    rm_sc = np.array([x - 1 + mu, y, z])
    rs_sc = np.array([x - rs0, y - rs1, z])

    # Com_sunpute the distances cubed
    re_sc3 = np.dot(re_sc, re_sc)**1.5
    rm_sc3 = np.dot(rm_sc, rm_sc)**1.5
    rs_sc3 = np.dot(rs_sc, rs_sc)**1.5
    
    re_sc5 = np.dot(re_sc, re_sc)**2.5
    rm_sc5 = np.dot(rm_sc, rm_sc)**2.5
    rs_sc5 = np.dot(rs_sc, rs_sc)**2.5
        
    jac_x = np.array([
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1], 
        [-3.0*mu*(mu + x)**2/re_sc5 + mu/re_sc3 + 3.0*mu*(mu + x - 1)**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*(rs0 - x)**2/rs_sc5 - mu_s/rs_sc3 + 3.0*(mu + x)**2/re_sc5 - 1/re_sc3 + 1, 3.0*(mu*re_sc5*rs_sc5*y*(mu + x - 1) + mu_s*re_sc5*rm_sc5*(rs0 - x)*(rs1 - y) - rm_sc5*rs_sc5*y*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 3.0*z*(mu*re_sc5*rs_sc5*(mu + x - 1) - mu_s*re_sc5*rm_sc5*(rs0 - x) - rm_sc5*rs_sc5*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 0, 2, 0], 
        [3.0*(mu*re_sc5*rs_sc5*y*(mu + x - 1) + mu_s*re_sc5*rm_sc5*(rs0 - x)*(rs1 - y) - rm_sc5*rs_sc5*y*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), -3.0*mu*y**2/re_sc5 + mu/re_sc3 + 3.0*mu*y**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*(rs1 - y)**2/rs_sc5 - mu_s/rs_sc3 + 3.0*y**2/re_sc5 - 1/re_sc3 + 1, 3.0*z*(mu*re_sc5*rs_sc5*y + mu_s*re_sc5*rm_sc5*(-rs1 + y) + rm_sc5*rs_sc5*y*(1 - mu))/(re_sc5*rm_sc5*rs_sc5), -2, 0, 0], 
        [3.0*z*(mu*re_sc5*rs_sc5*(mu + x - 1) - mu_s*re_sc5*rm_sc5*(rs0 - x) - rm_sc5*rs_sc5*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 3.0*z*(mu*re_sc5*rs_sc5*y + mu_s*re_sc5*rm_sc5*(-rs1 + y) + rm_sc5*rs_sc5*y*(1 - mu))/(re_sc5*rm_sc5*rs_sc5), -3.0*mu*z**2/re_sc5 + mu/re_sc3 + 3.0*mu*z**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*z**2/rs_sc5 - mu_s/rs_sc3 + 3.0*z**2/re_sc5 - 1/re_sc3, 0, 0, 0]
        ])

    return jac_x

@numba.njit
def jacobian_x_numba(t, state, control, p, mu_s, mu, om_sun, sun_angle_t0, a_sun):

    x, y, z, xdot, ydot, zdot = state


    sun_angle = om_sun * t + sun_angle_t0
    rs = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0.0])
    rs0, rs1, rs2 = rs
    
    # Com_sunpute the relative positions
    re_sc = np.array([x + mu, y, z])
    rm_sc = np.array([x - 1.0 + mu, y, z])
    rs_sc = np.array([x - rs0, y - rs1, z])

    # Com_sunpute the distances cubed
    re_sc3 = np.dot(re_sc, re_sc)**1.5
    rm_sc3 = np.dot(rm_sc, rm_sc)**1.5
    rs_sc3 = np.dot(rs_sc, rs_sc)**1.5
    
    re_sc5 = np.dot(re_sc, re_sc)**2.5
    rm_sc5 = np.dot(rm_sc, rm_sc)**2.5
    rs_sc5 = np.dot(rs_sc, rs_sc)**2.5
        
    jac_x = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
        [-3.0*mu*(mu + x)**2/re_sc5 + mu/re_sc3 + 3.0*mu*(mu + x - 1)**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*(rs0 - x)**2/rs_sc5 - mu_s/rs_sc3 + 3.0*(mu + x)**2/re_sc5 - 1/re_sc3 + 1, 3.0*(mu*re_sc5*rs_sc5*y*(mu + x - 1) + mu_s*re_sc5*rm_sc5*(rs0 - x)*(rs1 - y) - rm_sc5*rs_sc5*y*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 3.0*z*(mu*re_sc5*rs_sc5*(mu + x - 1) - mu_s*re_sc5*rm_sc5*(rs0 - x) - rm_sc5*rs_sc5*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 0, 2, 0], 
        [3.0*(mu*re_sc5*rs_sc5*y*(mu + x - 1) + mu_s*re_sc5*rm_sc5*(rs0 - x)*(rs1 - y) - rm_sc5*rs_sc5*y*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), -3.0*mu*y**2/re_sc5 + mu/re_sc3 + 3.0*mu*y**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*(rs1 - y)**2/rs_sc5 - mu_s/rs_sc3 + 3.0*y**2/re_sc5 - 1/re_sc3 + 1, 3.0*z*(mu*re_sc5*rs_sc5*y + mu_s*re_sc5*rm_sc5*(-rs1 + y) + rm_sc5*rs_sc5*y*(1 - mu))/(re_sc5*rm_sc5*rs_sc5), -2, 0, 0], 
        [3.0*z*(mu*re_sc5*rs_sc5*(mu + x - 1) - mu_s*re_sc5*rm_sc5*(rs0 - x) - rm_sc5*rs_sc5*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 3.0*z*(mu*re_sc5*rs_sc5*y + mu_s*re_sc5*rm_sc5*(-rs1 + y) + rm_sc5*rs_sc5*y*(1 - mu))/(re_sc5*rm_sc5*rs_sc5), -3.0*mu*z**2/re_sc5 + mu/re_sc3 + 3.0*mu*z**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*z**2/rs_sc5 - mu_s/rs_sc3 + 3.0*z**2/re_sc5 - 1/re_sc3, 0, 0, 0]
        ])

    return jac_x

def dynamics_stm(t, state, control, p, aux):
    mu_s = aux['param']['mu_s']
    mu = aux['param']['mu']
    om_sun = aux['param']['om_sun']
    sun_angle_t0 = aux['param']['sun_angle_t0']
    a_sun = aux['param']['a_sun']

    x, y, z, xdot, ydot, zdot = state

    state_dot = np.zeros(6)

    sun_angle = om_sun * t + sun_angle_t0
    rs = a_sun * np.array([np.cos(sun_angle), np.sin(sun_angle), 0])
    rs0, rs1, rs2 = rs

    # Com_sunpute the relative positions
    re_sc = np.array([x + mu, y, z])
    rm_sc = np.array([x - 1 + mu, y, z])
    rs_sc = np.array([x - rs0, y - rs1, z])

    # Com_sunpute the distances cubed
    re_sc3 = np.dot(re_sc, re_sc)**1.5
    rm_sc3 = np.dot(rm_sc, rm_sc)**1.5
    rs_sc3 = np.dot(rs_sc, rs_sc)**1.5
    
    a_sun2 = a_sun**2

    # Compute the accelerations
    xddot = (2 * ydot + x 
             - (1 - mu) * (x + mu) / re_sc3 
             - mu * (x - 1 + mu) / rm_sc3 
             - mu_s * (x - rs0) / rs_sc3 
             - mu_s * np.cos(sun_angle) / a_sun2)

    yddot = (- 2*xdot + y
             - (1 - mu) * y / re_sc3 
             - mu * y / rm_sc3 
             - mu_s * (y - rs[1]) / rs_sc3 
             - mu_s * np.sin(sun_angle) / a_sun2)

    zddot = (- (1 - mu) * z / re_sc3 
             - mu * z / rm_sc3 
             - mu_s * (z - rs[2]) / rs_sc3)

    # Fill in the derivatives
    state_dot[0] = xdot
    state_dot[1] = ydot
    state_dot[2] = zdot
    state_dot[3] = xddot
    state_dot[4] = yddot
    state_dot[5] = zddot
    
    re_sc5 = np.dot(re_sc, re_sc)**2.5
    rm_sc5 = np.dot(rm_sc, rm_sc)**2.5
    rs_sc5 = np.dot(rs_sc, rs_sc)**2.5
        
    jac_x = np.array([
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1], 
        [-3.0*mu*(mu + x)**2/re_sc5 + mu/re_sc3 + 3.0*mu*(mu + x - 1)**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*(rs0 - x)**2/rs_sc5 - mu_s/rs_sc3 + 3.0*(mu + x)**2/re_sc5 - 1/re_sc3 + 1, 3.0*(mu*re_sc5*rs_sc5*y*(mu + x - 1) + mu_s*re_sc5*rm_sc5*(rs0 - x)*(rs1 - y) - rm_sc5*rs_sc5*y*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 3.0*z*(mu*re_sc5*rs_sc5*(mu + x - 1) - mu_s*re_sc5*rm_sc5*(rs0 - x) - rm_sc5*rs_sc5*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 0, 2, 0], 
        [3.0*(mu*re_sc5*rs_sc5*y*(mu + x - 1) + mu_s*re_sc5*rm_sc5*(rs0 - x)*(rs1 - y) - rm_sc5*rs_sc5*y*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), -3.0*mu*y**2/re_sc5 + mu/re_sc3 + 3.0*mu*y**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*(rs1 - y)**2/rs_sc5 - mu_s/rs_sc3 + 3.0*y**2/re_sc5 - 1/re_sc3 + 1, 3.0*z*(mu*re_sc5*rs_sc5*y + mu_s*re_sc5*rm_sc5*(-rs1 + y) + rm_sc5*rs_sc5*y*(1 - mu))/(re_sc5*rm_sc5*rs_sc5), -2, 0, 0], 
        [3.0*z*(mu*re_sc5*rs_sc5*(mu + x - 1) - mu_s*re_sc5*rm_sc5*(rs0 - x) - rm_sc5*rs_sc5*(mu - 1)*(mu + x))/(re_sc5*rm_sc5*rs_sc5), 3.0*z*(mu*re_sc5*rs_sc5*y + mu_s*re_sc5*rm_sc5*(-rs1 + y) + rm_sc5*rs_sc5*y*(1 - mu))/(re_sc5*rm_sc5*rs_sc5), -3.0*mu*z**2/re_sc5 + mu/re_sc3 + 3.0*mu*z**2/rm_sc5 - mu/rm_sc3 + 3.0*mu_s*z**2/rs_sc5 - mu_s/rs_sc3 + 3.0*z**2/re_sc5 - 1/re_sc3, 0, 0, 0]
        ])
    
    return np.concatenate((state_dot, jac_x.flatten()))

