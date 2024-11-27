import numpy as np


def reshapeTo1d(x, u, p):
    return np.concatenate((x.flatten(), u.flatten(), p.flatten()))


def reshapeToNdim(sol, aux):
    n_x = aux['problem']['n_x']
    n_u = aux['problem']['n_u']
    n_p = aux['problem']['n_p']
    N = aux['problem']['N']
    n_man = aux['param']['n_man']
    xind = aux['indices']['x_ind']
    uind = aux['indices']['u_ind']
    pind = aux['indices']['p_ind']
     
    x1d = sol[xind]
    u1d = sol[uind]
    p1d = sol[pind]
    
    x = x1d.reshape((N, n_x))
    u = u1d.reshape((n_man, n_u))

    return x, u, p1d

