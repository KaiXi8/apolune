import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from plotting import plot_Lagrange_points
from propagation import threebody

def sort_eig(eigvals, **kwargs):
    """
    Use: Sorts eigenvalues of 6x6 matrix when they occur in reciprocal pairs

    Inputs:
    - eigvals: Array of the eigenvalues

    Outputs:
    - idx_array_out: The indices corresponding to the pairs, 3x2 array
    - eig_array_out: The eigenvalues, paired. Possibly sorted, 3x2 array
    - stab_array_out: The stability indices. Row-ordering matches that of eigenvalues, 3x2 array
    """
    sort_spec = kwargs.get('sort_spec', 'off')

    eig_array = np.zeros((3, 2), dtype=complex)
    idx_array = np.zeros((3, 2), dtype=int)
    test_mat = np.real(np.outer(eigvals, eigvals))
    
    for row in range(6):
        test_mat[row, row] = 0.0

    counter = 0
    mybin = [0, 1, 2, 3, 4, 5]
    while counter < 3:
        row = np.min(mybin)
        match = np.argmin(np.abs(test_mat[row, :] - 1.))
        mybin.remove(row)
        mybin.remove(match)
        idx_array[counter, :] = [row, match]
        eig_array[counter, :] = [eigvals[row], eigvals[match]]
        counter += 1

    eig_array_out = eig_array.copy()
    idx_array_out = idx_array.copy()
    
    if sort_spec == 'triv_top':
        # Move the trivial pair to the top:
        triv_idx = np.argmin(np.abs(np.array([np.real(eig_array[i, 0]) for i in range(3)]) - 1.))
        if triv_idx != 0:
            eig_array_out[[0, triv_idx], :] = eig_array[[triv_idx, 0], :]
            idx_array_out[[0, triv_idx], :] = idx_array[[triv_idx, 0], :]
            eig_array_out[triv_idx, :] = eig_array[0, :]
            idx_array_out[triv_idx, :] = idx_array[0, :]
        if np.imag(np.array([eig_array_out[i, 0] + eig_array_out[i, 1] for i in range(3)])).any() > 1.e-5:
            print('In sort_eig, unexpected imaginary residual...')
    stab_array_out = np.real(np.array([eig_array_out[i, 0] + eig_array_out[i, 1] for i in range(3)]))

    return eig_array_out, idx_array_out, stab_array_out

def plot_manifolds(sol, data, mu):
    filaments = 60

    # Monodromy matrix
    bodies = data['system']['name'].split('-') # names of primaries
    phi = np.reshape(sol[6:42,-1], (6,6), order='F') # Monodromy matrix
    eigval, eigvec = np.linalg.eig(phi) 
    eigval_new, idx, stability = sort_eig(eigval, sort_spec = 'triv_top') # Sort the eigenvalues by stability
    
    fig3 = plt.figure()
    ax3 = plt.axes()
    ax3.set_xlabel('x (nd)')
    ax3.set_ylabel('y (nd)')
    ax3.plot(sol[0], sol[1], 'k--', linewidth=1, zorder = 2)
    y_f = 384400 / 149597871 # Nd Moon radius
    circle2 = plt.Circle((1-mu, 0), y_f, color='k', fill=False)
    ax3.add_artist(circle2)


    #plt.annotate(bodies[0], (-mu,0), ha='right'); plt.plot(-mu,0,'ko') # Larger body
    plt.annotate(bodies[1], (1-mu,0), ha='left'); plt.plot(1 - mu,0,'ko') # Smaller body
    plt.title(f'Invariant Manifolds in the {bodies[0]}-{bodies[1]} CR3BP system')
    plot_Lagrange_points(ax3, data, 'L1 ')

    orbit = lambda t, state : threebody(t, state, mu)

    # Stopping condition functions
    U1_down = lambda t,y: y[1] if y[0]<0 else True; U1_down.terminal = True; U1_down.direction = -1
    U1_up = lambda t,y: y[1] if y[0]<0 else True; U1_up.terminal = True; U1_up.direction = 1
    U1 = lambda t,y: y[1] if y[0]<0 else True; U1.terminal = True
    P2 = lambda t,y: (1-mu) - y[0]; P2.terminal = True#; P2.direction = -1 # Stop at Primary 2 x
    P1 = lambda t,y: (-mu) - y[0]; P1.terminal = True # Stop at Primary 1 x

    prop_time = 30
    flags = [1, 0, 1, 0] # Which manifold branches to plot
    eps = 6.e-8  # 6.e-5, 2.e-8, 5.e-8

    eigvec_real = np.real(eigvec)
    for ind in [int(i * max(np.shape(sol))/filaments) for i in range(filaments)]: 
        # Multiply by the STM to obtain the correct Eigenvectors for the state (KoLoMaRo 4.4.3)
        def perturbation(sol, ind, idx): 
            stable_eigvec_real = eigvec_real[:,idx]/np.linalg.norm(eigvec_real[0:3, idx])
            dev_s_temp = np.matmul(np.reshape(sol[6:42, ind], (6, 6), order='F'), stable_eigvec_real)
            return dev_s_temp / np.linalg.norm(dev_s_temp[0:3])

        # Stable
        if flags[0]:
            y0 = sol[0:6, ind] + eps*perturbation(sol,ind,idx[1,1])
            manifold = solve_ivp(orbit, [0, -prop_time], y0, method='DOP853', rtol=1e-13, atol=1e-14, events=P2)
            ax3.plot(manifold.y[0], manifold.y[1], 'blue', linewidth = 0.6, zorder = 1)
            
        if flags[1]:
            y0 = sol[0:6, ind] - eps*perturbation(sol,ind,idx[1,1])
            manifold = solve_ivp(orbit, [0, -prop_time], y0, method='DOP853', rtol=1e-13, atol=1e-14, events=P1)
            ax3.plot(manifold.y[0], manifold.y[1], 'blue', linewidth = 0.6, zorder = 1)

        # Unstable
        if flags[2]:
            y0 = sol[0:6, ind] + eps*perturbation(sol,ind,idx[1,0])
            manifold = solve_ivp(orbit, [0, prop_time], y0, method='DOP853', rtol=1e-13, atol=1e-14, events=P2)            
            ax3.plot(manifold.y[0], manifold.y[1], 'red', linewidth = 0.6, zorder = 1)

        if flags[3]:
            y0 = sol[0:6, ind] - eps*perturbation(sol,ind,idx[1,0])
            manifold = solve_ivp(orbit, [0, prop_time], y0, method='DOP853', rtol=1e-13, atol=1e-14, events=P1)
            ax3.plot(manifold.y[0], manifold.y[1], 'red', linewidth = 0.6, zorder = 1)

 