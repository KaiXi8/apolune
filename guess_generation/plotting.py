import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import propagation

def plot_Lagrange_points(ax, data, points):
    if ax.name == '3d':
        if '1' in points: ax.plot(float(data['system']['L1'][0]), 0, 'ko'); ax.text(float(data['system']['L1'][0]), 0, 0, 'L1')
        if '2' in points: ax.plot(float(data['system']['L2'][0]), 0, 'ko'); ax.text(float(data['system']['L2'][0]), 0, 0, 'L2')
        if '3' in points: ax.plot(float(data['system']['L3'][0]), 0, 'ko'); ax.text(float(data['system']['L3'][0]), 0, 0, 'L3')
        if '4' in points: L4 = [float(val) for val in data['system']['L4']]; ax.plot(L4[0], L4[1], 'ko'); ax.text(L4[0], L4[1], 0, 'L4')
        if '5' in points: L5 = [float(val) for val in data['system']['L5']]; ax.plot(L5[0], L5[1], 'ko'); ax.text(L5[0], L5[1], 0, 'L5')
        print(float(data['system']['L1'][0]))
        print(float(data['system']['L2'][0]))
    else:
        if '1' in points: ax.plot(float(data['system']['L1'][0]), 0, 'ko'); ax.text(float(data['system']['L1'][0]), 0, 'L1')
        if '2' in points: ax.plot(float(data['system']['L2'][0]), 0, 'ko'); ax.text(float(data['system']['L2'][0]), 0, 'L2')
        if '3' in points: ax.plot(float(data['system']['L3'][0]), 0, 'ko'); ax.text(float(data['system']['L3'][0]), 0, 'L3')
        if '4' in points: L4 = [float(val) for val in data['system']['L4']]; ax.plot(L4[0], L4[1], 'ko'); ax.text(L4[0], L4[1], 'L4')
        if '5' in points: L5 = [float(val) for val in data['system']['L5']]; ax.plot(L5[0], L5[1], 'ko'); ax.text(L5[0], L5[1], 'L5')

def plot_orbit(ind, sol, data, mu):
    """
    Plot the orbit of the specified initial conditions
    Args:
         ind: index of orbit in dataset
        sol: array of states
        data: original parameters
        mu: Mass-ratio

    Returns:
    
    """
    if ind == 0 or True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x (nd)')
        ax.set_ylabel('y (nd)')
        ax.set_zlabel('z (nd)')
        system = data['system']['name']
        if data['family'] in ['lyapunov', 'halo', 'longp', 'short', 'axial', 'vertical']:
            ax.set_title('3D ' + system + ' ' + data['family'] + ' orbit about L' + data['libration_point'])
        else:
            ax.set_title('3D ' + system + ' ' + data['family'] + ' orbit')
        bodies = data['system']['name'].split('-')

        #ax.text(-mu, 0, 0, bodies[0].capitalize()); ax.plot3D(-mu,0,0,'ko') # Larger body
        ax.text(1-mu, 0, 0, bodies[1].capitalize()); ax.plot3D(1 - mu,0,0,'ko') # Smaller body
        plot_Lagrange_points(ax, data, '12')

        ax.plot3D(sol.y[0], sol.y[1], sol.y[2], 'blue', linewidth=1)
    
    # 2D view on xy-plane
    if ind == 0 or True:
        fig2 = plt.figure()
        ax2 = plt.axes()
        ax2.set_xlabel('x (nd)')
        ax2.set_ylabel('y (nd)')
        ax2.set_title('xy Plane View of ' + system + ' ' + data['family'] + ' orbit')
        #plt.annotate(bodies[0], (-mu,0), ha='right'); plt.plot(-mu,0,'ko') # Larger body
        plt.annotate(bodies[1], (1-mu,0), ha='left'); plt.plot(1 - mu,0,'ko') # Smaller body

        plot_Lagrange_points(ax2, data, '1')

    ax2.plot(sol.y[0], sol.y[1], 'blue', linewidth=1)
        
    plt.show()

def plot_jacobi(ind, sol, data, nT, mu):
    """
    Plot the Jacobi constant for a state propagation
    Args:
        ind: index of orbit in dataset
        sol: array of states
        data: original parameters
        nT: number of periods propagated
        mu: Mass-ratio

    Returns:
        
    """
    # Find Jacobi constant at every step
    J0 = float(data['data'][ind][6])
    J_change = propagation.jacobi(sol.y, mu)

    fig3 = plt.figure()
    ax3 = plt.axes()
    period = float(data['data'][ind][7]) # Non-dim
    TU = float(data['system']['tunit']) # Time factor in seconds
    LU = float(data['system']['lunit']) # Length factor in km
    
    for i in range(nT+1):
        plt.axvline(x=(i*period*TU/86400), color='silver', linestyle='-') # Each period
    plt.axhline(y=J_change[0], color='r', linestyle='-', zorder = 1) # If no API pull, use J_change[0] instead of J0
    plt.axhline(y=J_change[0] - 5E-15, color='dimgray', linestyle='--') # Tolerance line
    plt.axhline(y=J_change[0] + 5E-15, color='dimgray', linestyle='--')
    ax3.plot(sol.t*TU/86400, J_change, zorder = 2)
    
    plt.xlim(0, 1.1*sol.t[-1]*TU/86400)
    plt.ylim(J_change[0] - 1.5E-14, J_change[1] + 1.5E-14)
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('C')
    ax3.set_title('Jacobi Constant Fluctuation')
    plt.show() 
