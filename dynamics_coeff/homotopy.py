import numpy as np
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from dynamics_coeff.rnbp_rpf_dynamics_nonuniform_jit import compute_coeffs as rnbp_jit_coeffs
from dynamics_coeff.rnbp_rpf_dynamics_nonuniform import compute_coeffs as rnbp_coeffs
from dynamics_coeff.crtbp_dynamics import compute_coeff as crtbp_coeffs

# FFT, Piecewise, Polynomial Approximation and Homotopy
def fft_approx_function(x_data, y_fft, n_components):
    # xf = fftfreq(N, x[1] - x[0]) # generates the frequencies corresponding to the output of the FFT, based on number of samples and sample spacing
    yf_simplified = np.copy(y_fft)
    # yf_simplified[n_components:] = 0 # Set higher frequencies to zero
    return np.fft.ifft(yf_simplified).real

def piecewise_linear_approx_function(x_data, y_data, num_segments):
    segment_indices = np.round(np.linspace(0, len(x_data) - 1, num_segments + 1)).astype(int)
    x_segments = x_data[segment_indices]
    y_segments = y_data[segment_indices]
    return np.interp(x_data, x_segments, y_segments)


def polynomial_approx_function(x_data, y_data, polynomial_degree):
    coefficients = np.polyfit(x_data, y_data, polynomial_degree)
    return np.polyval(coefficients, x_data)

def general_homotopy(t, x_data, y_data, method, param, y_data_3bp):
    """
    General homotopy function for different approximation methods.

    Args:
        t: Homotopy parameter (0 <= t <= 1).
        x_data: The x-data.
        y_data: The original y-data.
        method: The approximation method ('fft', 'piecewise', 'polynomial').
        param: The parameter for the chosen method (n_components, num_segments, or polynomial_degree).

    Returns:
        The y-values of the homotopy at time t.
    """
    if method == 'fft':
        y_approx = fft_approx_function(x_data, fft(y_data), param) 
    elif method == 'piecewise':
        y_approx = piecewise_linear_approx_function(x_data, y_data, param)
    elif method == 'polynomial':
        y_approx = polynomial_approx_function(x_data, y_data, param)
    else:
        raise ValueError("Invalid method. Choose 'fft', 'piecewise', or 'polynomial'.")

    return (1 - t) * y_data_3bp + t * y_approx
    

def eval_homotopy_at_point(t, x_data, method, x_point, y_3bp_at_point, f_precomputed): 
    """
    Efficiently evaluates the homotopy at a specific x_data point.
    """

    # index = np.abs(x_data - x_point).argmin() 
    #assume x_data is evenly spaced
    dx = x_data[1] - x_data[0]  # Spacing between points
    index = round((x_point - x_data[0]) / dx)  # Compute nearest index
    index = max(0, min(index, len(x_data) - 1))  # Clamp to valid range
   
    if method == 1: # fft
        y_approx_at_point = np.interp(x_point % (x_data[-1] - x_data[0]), x_data, f_precomputed)
    elif method == 2: # piecewise
        y_approx_at_point = f_precomputed(x_point)
    elif method == 3: # polynomial
        coefficients = f_precomputed
        y_approx_at_point = np.polyval(coefficients, x_point)
    else:
        raise ValueError("Invalid method.")

    return (1 - t) * y_3bp_at_point + t * y_approx_at_point 


def get_homotopy_coefficients(sel_points, homotopy_vals, tau_vec, t_vec, id_primary, id_secondary, mu_bodies, naif_id_bodies, observer_id, reference_frame, epoch_t0, sel_homotopy=1, homotopy_param=10000, use_jit=False):
    '''
    Generate the pre-computed homotopy coefficients
        sel_points: number of points to compute for
        homotopy_vals: weighting for homotopy, as a time vector or a single value between 0 and 1. 0 implies 3bp coefficients, 1 implies nbp
        ... most are conventional, reference_frame must be encoded if use_jit is true
        sel_homotopy: homotopy type (1: FFT; 2: piecewise linear; 3: polynomial)
        homotopy_param: parameter for homotopy fitting (FFT: n_components_FFT; piecewise: num_segments_piecewise; polynomial: poly_degree)
        use_jit: use jit for computing coefficients

    Returns
        tau_linspace: n-length array containing tau values corresponding to coefficients
        y_homotopy: 13xn array containing each coefficient for each tau
    '''
 
    tau_linspace = np.linspace(tau_vec[0], tau_vec[-1], sel_points)
    b1 = np.zeros(sel_points)
    b2 = np.zeros(sel_points)
    b3 = np.zeros(sel_points)
    b4 = np.zeros(sel_points)
    b5 = np.zeros(sel_points)
    b6 = np.zeros(sel_points)
    b7 = np.zeros(sel_points)
    b8 = np.zeros(sel_points)
    b9 = np.zeros(sel_points)
    b10 = np.zeros(sel_points)
    b11 = np.zeros(sel_points)
    b12 = np.zeros(sel_points)
    b13 = np.zeros(sel_points)

    for i in range(len(tau_linspace)):
        if use_jit:
            b1[i], b2[i], b3[i], b4[i], b5[i], b6[i], b7[i], b8[i], b9[i], b10[i], b11[i], b12[i], b13[i] = rnbp_jit_coeffs(tau_linspace[i], id_primary, id_secondary, mu_bodies, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec)
        else:
            b1[i], b2[i], b3[i], b4[i], b5[i], b6[i], b7[i], b8[i], b9[i], b10[i], b11[i], b12[i], b13[i] = rnbp_coeffs(tau_linspace[i], id_primary, id_secondary, mu_bodies, naif_id_bodies, observer_id, reference_frame, epoch_t0, tau_vec, t_vec)
    # Define 3BP coefficients
    b_3bp = crtbp_coeffs()
        
    
    # Compute homotopy
    y = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13]
    y_homotopy = np.zeros((len(y), len(tau_linspace)))
    
    for i in range(len(y)):
        if sel_homotopy == 1:
            y_homotopy[i,:] = general_homotopy(homotopy_vals, tau_linspace, y[i], 'fft', homotopy_param, b_3bp[i])
        elif sel_homotopy == 2:
            y_homotopy[i,:] = general_homotopy(homotopy_vals, tau_linspace, y[i], 'piecewise', homotopy_param, b_3bp[i])
        elif sel_homotopy == 3:
            y_homotopy[i,:] = general_homotopy(homotopy_vals, tau_linspace, y[i], 'polynomial', homotopy_param, b_3bp[i])

    return tau_linspace, y_homotopy

        