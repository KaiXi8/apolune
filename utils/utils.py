import numpy as np
import spiceypy as spice


# transforms the string name of a frame (e.g., "J2000") into an int array that can be used within numba/jit
def frame_encoder(frame):
    if isinstance(frame, str):  # single frame name to encode
        return np.array([ord(c) for c in frame + '\0'], dtype=np.int8)
    if isinstance(frame, (tuple, list)):  # multiple frame names to encode
        nb_chars = [len(f) for f in frame]  # number of characters for each frame name
        nb_cols = max(nb_chars) + 1  # number of columns in the output array
        padded_frame = [f + '\0' * (nb_cols - nb_chars[i]) for i, f in enumerate(frame)]
        encoded_frame = np.asarray([ord(c) for c in ''.join(padded_frame)], dtype=np.int8)
        return encoded_frame.reshape(len(frame), max(nb_chars) + 1)
    raise Exception('frame must be a string or a tuple of strings representing valid frame names')


def compute_perturbing_accelerations(epoch_time, r_osc, mu_bodies, naif_id_bodies, id_primary, reference_frame, observer_id):

    num_bodies = len(mu_bodies)

#     observer_id = 0
    
    # retrieve states of celestial bodies from SPICE and compute perturbing accelerations
    acc = np.zeros((num_bodies, 3))
    for i in range(num_bodies):
        if i == id_primary:
            acc[i] = - mu_bodies[i] * r_osc / np.linalg.norm(r_osc)**3
        else:
            pos_body, _ = spice.spkgps(naif_id_bodies[i], epoch_time, reference_frame, observer_id)
            r_bsc = r_osc - pos_body # position of spacecraft wrt to body
            acc[i] = - mu_bodies[i] * (r_bsc / np.linalg.norm(r_bsc)**3 - pos_body / np.linalg.norm(pos_body)**3)
        
    acc_mag = np.linalg.norm(acc, axis=1)
#     print("acc: ", acc)
#     print("acc_mag: ", acc_mag)
    return acc_mag, acc
    