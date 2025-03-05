import numpy as np


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


    