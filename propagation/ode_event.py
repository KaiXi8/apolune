import numpy as np

    

# Define the plane crossing event function
def plane_crossing_event(plane='xz', direction=0, isterminal=1):
    """
    Event function for detecting crossing of specified plane in CR3BP.

    Parameters:
    - t : float : Time
    - plane : str : Plane to detect crossing ('yz', 'xz', 'xy')
    - direction : int : Direction of crossing (-1, 0, 1)
    - isterminal : int : If 1, terminate integration at this event

    Returns:
    - value : float : Value of the event function (plane crossing)
    - isterminal : int : Termination flag (1 if event terminates the integration)
    - direction : int : Direction of the event crossing (-1, 0, 1)
    """
    if plane.lower() == 'yz':
        state_component_index = 0  # x position
    elif plane.lower() == 'xz':
        state_component_index = 1  # y position
    elif plane.lower() == 'xy':
        state_component_index = 2  # z position
    else:
        raise ValueError("Invalid plane. Choose from 'yz', 'xz', or 'xy'.")

    if direction not in [-1, 0, 1]:
        raise ValueError("Invalid direction. Choose -1, 0, or 1.")

    if isterminal not in [0, 1]:
        raise ValueError("Invalid isterminal value. Choose 0 or 1.")


    def fun(_, state, a, b, c):
        """Event function. """
        return state[state_component_index]

    fun.direction = direction
    fun.terminal = bool(isterminal)
    
    return fun


