KPL/MK

    This meta kernel defines all SPICE kernels required at runtime.
    Caution: the specified path is relative to the imatpy src/init directory thus a call to the
    spiceypy routine furnsh has to be performed in a module located in this folder.

\begindata

    KERNELS_TO_LOAD = (
                        'kernels/lsk/naif0012.tls'
                        'kernels/pck/pck00010.tpc'
                        'kernels/pck/gm_de431.tpc'
                        'kernels/spk/de430.bsp'
                        )

\begintext
