"""
Load the SPICE kernels defined in the meta-kernel `kernel.tm`.

@author: Alberto FOSSA'
"""

import os
from spiceypy import furnsh, ktotal

DIRNAME = os.path.dirname(__file__)
KERNELS_ROOT = os.path.join(DIRNAME, 'kernels')


KERNELS_TO_LOAD = (
    'lsk/naif0012.tls',
    'pck/pck00010.tpc',
    'pck/gm_de431.tpc',
    'spk/de430.bsp',
    'spk/L2_de431.bsp',
    'spk/20003671.bsp',
)  # kernels for N-body ephemeris model with point masses
KERNELS_TO_LOAD_MOON_POT = (
    'fk/moon_080317.tf',
    'lsk/naif0012.tls',
    'pck/pck00010.tpc',
    'pck/moon_pa_de421_1900-2050.bpc',
    'pck/gm_de431.tpc',
    'spk/de421.bsp',
)  # kernels for N-body ephemeris model with point masses and Moon gravitational potential
ASTEROID_KERNELS_TO_LOAD = (
    'asteroids/2000SG344_3054374.15',
    'asteroids/2010UE51_3550232.15',
    'asteroids/2011MD_3568303.15',
    'asteroids/2012UV136_2478784.15',
    'asteroids/2014YD_3702319.15',
)  # kernels for N-body ephemeris model with point masses
RPF_VALIDATION_KERNELS_TO_LOAD = (
    'rpf_validation/20000001.bsp',
    'rpf_validation/20000019.bsp',
    'rpf_validation/20000121.bsp',
    'rpf_validation/20000624.bsp',
    'rpf_validation/20001221.bsp',
    'rpf_validation/20001862.bsp',
    'rpf_validation/20002001.bsp',
    'rpf_validation/20002062.bsp',
    'rpf_validation/20005261.bsp',
    'rpf_validation/20005335.bsp',
    'rpf_validation/20019521.bsp',
    'rpf_validation/20163693.bsp',
) 

def load_kernels():
    """Load SPICE kernels. """
    if ktotal('all') == 0:
        for k in KERNELS_TO_LOAD:
            furnsh(os.path.join(KERNELS_ROOT, k))


def load_kernels_moon_pot():
    """Load SPICE kernels. """
    if ktotal('all') == 0:
        for k in KERNELS_TO_LOAD_MOON_POT:
            furnsh(os.path.join(KERNELS_ROOT, k))


def load_asteroid_kernels():
    """Load SPICE kernels. """
    if ktotal('all') == 0:
        for k in ASTEROID_KERNELS_TO_LOAD:
            furnsh(os.path.join(KERNELS_ROOT, k))
       
            
def load_rpf_validation_kernels():
    """Load SPICE kernels. """
#     if ktotal('all') == 0:
    for k in RPF_VALIDATION_KERNELS_TO_LOAD:
        furnsh(os.path.join(KERNELS_ROOT, k))