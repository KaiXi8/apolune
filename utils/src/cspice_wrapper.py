from cffi import FFI

# define library paths where CSPICE is installed
# in our case, we will make use of the ones from the spiceypy package
# on my mac with miniforge and the conda environment env-cspice, the paths look like this:
LIB_DIR = "/Users/hofmannc/miniforge3/envs/env-cspice/lib"  # Directory containing libcspice.so (linux), libcspice.dll (windows), or libcspice.dylib (macos) 
INC_DIR = "/Users/hofmannc/miniforge3/envs/env-cspice/include/cspice"  # Directory containing SpiceUsr.h and all the header files, probably located in /.../include/cspice

# Initialize FFI
ffibuilder = FFI()

# Declare the C functions you want to wrap
ffibuilder.cdef("""
    void spkez_c(int targ, double et, const char * ref, const char * abcorr, int obs,
                 double starg[6], double * lt);
    void spkezp_c(int targ, double et, const char * ref, const char * abcorr, int obs,
                  double ptarg[3], double * lt);
    void spkezr_c(const char * targ, double et, const char * ref, const char * abcorr,
                  const char * obs, double starg[6], double * lt);
    void spkgeo_c(int targ, double et, const char * ref, int obs,
                  double state[6], double * lt);
    void spkgps_c(int targ, double et, const char * ref, int obs,
                  double pos[3], double * lt);
    void spkpos_c(const char * targ, double et, const char * ref, const char * abcorr,
                  const char * obs, double ptarg[3], double * lt);
    void pxform_c(const char * from, const char * to, double et, double rotate[3][3]);
""")

# Define the source file for the wrapper
ffibuilder.set_source("cspice_wrapper",
                      """
    #include <stdio.h>
    #include "SpiceUsr.h"
                      """,
                      libraries=["cspice", "m"],
                      # this depends on the OS and the compiler you use! i have macos and use clang. if you use some gcc compiler (e.g. on linux), this must be changed, e.g. to:
                      # extra_link_args=['-Wl,--rpath=' + LIB_DIR]
                      extra_link_args=['-Wl,-rpath,' + LIB_DIR],  
                      library_dirs=[LIB_DIR],
                      include_dirs=[INC_DIR],
                      )

if __name__ == "__main__":
    # Compile the wrapper
    ffibuilder.compile(verbose=True)
