'''
io_utils.py

Everything that touches disk: opening the GALFORM / MAGNETIZER hdf5 files,
pulling out the raw arrays we need, building & applying the selection mask,
and writing the final per-galaxy property table back out to disk.

Note: get_galf_z_output_list, find_tot_volume and find_nearest live in the
external `utils` module (unchanged) and are simply re-used here.
'''
import h5py
import numpy as np

from utils import get_galf_z_output_list, find_tot_volume, find_nearest
import params


def open_magnetizer_file(fno):
    '''Open the MAGNETIZER output hdf5 file for a given sub-volume number.'''
    fname = (params.magnetizer_op_dir + params.galform_model + "_" +
             params.mag_model + "_{0}_corr.hdf5".format(fno))
    Bfile = h5py.File(fname, "r")
    print("Magnetizer prop names ->", Bfile["Output/"].keys())
    return Bfile


def open_galform_file(fno):
    '''Open the GALFORM output hdf5 file for a given sub-volume number.'''
    fname = params.galform_op_dir + params.galform_model + "_{0}_corr.hdf5".format(fno)
    gfile = h5py.File(fname, "r")
    print("Galform prop names ->", gfile["Input/"].keys())
    return gfile


def get_volumes(fno):
    '''Total & per-file cosmological volumes for this sub-volume.'''
    return find_tot_volume(params.galform_op_dir, params.galform_model, [fno])


def read_galform_properties(gfile, iz):
    '''
    Read the GALFORM galaxy properties at redshift index `iz`.
    Returns a dict of 1-D arrays (one entry per galaxy).
    '''
    # Global galaxy properties
    Mstar   = gfile["Input/Mstars_disk"][:, iz] + gfile["Input/Mstars_bulge"][:, iz] # Msun (stellar mass)
    Mbulge  = gfile["Input/Mstars_bulge"][:, iz] # Msun (bulge stellar mass)
    Mgas    = gfile["Input/Mgas_disk"][:, iz]  # Msun (disc gas mass)
    Mhalo   = gfile["Input/Mhalo"][:, iz]     # Msun (helo mass)
    central = gfile["Input/central"][:, iz] # central(1) or satellite (0)
    SFR     = gfile["Input/SFR"][:, iz]  # Msun/yr (star formation rate)

    if params.SFR_correction:
        SFR_ratio_file = np.loadtxt(params.sfr_ratio_file_path)
        print("SFR_ratio_shape", SFR_ratio_file.shape)
        SFR_offset = SFR_ratio_file[iz]
        SFR = SFR * SFR_offset

    return dict(Mstar=Mstar, Mbulge=Mbulge, Mgas=Mgas, Mhalo=Mhalo,
                central=central, SFR=SFR)


def read_magnetizer_properties(Bfile, iz):
    '''
    Read the MAGNETIZER galaxy properties at redshift index `iz`.
    Scalar-per-galaxy quantities and radially-resolved quantities are
    both returned in the same dict.
    '''
    # Global magnetic field properties 
    Bavg = Bfile["Output/Bavg"][:, iz] # muG (vol. avg. magnetic field strength)
    Bmax = Bfile["Output/Bmax"][:, iz] # muG (max. large-scale magnetic field strength)
    rmax = Bfile["Output/rmax"][:, iz] # kpc (position of Bmax)

    # Resolved galaxy properties (including magnetic fields) 
    r     = Bfile["Output/r"][:, :, iz] # kpc (radius)
    h     = Bfile["Output/h"][:, :, iz] # pc (scale hight as a function as radius)
    v     = Bfile["Output/v"][:, :, iz] # km/s (turbulent speed)
    n     = Bfile["Output/n"][:, :, iz] # cm^-3 (gas number density)
    Br    = Bfile["Output/Br"][:, :, iz] # muG (radial comp. of large-scale field)
    Bp    = Bfile["Output/Bp"][:, :, iz] # muG (azimuthal comp. of large-scale field)
    Bz    = Bfile["Output/Bzmod"][:, :, iz] # muG (vertical comp. of large-scale field)
    Beq   = Bfile["Output/Beq"][:, :, iz] # muG (equipartition field strength. small-scale field: b = f_b * Beq)
    P     = Bfile["Output/P"][:, :, iz] # erg cm^-3 (mid-plane pressure)
    Omega = Bfile["Output/Omega"][:, :, iz] # km/s/kpc (angular velosity)
    Ur    = Omega * r # km/s (circular speed)

    return dict(Bavg=Bavg, Bmax=Bmax, rmax=rmax, r=r, h=h, v=v, n=n,
                Br=Br, Bp=Bp, Bz=Bz, Beq=Beq, P=P, Omega=Omega, Ur=Ur)


def build_mask(Bavg, rmax):
    '''Selection mask: keep galaxies with a valid (non-zero) magnetic field.'''
    return (abs(Bavg) > 0.0) & (rmax > 0.0)


def apply_mask_to_dict(data, mask):
    '''Apply `mask` to every array-valued entry of `data` and return a new dict.'''
    return {key: value[mask] for key, value in data.items()}


def make_galaxy_ids(n_galaxies):
    return np.arange(0, n_galaxies, 1)


def save_results(fno, redshift, galid, rmax, Bmax, Bavg, B_rms, Beq_rms,
                  B_total_avg, B_bar_avg, Mhalo, Mstar, Mbulge, SFR_11,
                  Mgas, M_diffuse_n, num_density, S_I, Lum, R_25, Ur_avg, v_t):
    '''Write the per-galaxy output table to the processed-output directory.'''
    fname = (params.processed_op_dir + "example_output/" +
              "Obs_total_Lum_SFRD_L16_mod1_" + params.galform_model + "_" +
              params.mag_model + "_fno{0}_z{1:.1f}.txt".format(fno, redshift))
    np.savetxt(fname, np.transpose([
        galid, rmax, Bmax, Bavg, B_rms, Beq_rms, B_total_avg, B_bar_avg,
        Mhalo, Mstar, Mbulge, SFR_11, Mgas, M_diffuse_n, num_density,
        S_I, Lum, R_25, Ur_avg, v_t
    ]))
    return fname
