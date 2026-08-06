'''
processed_output.py

Reads radial-dependent MAGNETIZER properties (paired with GALFORM galaxy
properties) and computes the total synchrotron luminosity along with other
global galaxy properties, at each requested redshift and for each requested
sub-volume file.

This is the top level code: all of the actual physics/statistics
live in Bfield_analysis.py, synchrotron.py and gal_props.py; all file
handling lives in io_utils.py; all constants/parameters live in params.py.
'''
import numpy as np

from utils import get_galf_z_output_list, find_nearest
import params
import io_utils as io
import Bfield_analysis as fa
import synchrotron as sync
import gal_props as gs


def process_galaxy(igal, radial, SFR_new, Mstar, redshift):
    '''
    Run the full per-galaxy analysis (field averaging, synchrotron
    luminosity, diffuse gas mass, structural radii, rotation curve) for
    galaxy index `igal`, given the pre-masked radial arrays for the whole
    population (`radial`, a dict of 2-D [galaxy, radius] arrays).

    Returns a dict of the scalar outputs for this galaxy.
    '''
    r = radial['r']
    h = radial['h']
    v = radial['v']
    n = radial['n']
    P = radial['P']
    Br = radial['Br']
    Bp = radial['Bp']
    Bz = radial['Bz']
    B_sq = radial['B_sq']
    Beq_sq = radial['Beq_sq']
    Br_sq = radial['Br_sq']
    Bp_sq = radial['Bp_sq']
    B_total_sq = radial['B_total_sq']
    Ur = radial['Ur']

    print('igal--->', igal, "Done")

    # --- midpoint (trapezoid-ready) radial arrays -----------------------
    mid = fa.radial_midpoints(
        r[igal, :], h[igal, :], v[igal, :], n[igal, :], P[igal, :],
        Br[igal, :], Bp[igal, :], Bz[igal, :],
        B_sq[igal, :], Beq_sq[igal, :], Br_sq[igal, :], Bp_sq[igal, :],
    )
    r1, h1, v1, n1, P1 = mid['r1'], mid['h1'], mid['v1'], mid['n1'], mid['P1']
    Br1, Bp1, Bz1 = mid['Br1'], mid['Bp1'], mid['Bz1']
    B_sq1, Beq_sq1 = mid['B_sq1'], mid['Beq_sq1']
    beta0_sq1, pitch_sq1 = mid['beta0_sq1'], mid['pitch_sq1']
    B_total_sq1 = fa.midpoint(B_total_sq[igal, :])

    SFR_11 = SFR_new[igal]

    # --- rms field strengths & tangling factor ---------------------------
    B_rms, Beq_rms, beta0_rms = fa.rms_field_strengths(B_sq1, Beq_sq1, beta0_sq1, h1, r1)
    f_b = fa.get_f_b(beta0_rms, SFR_11)

    # --- avg. field diagnostics ---------------------------
    avgF = fa.avg_field(B_sq1, Beq_sq1, B_total_sq1, pitch_sq1, h1, r1, n1)

    # --- cosmic-ray spectral index (model choice set in params) ----------
    if params.alpha_nt_model == 2:
        sSFR_11 = SFR_11 / Mstar[igal]
        s = sync.spectral_index(redshift, sSFR=sSFR_11, alpha_nt_model=params.alpha_nt_model)
    else:
        s = sync.spectral_index(redshift, alpha_nt_model=params.alpha_nt_model)

    # --- synchrotron luminosity & flux density ---------------------------
    inclination = sync.random_inclination()
    emissivity_fn = sync.make_emissivity_function(
        r1, h1, Br1, Bp1, Bz1, n1, Beq_sq1, inclination, f_b, s,
    )
    L = sync.compute_luminosity(r1, h1, emissivity_fn)
    S_I, d_L = sync.compute_flux_density(L, redshift)

    # --- diffuse gas mass --------------------------------------------------
    M_diffuse_P, M_diffuse_n = gs.diffuse_gas_mass(P1, n1, h1, r1, v1, method=2)

    # --- weighted pressure / turbulent velocity ----------------------------
    P_avg, v_t = gs.weighted_pressure_and_velocity(P1, v1, h1, r1)

    # --- half-mass radius, R200, R25 ---------------------------------------
    r_half, R_200, R_25 = gs.compute_R25(r[igal, :])

    # --- rotation curve sampling --------------------------------------------
    Ur_R, Ur_R1, Ur_R2, Ur_avg = gs.sample_rotation_curve(Ur[igal, :], r[igal, :])

    return dict(
        B_rms=B_rms, Beq_rms=Beq_rms, beta0_rms=beta0_rms,
        B_bar_avg=avgF['B_bar_avg'], B_total_avg=avgF['B_total_avg'],
        pitch_avg=avgF['pitch_avg'], deg_pol=avgF['deg_pol'],
        num_density=avgF['num_density'], Si=avgF['Si'],
        SFR_11=SFR_11, Lum=L, S_I=S_I,
        M_diffuse_P=M_diffuse_P, M_diffuse_n=M_diffuse_n,
        P_avg=P_avg, v_t=v_t,
        r_half=r_half, R_200=R_200, R_25=R_25,
        Ur_R=Ur_R, Ur_R1=Ur_R1, Ur_R2=Ur_R2, Ur_avg=Ur_avg,
    )


def process_redshift(gfile, Bfile, iz, redshift, fno):
    '''Process every masked galaxy at one redshift slice, then save results.'''
    print('redshift =', redshift)

    # --- read raw properties --------------------------------------------
    galform = io.read_galform_properties(gfile, iz)
    magnetizer = io.read_magnetizer_properties(Bfile, iz)

    galid = io.make_galaxy_ids(len(galform['Mstar']))
    print("Number of galform galaxies=", len(galform['Mstar']))
    print("Number of magnetizer galaxies=", len(magnetizer['Bavg']))

    # --- build & apply mask -----------------------------------------------
    mask = io.build_mask(magnetizer['Bavg'], magnetizer['rmax'])

    galform = io.apply_mask_to_dict(galform, mask)
    magnetizer = io.apply_mask_to_dict(magnetizer, mask)
    galid = galid[mask]

    print("Number of galform galaxies after masking=", len(galform['Mstar']))
    print("Number of magnetizer galaxies after masking=", len(magnetizer['Bavg']))

    # --- vertical-average correction & derived squared quantities ----------
    Br, Bp, Bz = fa.apply_exp_factor(magnetizer['Br'], magnetizer['Bp'], magnetizer['Bz'])
    Beq = magnetizer['Beq']

    B_sq = Br ** 2.0 + Bp ** 2.0 + Bz ** 2.0
    Beq_sq = Beq ** 2.0
    Br_sq = Br ** 2.0
    Bp_sq = Bp ** 2.0
    B_total_sq = B_sq + Beq_sq

    radial = dict(
        r=magnetizer['r'], h=magnetizer['h'], v=magnetizer['v'], n=magnetizer['n'],
        P=magnetizer['P'], Br=Br, Bp=Bp, Bz=Bz,
        B_sq=B_sq, Beq_sq=Beq_sq, Br_sq=Br_sq, Bp_sq=Bp_sq,
        B_total_sq=B_total_sq, Ur=magnetizer['Ur'],
    )

    n_gals = len(galform['Mstar'])
    np.random.seed(42)

    # how often (in galaxies) to write a checkpoint of everything done so far
    checkpoint_every = 100

    # --- per-galaxy output arrays --------------------------------------------
    out = {key: np.zeros(n_gals) for key in [
        'B_rms', 'Beq_rms', 'beta0_rms', 'B_bar_avg', 'B_total_avg',
        'pitch_avg', 'deg_pol', 'num_density', 'Si', 'SFR_11', 'Lum', 'S_I',
        'M_diffuse_P', 'M_diffuse_n', 'P_avg', 'v_t',
        'r_half', 'R_200', 'R_25', 'Ur_R', 'Ur_R1', 'Ur_R2', 'Ur_avg',
    ]}

    def checkpoint_save(n_done):
        '''Write out everything computed so far (galaxies [0, n_done)).'''
        sl = slice(0, n_done)
        fname = io.save_results(
            fno, redshift, galid[sl], magnetizer['rmax'][sl], magnetizer['Bmax'][sl],
            magnetizer['Bavg'][sl],
            out['B_rms'][sl], out['Beq_rms'][sl], out['B_total_avg'][sl], out['B_bar_avg'][sl],
            galform['Mhalo'][sl], galform['Mstar'][sl], galform['Mbulge'][sl], out['SFR_11'][sl],
            galform['Mgas'][sl], out['M_diffuse_n'][sl], out['num_density'][sl], out['S_I'][sl],
            out['Lum'][sl], out['R_25'][sl], out['Ur_avg'][sl], out['v_t'][sl],
        )
        print(f'Checkpoint saved ({n_done}/{n_gals} galaxies) ->', fname)
        return fname

    for igal in range(n_gals):
        result = process_galaxy(igal, radial, galform['SFR'], galform['Mstar'], redshift)
        for key, value in result.items():
            out[key][igal] = value

        if igal == 3:
            print('inclination/Lum check -> Lum:', out['Lum'][igal])
        if igal == 51:
            print("R_25", out['R_25'][igal])
            print('r_half', out['r_half'][igal])

        # write a checkpoint every `checkpoint_every` galaxies
        if (igal + 1) % checkpoint_every == 0:
            checkpoint_save(igal + 1)

    print('----Done----')

    # final save covering every galaxy (also catches any leftover
    # galaxies since the last checkpoint, e.g. n_gals not a multiple of 500)
    fname = checkpoint_save(n_gals)
    print('Saved ->', fname)


def main():
    galf_z = get_galf_z_output_list(params.galform_op_dir, params.galform_model, params.filenos[0])

    for fno in params.filenos:
        Bfile = io.open_magnetizer_file(fno)
        volume, v_ay = io.get_volumes(fno)
        gfile = io.open_galform_file(fno)

        for redshift in params.z_ay:
            iz, zgf = find_nearest(galf_z, redshift)
            print('closest redshift =', zgf)
            process_redshift(gfile, Bfile, iz, redshift, fno)


if __name__ == "__main__":
    main()
