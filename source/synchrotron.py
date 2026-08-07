'''
synchrotron.py

Everything needed to turn a galaxy's radial field/gas profile into a
total synchrotron luminosity and an observed flux density:
 - the equipartition-based cosmic-ray electron normalisation K_E and
   angular integration constant a_s,
 - the phi-z synchrotron emissivity model (accounts for an exponentially
   decaying Br/Bp with Bz reconstructed from div.B = 0, plus a randomly
   tangled component),
 - integration of that emissivity over azimuth, height and radius to get
   the total luminosity,
 - conversion of luminosity to an observed flux density at an assumed
   distance, using either the Astropy or the analytic luminosity distance.
'''
import numpy as np
import math
from astropy.cosmology import WMAP9 as cosmo

import constants
import params


# spectral index
def spectral_index(z, sSFR=None, alpha_nt_model=params.alpha_nt_model):
    '''Calculate cosmic-ray spectral index s based on model choice.'''
    # ---- model 1 ------------
    if alpha_nt_model == 1:  # redshift dependent
        alpha_nt = 1.8 * (1 + z) ** (-0.8)  # Eq.12 of Tabatabaei+25
    # ---- model 2 ------------
    elif alpha_nt_model == 2:  # sSFR dependent
        if sSFR is None:
            raise ValueError("sSFR must be provided for alpha_nt_model = 2")
        alpha_nt = (-0.25) * np.log10(sSFR) - 1.37  # Eq.13 of Tabatabaei+25
    # ---- model 3 ------------
    elif alpha_nt_model == 3:  # constant spectral index
        alpha_nt = 1.0  # spectral-index of CR-electron energy spectrum in Solar vicinity
    else:
        raise ValueError(f"Unknown alpha_nt_model: {alpha_nt_model}")

    # non-thermal spectral index
    s = 2.0 * alpha_nt + 1.0
    return s


def a_s_constant(s):
    '''Angular-integration constant appearing in the standard synchrotron
    emissivity formula for a power-law electron spectrum of index s.'''
    a_s = ((np.sqrt(3) / (4 * np.pi * (s + 1))) *
           math.gamma((3 * s - 1) / 12) * math.gamma((3 * s + 19) / 12))
    return a_s


def equipartition_KE(B_T, s, E=params.E, k_cr=params.k_cr):
    '''Cosmic-ray electron normalisation constant from field/energy equipartition.'''
    K_E = ((s - 2) / (8 * np.pi * k_cr)) * (B_T * constants.muG_G) ** 2 * (E * constants.GeV_erg) ** (s - 2)
    return K_E


def random_inclination(rng=None):
    '''Draw a random inclination angle (rad) between the line of sight and
    the disc normal, uniform in sin(i).'''
    if rng is None:
        u = np.random.uniform(0.0, 1.0)
    else:
        u = rng.uniform(0.0, 1.0)
    return np.arccos(1.0 - u)


def luminosity_distance_analytic(H_0, Omega_m, redshift):
    '''Analytic approximation to the luminosity distance (Mpc).'''
    D_L = ((2 * constants.c_light * constants.cm_km / H_0 / Omega_m ** 2) *
           (Omega_m * redshift +
            (Omega_m - 2) * (np.sqrt(Omega_m * redshift + 1) - 1)))
    return D_L


def make_emissivity_function(r1, h1, Br1, Bp1, Bz1, n1, Beq_sq1,
                             inclination, f_b, s, E=params.E, k_cr=params.k_cr,
                             nu=params.nu):
    '''
    Build the phi-z synchrotron emissivity function for one galaxy.

    Returns a function `synchrotron_emissivity(phi, j, z)` matching the
    original closure, where `j` indexes the radial bin (into r1/h1/Br1/...).
    '''
    dBr1_dr1 = np.gradient(Br1, r1)        # d(Br1)/d(r1)
    dh1_dr1 = np.gradient(h1 / 1000.0, r1)  # d(h1)/d(r1) (h1 converted from pc -> kpc)
    a_s = a_s_constant(s)

    def synchrotron_emissivity(phi, j, z):
        r_clamped = np.clip(r1[j], np.min(r1), np.max(r1))

        h_r = h1[j] / 1000.0
        Br_r = Br1[j]
        Bp_r = Bp1[j]
        Beq_sq_r = Beq_sq1[j]
        delBr_delr_r = dBr1_dr1[j]
        delh_delr_r = dh1_dr1[j]

        PHI, Z = np.meshgrid(phi, z)

        sign_z = np.sign(Z)
        sign_z[Z == 0] = 1.0

        Br_rz = Br_r * np.exp(-np.abs(Z) / h_r)
        Bp_rz = Bp_r * np.exp(-np.abs(Z) / h_r)

        # Bz reconstructed from div.B = 0 for an exponentially decaying Br profile
        Bz_rz = sign_z * ((Br_r / r_clamped + delBr_delr_r) *
                          h_r * (np.exp(-np.abs(Z) / h_r) - 1) +
                          Br_r * delh_delr_r *
                          (((np.abs(Z) / h_r) + 1) * np.exp(-np.abs(Z) / h_r) - 1))

        B_bar_rz = np.sqrt(Br_rz ** 2 + Bp_rz ** 2 + Bz_rz ** 2)

        # Sky plane magnetic field components
        Bx = Br_rz * np.cos(PHI) - Bp_rz * np.sin(PHI)
        By = (Br_rz * np.sin(PHI) * np.cos(inclination) +
              Bp_rz * np.cos(PHI) * np.cos(inclination) +
              Bz_rz * np.sin(inclination))

        # Perpendicular components of large-scale magnetic field to LoS
        B_perp_LoS = np.sqrt(Bx ** 2 + By ** 2)

        # Small-scale rms field strength
        b_rms_rz = f_b * np.sqrt(Beq_sq_r) * np.exp(-np.abs(Z) / (2 * h_r))
        b_perp_rz = np.sqrt((2.0 / 3.0) * b_rms_rz ** 2)

        B_T = np.sqrt(B_bar_rz ** 2 + b_rms_rz ** 2)
        K_E = equipartition_KE(B_T, s=s, E=E, k_cr=k_cr)

        # Synchrotron emissivity
        eps = (K_E * a_s * (constants.e_charge ** 3 / (constants.m_e * constants.c_light ** 2)) *
               (3 * constants.e_charge / (4 * np.pi * constants.m_e ** 3 * constants.c_light ** 5)) ** ((s - 1) / 2) *
               (np.sqrt(B_perp_LoS ** 2 + b_perp_rz ** 2) * constants.muG_G) ** ((s + 1) / 2) *
               (nu * constants.GHz_Hz) ** (-(s - 1) / 2))

        return eps

    return synchrotron_emissivity


def compute_luminosity(r1, h1, emissivity_fn, n_phi=51, n_z=501):
    '''
    Integrate the emissivity over azimuth (phi), height (z) and radius (r)
    to obtain the total synchrotron luminosity (erg/s/Hz, cgs) for one galaxy.
    '''
    phi = np.linspace(0.0, 2 * np.pi, n_phi)
    integrated_values_phi = np.zeros(len(r1))

    for j in range(len(r1)):
        z = np.linspace(-5 * h1[j] / 1000.0, 5 * h1[j] / 1000.0, n_z)
        eps = emissivity_fn(phi, j, z)
        integrated_over_phi = np.trapz(eps, phi, axis=1)
        integrated_values_phi[j] = np.trapz(integrated_over_phi, z)

    integrated_value = np.trapz(integrated_values_phi * r1, r1)

    L = integrated_value * (constants.kpc_cm ** 3) * 4 * np.pi
    return L


def compute_flux_density(L, redshift, d=None, use_astropy=False):
    '''
    Convert total luminosity L (erg/s/Hz) to an observed flux density
    S_I (mJy).

    By default the luminosity distance at `redshift` (analytic, or
    Astropy/WMAP9 if use_astropy=True) is used to compute the flux, so the
    result is redshift-dependent as expected. Pass an explicit `d` (Mpc) to
    override this with a fixed distance (e.g. for testing against
    params.d_assumed).
    '''
    if use_astropy:
        d_L = cosmo.luminosity_distance(redshift).value
    else:
        d_L = luminosity_distance_analytic(constants.H_0, constants.Omega_m, redshift)

    d_use = d if d is not None else d_L

    d_cm = d_use * constants.Mpc_cm
    S_I = (L / (4 * np.pi * d_cm ** 2)) * constants.Jy_cgs * constants.Jy_mJy
    return S_I, d_L
