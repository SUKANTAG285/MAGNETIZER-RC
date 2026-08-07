'''
gal_props.py

Structural / bulk properties of a single galaxy derived from its radial
profile:
 - diffuse (cold) gas mass, either from the mid-plane pressure or directly
   from the gas number density,
 - weighted-average mid-plane pressure and turbulent velocity,
 - half-mass radius, R200 and R25 (Kravtsov 2013 scaling),
 - rotation speed sampled at fixed multiples of the half-mass radius and
   averaged over the "flat" part of the rotation curve.

All functions operate on the radial arrays for ONE galaxy.
'''
import numpy as np

from utils import find_nearest
import constants


def diffuse_gas_mass(P1, n1, h1, r1, v1, method=2):
    '''
    Diffuse gas mass (Msun), computed by one of two methods:
      1 - from mid-plane pressure (R19 eq. 8, SI units)
      2 - directly from the gas number density (CGS units)
    Returns (M_diffuse_P, M_diffuse_n) - whichever wasn't computed is 0.0.
    '''
    M_diffuse_P = 0.0
    M_diffuse_n = 0.0

    if method == 1:
        M_diffuse_P = (np.trapz(P1 * 2 * np.pi * h1 * r1 / 1000 / v1 ** 2 / constants.zeta, r1) *
                       (constants.erg_j * (1 / (constants.m_cm ** 3)) *
                        (constants.km_kpc * constants.m_km) ** 3 / (constants.m_km) ** 2 / constants.M_sun))
    elif method == 2:
        M_diffuse_n = (np.trapz(n1 * constants.m_H * 2 * np.pi * h1 * r1 / 1000, r1) *
                       (constants.cm_kpc) ** 3 / (constants.M_sun * constants.g_kg))
    else:
        raise ValueError("choose a particular method (1 or 2) to compute diffuse gas mass")

    return M_diffuse_P, M_diffuse_n


def weighted_pressure_and_velocity(P1, v1, h1, r1):
    '''Mass(area)-weighted average mid-plane pressure and turbulent speed.'''
    P_avg = np.trapz(P1 * h1 * r1, r1) / np.trapz(h1 * r1, r1)
    v_t = np.trapz(v1 * h1 * r1, r1) / np.trapz(h1 * r1, r1)
    return P_avg, v_t


def rms_field_within_rhalf(r_row, B_sq_row, Beq_sq_row):
    '''
    Recompute B_rms/Beq_rms-style quantities restricted to r < rhalf
    (rhalf = r_max/2.7), matching the original script's second pass.
    Returns (irhalf, rhalf, B_sq1, Beq_sq1, r1_half, beta0_sq1).
    '''
    irhalf, rhalf = find_nearest(r_row, r_row[-1] / 2.7)
    B_sq1 = B_sq_row[1:irhalf] + B_sq_row[:irhalf - 1]
    Beq_sq1 = (Beq_sq_row[1:irhalf] + Beq_sq_row[:irhalf - 1]) / 2.0
    r1_half = (r_row[1:irhalf] + r_row[:irhalf - 1]) / 2.0
    beta0_sq1 = B_sq1 / Beq_sq1
    return irhalf, rhalf, B_sq1, Beq_sq1, r1_half, beta0_sq1


def compute_R25(r_row):
    '''
    Half-mass radius, R200 and R25 for one galaxy, following the
    Kravtsov (2013, ApJ) scaling relations used in the original script.
    '''
    _, rhalf = find_nearest(r_row, r_row[-1] / 2.7)
    r_half = rhalf                      # kpc
    R_200 = r_half / 0.015              # kpc
    R_25 = 0.048 * R_200                # kpc
    return r_half, R_200, R_25


def sample_rotation_curve(Ur_row, r_row):
    '''
    Sample the rotation speed Ur at 1.5, 2.0 and 2.5 half-mass radii, and
    average Ur over the "flat" segment between 1.5 and 2.5 * r_half.
    Returns (Ur_R, Ur_R1, Ur_R2, Ur_avg).
    '''
    r_max = r_row[-1]

    iR, _ = find_nearest(r_row, (r_max * 2.0) / 2.7)
    Ur_R = Ur_row[iR]

    iR1, _ = find_nearest(r_row, (r_max * 1.5) / 2.7)
    Ur_R1 = Ur_row[iR1]

    iR2, _ = find_nearest(r_row, (r_max * 2.5) / 2.7)
    Ur_R2 = Ur_row[iR2]

    Ur_slice = Ur_row[iR1:iR2 + 1]
    Ur_avg = np.mean(Ur_slice)

    return Ur_R, Ur_R1, Ur_R2, Ur_avg
