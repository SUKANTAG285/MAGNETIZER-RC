'''
Radial-profile analysis of a single galaxy's magnetic field:
 - converting the tabulated radial arrays into cell-centred ("midpoint")
   arrays suitable for trapezoidal integration,
 - volume/area-weighted rms field strengths (B_rms, Beq_rms, beta0_rms),
 - the large-scale tangling factor f_b,
 - luminosity/area-weighted averages used for polarization diagnostics
   (B_bar_avg, B_total_avg, pitch_avg, deg_pol, num_density, Si).

All functions here operate on the radial arrays for ONE galaxy, matching the per-galaxy loop in the
original script.
'''
import numpy as np

import params

def midpoint(arr):
    '''Cell-centred (trapezoid-ready) version of a 1-D radial array.'''
    return (arr[1:] + arr[:-1]) / 2.0

def diff(arr):
    return arr[1:] - arr[:-1]


def apply_exp_factor(Br, Bp, Bz):
    '''
    Convert mid-plane Br/Bp/Bz into the vertically-averaged (between -h,h)
    field strengths for an exponentially decaying profile, if requested.
    '''
    if params.vertical_avg_Bbar:
        exp_factor = 1 / (1 - np.exp(-1))
    else:
        exp_factor = 1
        
    Br_midplane = Br * exp_factor
    Bp_midplane = Bp * exp_factor
    Bz_midplane = Bz * exp_factor
    return Br_midplane, Bp_midplane, Bz_midplane


def radial_midpoints(r, h, v, n, P, Br, Bp, Bz, B_sq, Beq_sq, Br_sq, Bp_sq):
    '''
    Build the midpoint versions of every radial array for one galaxy,
    plus the radial step dr1 needed by np.gradient-style calculations.
    Returns a dict of midpoint arrays.
    '''
    r1 = midpoint(r)
    h1 = midpoint(h)
    dr1 = diff(r)
    v1 = midpoint(v)
    n1 = midpoint(n)
    P1 = midpoint(P)
    Br1 = midpoint(Br)
    Bp1 = midpoint(Bp)
    Bz1 = midpoint(Bz)
    B_sq1 = midpoint(B_sq)
    Beq_sq1 = midpoint(Beq_sq)
    Br_sq1 = midpoint(Br_sq)
    Bp_sq1 = midpoint(Bp_sq)
    beta0_sq1 = B_sq1 / Beq_sq1
    pitch_sq1 = Br_sq1 / Bp_sq1

    return dict(r1=r1, h1=h1, dr1=dr1, v1=v1, n1=n1, P1=P1,
                Br1=Br1, Bp1=Bp1, Bz1=Bz1, B_sq1=B_sq1, Beq_sq1=Beq_sq1,
                Br_sq1=Br_sq1, Bp_sq1=Bp_sq1,
                beta0_sq1=beta0_sq1, pitch_sq1=pitch_sq1)


def rms_field_strengths(B_sq1, Beq_sq1, beta0_sq1, h1, r1):
    '''
    Volume-weighted rms of the large-scale field, the equipartition field,
    and beta0, integrated across the full radial range (out to ~2.7 rhalf).
    '''
    B_rms = (np.trapz(B_sq1 * h1 * r1, r1) / np.trapz(h1 * r1, r1)) ** 0.5
    Beq_rms = (np.trapz(Beq_sq1 * h1 * r1, r1) / np.trapz(h1 * r1, r1)) ** 0.5
    beta0_rms = (np.trapz(beta0_sq1 * h1 * r1, r1) / np.trapz(h1 * r1, r1)) ** 0.5
    return B_rms, Beq_rms, beta0_rms


def get_f_b(beta0_rms_val, SFR_val, profile = None):
    '''
    Large-scale field tangling factor f_b, following one of four
    prescriptions selected by `profile` (defaults to config.f_b_profile).
    '''
    if profile is None:
        profile = params.f_b_profile

    if profile == 1:  # step function of beta0_rms
        return 0.1 if beta0_rms_val < 0.1 else 1
    elif profile == 2:  # step function of SFR
        if SFR_val > 1:
            return 1.5
        elif SFR_val > 0.1:
            return 1.2
        else:
            return 1
    elif profile == 3:  # continuous function of beta0_rms
        return beta0_rms_val if beta0_rms_val <= 1 else 1
    elif profile == 4:  # constant
        return 0.8 #(fiducial value used in the MAGNETIZER)
    else:
        raise ValueError("choose f_b profile (1-4)")


def avg_field(B_sq1, Beq_sq1, B_total_sq1, pitch_sq1, h1, r1, n1):
    '''
    Luminosity/area-weighted diagnostics used to characterise the
    (unresolved) polarization properties of the disc:
      B_bar_avg   - rms mean-field strength (line-of-sight weighted)
      B_total_avg - rms total-field strength (4th-moment weighted)
      pitch_avg   - luminosity-weighted pitch-angle-related quantity
      deg_pol     - degree of polarization proxy
      num_density - volume-averaged gas number density
      Si          - unpolarized intensity-like integral
    '''
    B_bar_avg = (np.trapz((B_sq1 / B_total_sq1) * h1 * r1 * (1 - np.exp(-2)), r1) /
                 np.trapz(2 * h1 * r1, r1)) ** 0.5

    B_total_avg = (np.trapz((np.sqrt(B_total_sq1) ** 4) * (1 - np.exp(-4)) * h1 * r1, r1) /
                   np.trapz(2 * h1 * r1, r1)) ** (1 / 4)

    pitch_avg = (np.trapz((np.sqrt(B_total_sq1)) ** 2 * (np.sqrt(B_sq1)) ** 2 *
                           np.sqrt(pitch_sq1) * 2 * np.pi * r1 * h1, r1) /
                 np.trapz((np.sqrt(B_total_sq1)) ** 2 * (np.sqrt(B_sq1)) ** 2 *
                          2 * np.pi * r1 * h1, r1))

    num_density = np.trapz(n1 * r1, r1) / np.trapz(r1, r1)

    deg_pol = (np.trapz((np.sqrt(B_total_sq1)) ** 2 * (np.sqrt(B_sq1)) ** 2 *
                         2 * np.pi * r1 * h1, r1) /
               np.trapz((np.sqrt(B_total_sq1)) ** 4 * 2 * np.pi * r1 * h1, r1))

    Si = np.trapz(((np.sqrt(B_total_sq1)) ** 2) * ((np.sqrt(B_total_sq1)) ** 2) *
                  2 * np.pi * r1 * h1 / 1000, r1)

    return dict(B_bar_avg=B_bar_avg, B_total_avg=B_total_avg,
                pitch_avg=pitch_avg, deg_pol=deg_pol,
                num_density=num_density, Si=Si)
