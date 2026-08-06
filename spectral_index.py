import numpy as np
import params


def spectral_index(alpha_nt_model= params.alpha_nt_model, z)
    #---- model 1 ------------
    if alpha_nt_model == 1: # redshift dependent
        alpha_nt_z = 1.8 * (1+z)**(-0.8) # Eq.12 of Tabatabaei+25
        alpha_nt   = alpha_nt_z
    #---- model 2 ------------
    elif alpha_nt_model == 2: # sSFR dependent
        alpha_nt_sSFR = (-0.25)*np.log10(sSFR[igal]) - 1.37 # Eq.13 of Tabatabaei+25 
        alpha_nt = alpha_nt_sSFR
    #---- model 3 ------------
    elif alpha_nt_model == 3: # constant spectral index
        alpha_nt_const = 1 #spectral-index of CR-electron energy spectrum in the Solar vicinity (ss21 fig 10.10)
        alpha_nt = alpha_nt_const
    # non-thermal spectral index
    s    = (2.0 * alpha_nt + 1) 
    
    return s
