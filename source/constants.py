'''
constants.py

Physical constants and unit conversions only. Nothing in this file should
ever need editing between runs - if you want to change what the pipeline
does (paths, redshifts, model choice, synchrotron parameters like the
spectral index), see params.py instead.
'''

# ----------------------------------------------------------------------
# Diffuse gas mass unit conversions & constants (SG)
# ----------------------------------------------------------------------
M_sun  = 1.989 * (10 ** 30)          # kg
km_kpc = 3.08567758 * (10 ** 16)     # km
cm_kpc = 3.08567758 * (10 ** 21)     # cm
g_kg   = 10 ** 3                     # g
m_km   = 10 ** 3                     # m
m_cm   = 10 ** (-2)                  # m
V0     = 25                          # km/s
m_H    = 1.67372 * 10 ** (-24)       # g

# Zeta (from LC's "pressure" note)
fb   = 1
xi   = (3 / 2) * fb ** 2
eps_ = 1
zeta = (1 + xi + xi * eps_) / 3

erg_j = 10 ** (-7)
pi    = 3.14156295358

# ----------------------------------------------------------------------
# Synchrotron / CGS physical constants
# ----------------------------------------------------------------------
e_charge = 4.8032 * 10 ** (-10)   # cm^(3/2) g^(1/2) s^-1 : electron charge (statcoulomb)
m_e      = 9.1094 * 10 ** (-28)   # g   : electron mass
c_light  = 2.9979 * 10 ** 10      # cm/s: speed of light
H_0      = 69.32                  # km/s/Mpc : Hubble constant
Omega_m  = 0.32                   # mass density parameter (Planck 2016)

# ----------------------------------------------------------------------
# Unit conversions
# ----------------------------------------------------------------------
kpc_cm  = 3.086 * 10 ** 21   # cm
Mpc_cm  = 3.086 * 10 ** 24   # cm
cm_km   = 10 ** (-5)         # km
GHz_Hz  = 10 ** 9            # Hz
muG_G   = 10 ** (-6)         # muG -> G
GeV_erg = 0.00160218         # GeV -> erg
Jy_mJy  = 10 ** 3            # Jy -> mJy
Jy_cgs  = 10 ** (-23)        # erg s^-1 cm^-2 Hz^-1
