#################################### Observed RLF Data #################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from matplotlib import rcParams
from shapely.geometry import Polygon
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.odr import ODR, Model, RealData
import matplotlib.colors as mcolors
from scipy.interpolate import griddata




########################################## 1.4 GHz #####################################################

########################################### Best+05
LogL_Best05 = np.array([21.75, 22.15, 22.35, 22.65, 22.95,
23.25, 23.55, 23.85
])
RLF_Best05 = np.array([-2.92, -3.20, -3.46, -3.96, -4.41, -4.99,
-5.55, -6.44
])

RLF_Best05_err_plus = np.array([0.13,0.05,0.05,0.05,0.06,0.05,0.06,0.12
])

RLF_Best05_err_minus = np.array([0.18,0.06,0.06,0.05, 0.07, 0.06, 0.07,0.17
])

########################################## Condon+02 ###################################################
# Data
LogL_Condon02 = np.array([
    18.8, 19.2, 19.6, 20.0, 20.4, 20.8,
    21.2, 21.6, 22.0, 22.4, 22.8, 23.2, 23.6
])

RLF_Condon02 = np.array([
    -2.44, -2.29, -2.23, -2.39, -2.53, -2.62,
    -2.94, -3.14, -3.44, -3.98, -4.68, -5.58, -6.25
])+ np.log10(2.5)

RLF_Condon02_err_plus = np.array([
    0.52, 0.19, 0.12, 0.06, 0.06, 0.04,
    0.04, 0.03, 0.04, 0.04, 0.07, 0.14, 0.52
])

RLF_Condon02_err_minus = np.array([
    0.74, 0.38, 0.16, 0.08, 0.07, 0.05,
    0.04, 0.04, 0.05, 0.04, 0.08, 0.21, 0.74
])


########################################## Condon+19 ###################################################
# Data
LogL_Condon19 = np.array([
    19.4, 19.6, 19.8, 20.0, 20.2, 20.4, 20.6, 20.8,
    21.0, 21.2, 21.4, 21.6, 21.8, 22.0, 22.2, 22.4,
    22.6, 22.8, 23.0, 23.2, 23.4, 23.6, 23.8
])

RLF_Condon19 = np.array([
    -1.81, -2.39, -2.29, -2.37, -2.46, -2.20, -2.29, -2.36, -2.39, -2.47, 
    -2.58, -2.68, -2.88, -3.00, -3.25, -3.57, -3.90, -4.34, -4.74, -5.30, 
    -5.63, -7.39, -6.96
])

RLF_Condon19_err_plus = np.array([
    0.16, 0.30, 0.12, 0.14, 0.09, 0.07, 0.05, 0.04,
    0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    0.03, 0.04, 0.05, 0.10, 0.12, 0.52, 0.52
])

RLF_Condon19_err_minus = np.array([
    0.26, 0.34, 0.17, 0.21, 0.11, 0.08, 0.05, 0.04,
    0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    0.03, 0.05, 0.06, 0.13, 0.17, 0.76, 0.76
])






########################################## Mauch+07 ###################################################
# Data (1st column vs 5th column)

LogL_Mauch07 = np.array([
    20.0, 20.4, 20.8, 21.2, 21.6, 22.0, 22.4, 22.8, 23.2, 23.6
])

RLF_Mauch07 = np.array([
    -2.90, -2.56, -2.89, -2.85, -3.05, -3.31, -3.79, -4.45, -5.36, -6.12
]) + np.log10(2.5)

# Errors
RLF_Mauch07_err_plus = np.array([
    0.20, 0.09, 0.04, 0.03, 0.02, 0.01, 0.02, 0.02, 0.04, 0.11
])

RLF_Mauch07_err_minus = np.array([
    0.39, 0.11, 0.05, 0.03, 0.02, 0.01, 0.02, 0.02, 0.05, 0.14
])
########################################## Novak+17 ####################################################

# Redshift range 0.1<z<0.4 [z_mid = 0.25]----------------------------------------------------------------
# Data
LogL_z0p25_Novak17 = np.array([21.77, 22.15, 22.46, 22.77, 23.09, 23.34])
RLF_z0p25_Novak17  = np.array([-2.85, -2.88, -3.12, -3.55, -4.05, -4.63])
# errors
LogL_z0p25_Novak17_err_plus  = np.array([0.23, 0.18, 0.19, 0.20, 0.21, 0.28])
LogL_z0p25_Novak17_err_minus = np.array([1.10, 0.15, 0.14, 0.12, 0.12, 0.048])

RLF_z0p25_Novak17_err_plus  = np.array([0.094, 0.031, 0.037, 0.063, 0.11, 0.25])
RLF_z0p25_Novak17_err_minus = np.array([0.077, 0.029, 0.034, 0.055, 0.090, 0.22])
# counts
counts_z0p2_Novak17 = np.array([189, 217, 149, 56, 19, 5])



# Redshift range 0.4<z<0.6 [z_mid = 0.5]---------------------------------------------------------------
# Data
LogL_z0p5_Novak17 = np.array([22.29, 22.54, 22.80, 23.04, 23.31, 23.68])
RLF_z0p5_Novak17  = np.array([-2.97, -3.19, -3.33, -3.67, -4.32, -5.05])

# errors
LogL_z0p5_Novak17_err_plus  = np.array([0.11, 0.13, 0.15, 0.18, 0.18, 0.081])
LogL_z0p5_Novak17_err_minus = np.array([0.31, 0.14, 0.12, 0.090, 0.096, 0.19])

RLF_z0p5_Novak17_err_plus  = np.array([0.052, 0.036, 0.038, 0.059, 0.12, 0.34])
RLF_z0p5_Novak17_err_minus = np.array([0.046, 0.033, 0.035, 0.052, 0.097, 0.30])

# counts
counts_z0p5_Novak17 = np.array([142, 160, 143, 65, 16, 3])

# Redshift range 0.6<z<0.8 [z_mid = 0.7]---------------------------------------------------------------
# Data
LogL_z0p7_Novak17 = np.array([22.61, 22.84, 23.11, 23.40, 23.71, 24.06])
RLF_z0p7_Novak17  = np.array([-2.89, -3.13, -3.47, -3.99, -4.68, -5.43])

# errors
LogL_z0p7_Novak17_err_plus  = np.array([0.080, 0.15, 0.17, 0.17, 0.16, 0.10])
LogL_z0p7_Novak17_err_minus = np.array([0.23, 0.15, 0.12, 0.12, 0.13, 0.19])

RLF_z0p7_Novak17_err_plus  = np.array([0.091, 0.027, 0.035, 0.066, 0.16, 0.45])
RLF_z0p7_Novak17_err_minus = np.array([0.075, 0.025, 0.033, 0.057, 0.11, 0.37])

# counts
counts_z0p7_Novak17 = np.array([179, 283, 165, 51, 11, 2])


# Redshift range 0.8<z<1.0 [z_mid = 0.9]---------------------------------------------------------------
# Data
LogL_z0p9_Novak17 = np.array([22.85, 23.05, 23.30, 23.54, 23.81, 24.11])
RLF_z0p9_Novak17  = np.array([-3.01, -3.13, -3.45, -3.85, -4.31, -4.89])

# errors
LogL_z0p9_Novak17_err_plus  = np.array([0.074, 0.13, 0.14, 0.16, 0.15, 0.11])
LogL_z0p9_Novak17_err_minus = np.array([0.17, 0.13, 0.12, 0.099, 0.11, 0.15])

RLF_z0p9_Novak17_err_plus  = np.array([0.046, 0.025, 0.032, 0.051, 0.088, 0.18])
RLF_z0p9_Novak17_err_minus = np.array([0.041, 0.024, 0.030, 0.046, 0.073, 0.17])

# counts
counts_z0p9_Novak17 = np.array([172, 312, 198, 82, 30, 8])


# Redshift range 1.0<z<1.3 [z_mid = 1.15]--------------------------------------------------------------
# Data
LogL_z1p15_Novak17 = np.array([23.10, 23.31, 23.57, 23.84, 24.06, 24.38])
RLF_z1p15_Novak17  = np.array([-3.19, -3.42, -3.86, -4.15, -4.74, -5.25])

# errors
LogL_z1p15_Novak17_err_plus  = np.array([0.081, 0.15, 0.16, 0.16, 0.22, 0.17])
LogL_z1p15_Novak17_err_minus = np.array([0.21, 0.12, 0.12, 0.11, 0.051, 0.10])

RLF_z1p15_Novak17_err_plus  = np.array([0.052, 0.025, 0.036, 0.052, 0.10, 0.20])
RLF_z1p15_Novak17_err_minus = np.array([0.046, 0.024, 0.034, 0.046, 0.084, 0.19])

# counts
counts_z1p15_Novak17 = np.array([216, 321, 156, 81, 22, 7])


# Redshift range 1.3<z<1.6 [z_mid = 1.45]--------------------------------------------------------------
# Data
LogL_z1p45_Novak17 = np.array([23.32, 23.53, 23.81, 24.15, 24.39, 24.82])
RLF_z1p45_Novak17  = np.array([-3.36, -3.55, -4.10, -4.53, -5.30, -5.94])

# errors
LogL_z1p45_Novak17_err_plus  = np.array([0.070, 0.18, 0.21, 0.19, 0.26, 0.14])
LogL_z1p45_Novak17_err_minus = np.array([0.16, 0.14, 0.10, 0.12, 0.053, 0.17])

RLF_z1p45_Novak17_err_plus  = np.array([0.043, 0.025, 0.041, 0.068, 0.18, 0.45])
RLF_z1p45_Novak17_err_minus = np.array([0.039, 0.024, 0.037, 0.059, 0.17, 0.37])

# counts
counts_z1p45_Novak17 = np.array([156, 323, 126, 48, 8, 2])


# Redshift range 1.6<z<2.0 [z_mid = 1.8]---------------------------------------------------------------
# Data
LogL_z1p8_Novak17 = np.array([23.55, 23.72, 23.98, 24.28, 24.53, 24.90])
RLF_z1p8_Novak17  = np.array([-3.47, -3.66, -4.15, -4.56, -5.06, -5.86])

# errors
LogL_z1p8_Novak17_err_plus  = np.array([0.065, 0.18, 0.21, 0.18, 0.23, 0.14])
LogL_z1p8_Novak17_err_minus = np.array([0.18, 0.11, 0.080, 0.10, 0.059, 0.14])

RLF_z1p8_Novak17_err_plus  = np.array([0.084, 0.025, 0.038, 0.062, 0.12, 0.34])
RLF_z1p8_Novak17_err_minus = np.array([0.070, 0.024, 0.035, 0.054, 0.092, 0.30])

# counts
counts_z1p8_Novak17 = np.array([156, 312, 141, 57, 18, 3])


# Redshift range 2.0<z<2.5 [z_mid = 2.25]--------------------------------------------------------------
# Data
LogL_z2p25_Novak17 = np.array([23.74, 23.94, 24.19, 24.43, 24.66, 24.96])
RLF_z2p25_Novak17  = np.array([-3.61, -3.86, -4.35, -4.71, -5.04, -5.61])

# errors
LogL_z2p25_Novak17_err_plus  = np.array([0.086, 0.13, 0.13, 0.13, 0.14, 0.092])
LogL_z2p25_Novak17_err_minus = np.array([0.15, 0.11, 0.12, 0.11, 0.10, 0.15])

RLF_z2p25_Novak17_err_plus  = np.array([0.050, 0.031, 0.046, 0.072, 0.11, 0.22])
RLF_z2p25_Novak17_err_minus = np.array([0.045, 0.029, 0.042, 0.061, 0.086, 0.20])

# counts
counts_z2p25_Novak17 = np.array([141, 219, 98, 44, 21, 6])


# Redshift range 2.5<z<3.3 [z_mid = 2.9]---------------------------------------------------------------
# Data
LogL_z2p9_Novak17 = np.array([24.01, 24.20, 24.42, 24.68, 24.92, 25.18])
RLF_z2p9_Novak17  = np.array([-3.96, -4.21, -4.62, -4.94, -5.43, -6.27])

# errors
LogL_z2p9_Novak17_err_plus  = np.array([0.079, 0.13, 0.15, 0.13, 0.13, 0.10])
LogL_z2p9_Novak17_err_minus = np.array([0.21, 0.11, 0.091, 0.11, 0.11, 0.13])

RLF_z2p9_Novak17_err_plus  = np.array([0.057, 0.037, 0.051, 0.076, 0.16, 0.45])
RLF_z2p9_Novak17_err_minus = np.array([0.051, 0.034, 0.046, 0.065, 0.12, 0.37])

# counts
counts_z2p9_Novak17 = np.array([128, 155, 81, 39, 11, 2])


# Redshift range 3.3<z<4.6 [z_mid = 3.95]--------------------------------------------------------------
# Data
LogL_z3p95_Novak17 = np.array([24.30, 24.48, 24.67, 24.86, 25.13])
RLF_z3p95_Novak17  = np.array([-4.58, -5.07, -5.24, -5.76, -5.91])

# errors
LogL_z3p95_Novak17_err_plus  = np.array([0.097, 0.12, 0.13, 0.13, 0.067])
LogL_z3p95_Novak17_err_minus = np.array([0.24, 0.081, 0.074, 0.066, 0.13])

RLF_z3p95_Novak17_err_plus  = np.array([0.096, 0.093, 0.10, 0.22, 0.25])
RLF_z3p95_Novak17_err_minus = np.array([0.079, 0.077, 0.082, 0.20, 0.22])

# counts
counts_z3p95_Novak17 = np.array([55, 27, 23, 6, 5])


# Redshift range 4.6<z<5.7 [z_mid = 5.15]--------------------------------------------------------------
# Data
LogL_z5p15_Novak17 = np.array([24.51, 24.71, 24.88, 25.06])
RLF_z5p15_Novak17  = np.array([-4.85, -5.50, -5.86, -6.03])

# errors
LogL_z5p15_Novak17_err_plus  = np.array([0.085, 0.095, 0.13, 0.17])
LogL_z5p15_Novak17_err_minus = np.array([0.13, 0.12, 0.076, 0.042])

RLF_z5p15_Novak17_err_plus  = np.array([0.24, 0.20, 0.28, 0.34])
RLF_z5p15_Novak17_err_minus = np.array([0.15, 0.19, 0.25, 0.30])

# counts
counts_z5p15_Novak17 = np.array([11, 7, 4, 3])








########################################## van-der-Vlught+22 ####################################################
# Redshift range 0.1<z<0.4 [z_mid = 0.25]--------------------------------------------------------------
# Data
LogL_z0p25_vanderVlught22 = np.array([20.88, 21.13, 21.5, 21.74, 21.94, 22.23, 22.5, 22.88])
RLF_z0p25_vanderVlught22  = np.array([-2.4, -2.25, -2.31, -2.5, -2.5, -2.9, -3.31, -3.21])

# errors
LogL_z0p25_vanderVlught22_err_minus  = np.array([0.46, 0.13, 0.22, 0.17, 0.08, 0.08, 0.07, 0.17])
LogL_z0p25_vanderVlught22_err_plus = np.array([0.11, 0.16, 0.07, 0.12, 0.20, 0.20, 0.22, 0.12])

RLF_z0p25_vanderVlught22_err_minus  = np.array([0.34, 0.11, 0.08, 0.07, 0.07, 0.10, 0.23, 0.20])
RLF_z0p25_vanderVlught22_err_plus = np.array([0.21, 0.11, 0.08, 0.07, 0.07, 0.10, 0.16, 0.14])


# Redshift range 0.4<z<0.6 [z_mid = 0.5]---------------------------------------------------------------
# Data
LogL_z0p5_vanderVlught22 = np.array([21.68, 21.9, 22.02, 22.24, 22.42, 22.57, 22.68, 22.93, 23.14])
RLF_z0p5_vanderVlught22  = np.array([-2.58, -2.56, -2.54, -2.58, -2.84, -3.04, -3.38, -3.78, -3.78])

# errors
LogL_z0p5_vanderVlught22_err_minus  = np.array([0.05, 0.10, 0.05, 0.10, 0.11, 0.08, 0.03, 0.10, 0.14])
LogL_z0p5_vanderVlught22_err_plus = np.array([0.12, 0.07, 0.12, 0.07, 0.06, 0.09, 0.14, 0.07, 0.03])

RLF_z0p5_vanderVlught22_err_minus  = np.array([0.29, 0.13, 0.10, 0.09, 0.11, 0.13, 0.29, 0.57, 0.57])
RLF_z0p5_vanderVlught22_err_plus = np.array([0.19, 0.13, 0.10, 0.09, 0.11, 0.13, 0.19, 0.28, 0.28])


# Redshift range 0.6<z<0.8 [z_mid = 0.7]---------------------------------------------------------------
# Data
LogL_z0p7_vanderVlught22 = np.array([22.05, 22.26, 22.51, 22.72, 22.91, 23.14, 23.41, 23.85])
RLF_z0p7_vanderVlught22  = np.array([-2.47, -2.46, -2.82, -2.95, -3.2, -3.36, -3.95, -4.49])

# errors
LogL_z0p7_vanderVlught22_err_minus  = np.array([0.15, 0.12, 0.14, 0.12, 0.08, 0.07, 0.12, 0.32])
LogL_z0p7_vanderVlught22_err_plus = np.array([0.09, 0.11, 0.09, 0.11, 0.16, 0.16, 0.12, 0.14])

RLF_z0p7_vanderVlught22_err_minus  = np.array([0.10, 0.06, 0.06, 0.06, 0.08, 0.10, 0.29, 0.42])
RLF_z0p7_vanderVlught22_err_plus = np.array([0.10, 0.06, 0.06, 0.06, 0.08, 0.10, 0.19, 0.24])


# Redshift range 0.8<z<1.0 [z_mid = 0.9]---------------------------------------------------------------
# Data
LogL_z0p9_vanderVlught22 = np.array([22.31, 22.51, 22.69, 22.9, 23.15, 23.37, 23.76, 24.21])
RLF_z0p9_vanderVlught22  = np.array([-2.39, -2.4, -2.77, -2.92, -3.3, -3.7, -4.55, -4.27])

# errors
LogL_z0p9_vanderVlught22_err_plus = np.array([0.06, 0.10, 0.14, 0.18, 0.16, 0.18, 0.25, 0.04])
LogL_z0p9_vanderVlught22_err_minus  = np.array([0.18, 0.14, 0.09, 0.06, 0.07, 0.06, 0.22, 0.19])

RLF_z0p9_vanderVlught22_err_minus  = np.array([0.10, 0.06, 0.07, 0.07, 0.10, 0.23, 0.57, 0.57])
RLF_z0p9_vanderVlught22_err_plus = np.array([0.10, 0.06, 0.07, 0.07, 0.10, 0.16, 0.28, 0.28])


# Redshift range 1.0<z<1.3 [z_mid = 1.15]--------------------------------------------------------------
# Data
LogL_z1p15_vanderVlught22 = np.array([22.58, 22.78, 22.95, 23.15, 23.39, 23.51, 23.72, 23.95, 24.17])
RLF_z1p15_vanderVlught22  = np.array([-2.86, -2.64, -2.83, -3.22, -3.46, -3.88, -4.06, -4.36, -4.19])

# errors
LogL_z1p15_vanderVlught22_err_plus = np.array([0.07, 0.07, 0.10, 0.09, 0.05, 0.14, 0.12, 0.09,0.07])
LogL_z1p15_vanderVlught22_err_minus  = np.array([0.13, 0.13, 0.10, 0.11, 0.15, 0.06, 0.08, 0.11, 0.13])

RLF_z1p15_vanderVlught22_err_plus  = np.array([0.14, 0.07, 0.07, 0.09, 0.11, 0.17, 0.21, 0.28, 0.24])
RLF_z1p15_vanderVlught22_err_minus = np.array([0.14, 0.07, 0.07, 0.09, 0.11, 0.26, 0.34, 0.57, 0.42])


# Redshift range 1.3<z<1.6 [z_mid = 1.45]--------------------------------------------------------------
# Data
LogL_z1p45_vanderVlught22 = np.array([22.73, 22.87, 23.06, 23.25, 23.47, 23.62, 23.84, 23.95, 24.18])
RLF_z1p45_vanderVlught22  = np.array([-2.63, -2.71, -2.85, -3.12, -3.4, -3.6, -3.99, -3.93, -4.08])

# errors
LogL_z1p45_vanderVlught22_err_minus  = np.array([0.13, 0.08, 0.08, 0.09, 0.12, 0.08, 0.12, 0.03, 0.08])
LogL_z1p45_vanderVlught22_err_plus = np.array([0.06, 0.11, 0.11, 0.10, 0.07, 0.10, 0.07, 0.15, 0.11])

RLF_z1p45_vanderVlught22_err_minus  = np.array([0.12, 0.08, 0.07, 0.07, 0.09, 0.11, 0.26, 0.23, 0.29])
RLF_z1p45_vanderVlught22_err_plus = np.array([0.12, 0.08, 0.07, 0.07, 0.09, 0.11, 0.17, 0.16, 0.19])


# Redshift range 1.6<z<2.0 [z_mid = 1.8]---------------------------------------------------------------
# Data
LogL_z1p8_vanderVlught22 = np.array([22.99, 23.21, 23.43, 23.62, 23.83, 24.10, 24.40])
RLF_z1p8_vanderVlught22  = np.array([-2.86, -3.10, -3.31, -3.56, -3.93, -4.26, -4.97])

# errors
LogL_z1p8_vanderVlught22_err_minus  = np.array([0.15, 0.14, 0.13, 0.09, 0.07, 0.11, 0.18])
LogL_z1p8_vanderVlught22_err_plus = np.array([0.08, 0.09, 0.10, 0.14, 0.16, 0.12, 0.51])

RLF_z1p8_vanderVlught22_err_minus  = np.array([0.11, 0.09, 0.08, 0.09, 0.13, 0.29, 0.42])
RLF_z1p8_vanderVlught22_err_plus = np.array([0.11, 0.09, 0.08, 0.09, 0.13, 0.19, 0.24])


# Redshift range 2.0<z<2.5 [z_mid = 2.25]--------------------------------------------------------------
# Data
LogL_z2p25_vanderVlught22 = np.array([23.23, 23.39, 23.65, 23.83, 24.00, 24.21, 24.55])
RLF_z2p25_vanderVlught22  = np.array([-3.20, -3.29, -3.57, -3.80, -4.28, -4.28, -4.85])

# errors
LogL_z2p25_vanderVlught22_err_minus  = np.array([0.19, 0.13, 0.16, 0.12, 0.06, 0.05, 0.17])
LogL_z2p25_vanderVlught22_err_plus = np.array([0.03, 0.10, 0.06, 0.10, 0.16, 0.17, 0.50])

RLF_z2p25_vanderVlught22_err_minus  = np.array([0.20, 0.12, 0.11, 0.12, 0.29, 0.29, 0.34])
RLF_z2p25_vanderVlught22_err_plus = np.array([0.14, 0.12, 0.11, 0.12, 0.19, 0.19, 0.21])


# Redshift range 2.5<z<3.3 [z_mid = 2.9]---------------------------------------------------------------
# Data
LogL_z2p9_vanderVlught22 = np.array([23.53, 23.71, 23.92, 24.11, 24.31, 24.46, 24.70])
RLF_z2p9_vanderVlught22  = np.array([-3.52, -3.73, -3.95, -4.07, -4.39, -4.87, -5.04])

# errors
LogL_z2p9_vanderVlught22_err_minus  = np.array([0.13, 0.11, 0.13, 0.12, 0.12, 0.08, 0.13])
LogL_z2p9_vanderVlught22_err_plus = np.array([0.07, 0.09, 0.07, 0.08, 0.07, 0.12, 0.46])

RLF_z2p9_vanderVlught22_err_minus  = np.array([0.19, 0.13, 0.13, 0.13, 0.26, 0.57, 0.34])
RLF_z2p9_vanderVlught22_err_plus = np.array([0.14, 0.13, 0.13, 0.13, 0.17, 0.28, 0.21])


# Redshift range 3.3<z<4.6 [z_mid = 3.95]--------------------------------------------------------------
# Data
LogL_z3p95_vanderVlught22 = np.array([23.67, 24.10, 24.40])
RLF_z3p95_vanderVlught22  = np.array([-3.97, -4.50, -4.99])

# errors
LogL_z3p95_vanderVlught22_err_minus  = np.array([0.13, 0.19, 0.13])
LogL_z3p95_vanderVlught22_err_plus = np.array([0.23, 0.17, 0.59])

RLF_z3p95_vanderVlught22_err_minus  = np.array([0.34, 0.23, 0.21])
RLF_z3p95_vanderVlught22_err_plus = np.array([0.21, 0.16, 0.15])

####################################### Somolcic+09 ##################################################
# ================== 0.1 < z < 0.35 ==================
LogL_z0p25_Smol = np.log10(np.array([
    1.0e22, 2.0e22, 4.0e22, 8.9e22, 1.8e23, 3.2e23
]))

RLF_z0p25_Smol = np.log10(np.array([
    7.0e-4, 2.9e-4, 3.5e-4, 5.0e-5, 3.2e-5, 1.4e-5
]))

RLF_z0p25_Smol_err = np.array([
    1.5e-4/(7.0e-4*np.log(10)),
    0.7e-4/(2.9e-4*np.log(10)),
    0.6e-4/(3.5e-4*np.log(10)),
    2.1e-5/(5.0e-5*np.log(10)),
    1.6e-5/(3.2e-5*np.log(10)),
    1.0e-5/(1.4e-5*np.log(10))
])


# ================== 0.35 < z < 0.6 ==================
LogL_z0p5_Smol = np.log10(np.array([
    3.5e22, 6.3e22, 1.0e23, 1.6e23, 2.2e23, 3.5e23
]))

RLF_z0p5_Smol = np.log10(np.array([
    2.4e-4, 1.5e-4, 1.3e-4, 5.8e-5, 9.0e-6, 1.5e-6
]))

RLF_z0p5_Smol_err = np.array([
    0.9e-4/(2.4e-4*np.log(10)),
    0.4e-4/(1.5e-4*np.log(10)),
    0.3e-4/(1.3e-4*np.log(10)),
    1.6e-5/(5.8e-5*np.log(10)),
    6.4e-6/(9.0e-6*np.log(10)),
    0.7e-6/(1.5e-6*np.log(10))
])


# ================== 0.6 < z < 0.9 ==================
LogL_z0p75_Smol = np.log10(np.array([
    1.4e23, 2.8e23, 5.6e23, 1.1e24, 2.5e24, 5.0e24
]))

RLF_z0p75_Smol = np.log10(np.array([
    1.2e-5, 4.8e-5, 1.7e-5, 5.8e-6, 5.2e-6, 2.0e-6
]))

RLF_z0p75_Smol_err = np.array([
    0.2e-5/(1.2e-5*np.log(10)),
    0.9e-5/(4.8e-5*np.log(10)),
    0.5e-5/(1.7e-5*np.log(10)),
    2.6e-6/(5.8e-6*np.log(10)),
    2.3e-6/(5.2e-6*np.log(10)),
    1.4e-6/(2.0e-6*np.log(10))
])


# ================== 0.9 < z < 1.3 ==================
LogL_z1p1_Smol = np.log10(np.array([
    3.2e23, 6.3e23, 1.3e24, 2.8e24, 5.6e24, 1.0e25, 1.8e25
]))

RLF_z1p1_Smol = np.log10(np.array([
    9.2e-5, 2.4e-5, 1.8e-5, 3.2e-6, 1.2e-6, 2.0e-6, 1.0e-6
]))

RLF_z1p1_Smol_err = np.array([
    1.5e-5/(9.2e-5*np.log(10)),
    0.5e-5/(2.4e-5*np.log(10)),
    0.4e-5/(1.8e-5*np.log(10)),
    1.4e-6/(3.2e-6*np.log(10)),
    0.8e-6/(1.2e-6*np.log(10)),
    1.0e-6/(2.0e-6*np.log(10)),
    0.7e-6/(1.0e-6*np.log(10))
])
