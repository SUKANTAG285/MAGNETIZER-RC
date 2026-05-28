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

import Bell2003_data as Bell03


# Constants
K = 0.81  # rad m^−2 cm^3 μG^−1 pc^−1
L_sun = 3.828*10**26 # W Hz^-1
# unit conversions :
kpc_cm = 3.086 * 10**21   # cm
Mpc_cm = 3.086 * 10**24   # cm
Mpc_m  = 3.086*  10**22   # m
cm_km  = 10**(-5)         # km
GHz_Hz = 10**9            # Hz
muG_G  = 10**(-6)         # muG
GeV_erg= 0.00160218       # GeV
Jy_mJy = 10**3            # mJy
Jy_cgs = 10**(-23)        # erg s^-1 cm^-2 Hz^-1
mJy_si  = 10**(-29)       # W m^-2 Hz^-1

# Update font size for plots
plt.rcParams.update({'font.size': 24})
rcParams['font.family'] = 'serif'

# Initialize DataFrame to hold data
data_sam = pd.DataFrame()

# Redshift for which we need to plot the PDF
redshift = 0.001

# Galform_model name:
galform_model = 'Lacey14_new'

# Magnetizer_model name:
# mag_model      ="G24_fitted_median"#"SG_R19"#_floor_seed_frac0.01"#"CJ24_R19seed.frac_seed0.0001"# "CJ24_R19seed"
# mag_model = "CJ24_total_gas_model6"
# mag_model     = "CJ24fiducial"#_Bbar_contb"#"CJ24_total_gas_model6"
# mag_model ="CJ24_Rmol0.0_Rk1.5_grid_2x"#'CJ24_const_vt'#
mag_model      ="G25_fb0.8_Rk0.3_vtmod2_no_alp_sq_SFRD_L16_Fiducial_mod1"#_2h"
# mag_model      ="CJ24_Rmol0.0_Rk1.0_fb1.0"


# The PDFs are constructed using these file numbers
filenos = range(1,6,1)#np.r_[1:11, 13:20]

# Loop through each file number to read data
for fno in filenos:
    # Read the data from the corresponding .txt file
#     filename = f"/media/sukanta/New Volume1/New folder/magnetizer_cj24/Bprops/props_PI_pc_{galform_model}_{mag_model}_fno{fno}_z{redshift:.1f}.txt"
    filename = f"/media/sukanta/New Volume1/New folder/magnetizer_cj24/Bprops/props_pola_trial_total_sfr_L16_mod1_{galform_model}_{mag_model}_fno{fno}_z{redshift:.1f}.txt"
#     filename = f"/media/sukanta/New Volume1/New folder/magnetizer_cj24/Bprops/props_ani_all_{galform_model}_{mag_model}_fno{fno}_z{redshift:.1f}.txt"
#     filename = f"/media/sukanta/New Volume1/New folder/magnetizer_cj24/Bprops/props_fiducial_fast_adaptive_fbmedian_{galform_model}_{mag_model}_fno{fno}_z{redshift:.1f}.txt"

    df = pd.read_csv(filename, header=None, sep=' ')
    # Define column names
#     df.columns = ['galid','rmax','Bmax','Bavg','B_rms','Beq_rms','B_total_avg','B_bar_avg', 
#                   'Mhalo', 'Mstar','Mbulge', 'SFR_new', 'Mgas','M_diffuse_n','num_density','S_I','Lum','R_25','Ur_avg']
#     galid,rmax,Bmax,Bavg,B_rms,Beq_rms,B_total_avg,B_bar_avg, \
#               Mhalo, Mstar,Mbulge, SFR_new, Mgas,M_diffuse_n,num_density,S_I,Lum,R_25,Ur_avg

    df.columns = ['galid','rmax','Bmax','Bavg','B_rms','Beq_rms','B_total_avg','B_bar_avg', 
                  'Mhalo', 'Mstar','Mbulge', 'SFR_new', 'Mgas','M_diffuse_n','num_density','S_I','Lum','R_25','Ur_avg','v_t']
#     df.columns = ['galid','rmax','Bmax','Bavg','B_rms','Beq_rms','B_total_avg','B_bar_avg', 
#                   'Mhalo', 'Mstar','Mbulge', 'SFR_new', 'Mgas','M_diffuse_n','num_density','S_I','Lum_I','S_PI','Lum_PI','R_25','Ur_avg','ang']
                


    df['redshift'] = redshift
    
    # Concatenate the data
    data_sam = pd.concat([data_sam, df])

# Applying Filters
# Mstar filter
apply_Mstar_cutoff = True
Mstar_cutoff_value = 10**(9) # M_star

# sSFR filter 
apply_sSFR_cutoff = False
sSFR_cutoff_value = -10.5 # M_sun/yr (-10.5 for z = 0 & -9.3 for z = 1.5)

# SFR filter 
apply_SFR_cutoff =True
SFR_cutoff_value = 0.001

# Luminosity filter (needed as Observed data have selection condition)
apply_L_cutoff = False

L_cutoff_value = 16

# Bulge to Total stellar mass ratio (BT) filter
apply_BT_cutoff = True
BT_cutoff_value = 0.4

# R_25 radius filter
apply_R_25_cutoff = False
R_25_cutoff_value = 6 # kpc


# Condition for filtering the data
condition = np.ones(len(data_sam), dtype=bool)

# Applying the filters

if apply_Mstar_cutoff:
    condition &= (data_sam['Mstar'] > Mstar_cutoff_value)

if apply_sSFR_cutoff:
    condition &= (np.log10(data_sam['SFR_new']/data_sam['Mstar']) > sSFR_cutoff_value)
    
if apply_SFR_cutoff:
    condition &= ((data_sam['SFR_new']) > SFR_cutoff_value)
    
if apply_L_cutoff:
    condition &= (np.log10(data_sam['Lum']*10**(-7)) > L_cutoff_value)
     
if apply_BT_cutoff:
    condition &= (data_sam['Mbulge']/data_sam['Mstar']< BT_cutoff_value )    

if apply_R_25_cutoff:
    condition &= (data_sam['R_25'] > R_25_cutoff_value)
    

# Apply the combined filter
data_sam = data_sam[condition]

# Plotting
fig = plt.figure(figsize=(30,12))
gs = GridSpec(2, 3, width_ratios=[7, 1, 7], height_ratios=[1.5, 7])  # Adjust for one redshift
fig.subplots_adjust(hspace=0.0, wspace=0.0)
ax = fig.add_subplot(gs[1, 0])

# Scatter Plot for the selected redshift
data_sam_z = data_sam[data_sam['redshift'] == redshift]
df1 = pd.DataFrame(data_sam_z)
# Sample 1000 galaxies from the filtered data
data_z = df1#.sample(40000)


# PLOT VARIABLES :

# Galform Parameters :
Mstar       = np.array((data_z['Mstar']))
bulge_mass  = np.array((data_z['Mbulge']))
Mgas   = (np.array((data_z['Mgas'])))
#     Mdiffuse_P  = (np.array(data_z['M_diffuse_P']))
Mdiffuse_n  = (np.array(data_z['M_diffuse_n']))
# Mhalo       = np.array((data_z['Mhalo']))
SFR         = data_z['SFR_new']
B_T_ratio   = bulge_mass/Mstar
sSFR        = SFR/Mstar
print(np.max(sSFR))

# Magnetizer Parameters :
B_0         = (data_z['B_rms'])
B_total_avg = (data_z['B_total_avg'])
B_bar_avg   = (data_z['B_bar_avg'])
# v_t  = (data_z['v_t'])
#     pitch_avg   = -(data_z['pitch_avg'])
#     deg_pol     = data_z['deg_pol']
S_I       = data_z['S_I']*((1.4)**(-1))/((4.8)**(-1))
# S_I_los         = data_z['S_I_los']
L           = data_z['Lum']*10**(-7)*((1.4)**(-1))/((4.8)**(-1)) # unit : W Hz^(-1)
v_rot_avg   = data_z['Ur_avg']
R_25        = data_z['R_25']
L_FIR       = SFR/(0.79*1.7*10**(-10)) # unit: L_sun
alp         = 0.71
IRX0        = 1.32
IRX         = alp*(np.log10(Mstar)-10.35) + IRX0
K_IR        = 1.7 * 10**(-10) # M_sun yr^−1 L^−1
K_UV        = 2.8 * 10**(-10) # M_sun yr^−1 L^−1
L_FIR2      = (SFR/K_UV) * 10**IRX/(1+(K_IR/K_UV)*10**IRX) 




# Observational data 

# Data from Taba+17
taba17 = np.genfromtxt("galaxies_data_T17_1.4.txt", delimiter='\t', names=True, dtype=None, encoding='utf-8')

values        = (taba17['SI'].astype(float))*mJy_si*4*np.pi*(10*Mpc_m)**2
uncertainties = (taba17['SI_err'].astype(float))*mJy_si*4*np.pi*(10*Mpc_m)**2
sfr_obs       = taba17['sfr'].astype(float)
sfr_err_obs   = taba17['sfr_err'].astype(float)
log_Mstar_obs = taba17['Mstar'].astype(float)
sSFR_obs  = sfr_obs/10**(log_Mstar_obs)
h_type    = taba17['hubble_type']


sfr_obs2 = np.array([4.94,0.31,4.13, 7, 2.45,0.39,3.46,2.17,1.7, 0.60,1.77, 2.61, 0.29, 2.36,
                0.90,9.20, 0.30])
log_Mstar_obs2 = np.array([10.33,10.72,10.74, 10.95,10.17,10.74, 10.47, 10.21, 10.49, 10.75, 9.46,
                  10.30, 10.00, 10.6, 11.03, 10.52, 9.68])
logL_obs2 = np.array([22.02, 21.81, 22.11, 22.29, 21.21, 20.85, 22.53, 21.66, 21.43, 21.59,21.01,
                  22.04, 21.51, 21.96, 22.12, 22.17, 20.37])
sSFR_obs2  = sfr_obs2/10**(log_Mstar_obs2)

sfr_obs3    = Bell03.SFR_B03
logL_obs_total   = Bell03.L_1_4GHz
f_th_1p4GHz = 0.095
logL_obs3 = np.log10(10**logL_obs_total - f_th_1p4GHz * 10**logL_obs_total)

# taba16 = np.genfromtxt("galaxies_data_T16.txt", delimiter='\t', names=True, dtype=None, encoding='utf-8')

# values_T16 = taba16['SI'].astype(float)#np.array([108, 270, 420, 111, 88, 27, 90, 264, 359, 744, 28, 4, 232, 250, 57, 129, 546, 44, 379, 9, 9.06, 7.6, 0.9, 0.88, 0.37, 0.07])#,119,105,14.99,42.48,4.45])
# uncertainties_T16 = taba16['SI_err'].astype(float)#np.array([9, 55, 25, 20, 10, 6, 7, 45, 30, 47, 2, 1, 13, 14, 4, 8, 87, 8, 46, 1, 0.95, 1.5, 0.2, 0.15, 0.07, 0.01])#,2,8,1,2,1])
# sfr_obs_T16 = taba16['sfr'].astype(float)#np.array([0.59, 1.15, 4.94, 0.6, 0.36, 0.37, 0.77, 0.31, 4.13, 7, 0.25, 0.14, 2.1, 1.2, 0.7, 2.2, 9.2, 2.8, 0.9, 0.6, 0.30, 0.26, 0.046, 0.06, 0.02, 0.003])#,1.30,1.04,0.26,0.44,0.16])
# sfr_err_obs_T16 = 0.25 * sfr_obs_T16
# log_Mstar_obs_T16 = taba16['Mstar'].astype(float)
# sSFR_obs_T16  = sfr_obs_T16/10**(log_Mstar_obs_T16)
# h_type_T16    = taba16['hubble_type']



mask17 = (h_type != 'Im') & (h_type != 'IBm') & (h_type != 'IABm') & (h_type != 'E') & (h_type != 'S0')& (h_type != 'SA0')# & (np.log10(sSFR_obs)> -10.5)
# mask16 = (h_type_T16 != 'Irr')  # & (np.log10(sSFR_obs_T16)> -10.5)

# plt.hist(log_Mstar_obs[mask17])

# Calculate global sSFR_min and sSFR_max
sSFR_min = min(np.log10(sSFR_obs).min(), np.log10(sSFR).min(), np.log10(sSFR_obs2).min())
sSFR_max = max(np.log10(sSFR_obs).max(), np.log10(sSFR).max(), np.log10(sSFR_obs2).max())
norm = mcolors.Normalize(vmin=sSFR_min, vmax=sSFR_max)  # Define norm once



# Define x and y variables and color variable for the scatter plot
x_var = np.log10(SFR)
y_var = np.log10(L)
# y_var_1 = np.log10(S_I_los)
colour_var = np.log10(sSFR)

# Calculate standard deviation
SI_std = np.std(S_I)
SFR_std = np.std(SFR)

# print(r"$\sigma_{S_I}$:", SI_std)
# print(r"$\sigma_{SFR}$", SFR_std)
# Combine your arrays into one list of indices and shuffle them
indices = np.arange(len(x_var))
np.random.shuffle(indices)

# Shuffle the arrays
x_shuffled = np.array(x_var)[indices]
y_shuffled = np.array(y_var)[indices]
colour_shuffled = np.array(colour_var)[indices]

# Now plot in shuffled order
sc = ax.scatter(x=x_shuffled, y=y_shuffled, c=colour_shuffled, alpha=1.0, s=15,
                cmap='rainbow', norm=norm)

# Create scatter plot
# sc = ax.scatter(x=x_var, y=y_var, c=colour_var, alpha=1.0, s=20, cmap="rainbow",norm=norm)
# ax.scatter(x=x_var, y=y_var_1, c=colour_var, alpha=1.0, s=50, cmap="viridis")

# Set axis limits ------------------------------------------------------------------------
x_min, x_max = -3, 2
y_min, y_max = 17, 25
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(np.arange(-3, 2, 1))
ax.set_yticks(np.arange(17, 25, 1))

# Filter data within axis limits for correlation and fitting
filter_data = (x_var >= x_min) & (x_var <= x_max) & (y_var >= y_min) & (y_var <= y_max)
x_var_filter = x_var[filter_data]
y_var_filter = y_var[filter_data]

# -------------------------add contoures---------------------------------------------------------------------------
from scipy.stats import gaussian_kde

# 2D KDE
xy = np.vstack([x_var_filter, y_var_filter])
kde = gaussian_kde(xy)

# Grid for evaluation
x_grid = np.linspace(x_min, x_max, 100)
y_grid = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
# Z = Z1/np.max(Z1)
# levels = [0.005, 0.01, 0.02, 0.05, 0.1,0.5]

Z_flat = Z.flatten()
Z_sort = np.sort(Z_flat)[::-1]   # sort descending
cumsum = np.cumsum(Z_sort)
cumsum /= cumsum[-1]             # normalize to 1

# Find KDE values at confidence levels
def find_level(prob):
    idx = np.searchsorted(cumsum, prob)
    return Z_sort[idx]

level_05sig = find_level(0.1175)
level_1sig = find_level(0.3935)
level_15sig = find_level(0.6827)
level_2sig = find_level(0.8647)
level_3sig = find_level(0.9889)

# levels = np.sort([level_05sig, level_1sig,level_15sig, level_2sig, level_3sig])
levels = np.sort([level_1sig, level_2sig, level_3sig])

# levels = [0.005, 0.01, 0.02, 0.05, 0.1,0.3,0.5]
# Contour lines
contour = ax.contour(
    X, Y, Z,
    levels=levels,
    colors='black',
    linewidths=2.5,
    linestyles='dotted',
    alpha=1
)

# Add contour labels-------------------------------------
# fmt = {level: label for level, label in zip(levels, [r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])}
fmt = {level: label for level, label in zip(levels, [r'0.99', r'0.86', r'0.39'])}
# fmt = {level: f"{level}" for level in levels}

ax.clabel(contour, inline=True, fontsize=15, fmt=fmt)

# ax.clabel(contour, inline=True, fontsize=8, fmt="%.2e")  # or fmt="%.3f" for floats

# # KDE plot with only contour lines (no fill)
# sns.kdeplot(
#     x=x_var,
#     y=y_var,
#     ax=ax,
#     fill=False,          # Only contours
#     levels=7,           # Number of contour lines
#     color = 'black',
# #     cmap = 'viridis',       # Line color
#     linewidths=1.0,      # Line thickness
#     linestyles="--",
#     alpha = 0.54
# )


#--------------------------------------------------------------------------------   
#  # Create a twin y-axis sharing the same x-axis for variable L
# ax2 = ax.twinx()

# # Scatter plot for the second y-axis (L)
# y_var_2 = np.log10(S_I)

# # ax2.set_ylabel(r'$\log\left(L_{\rm{I}}^{\rm{nt}} \,[\rm{W} \,\rm{Hz}^{-1}]\right)$', color='blue')  # Assuming L is in Jy units
# ax2.set_ylabel(r'$\log\left( S_{\rm{I}}\,[\mathrm{mJy}]\right)$', color='blue')  # Assuming L is in Jy units

# # Adjust the tick parameters for the second y-axis
# ax2.tick_params(axis='y', labelcolor='blue',which='both', length=12, width=2)
# ax2.tick_params(axis='y', which='minor', length=5, width=1.5)  # Minor ticks


# # Optional: adjust the y-limits for the second y-axis
# y_max2 = (10**y_max)/(mJy_si*4*np.pi*(10*Mpc_m)**2)
# y_min2 = (10**y_min)/(mJy_si*4*np.pi*(10*Mpc_m)**2)
# ax2.set_ylim(np.log10(y_min2), np.log10(y_max2))
# # Enable minor ticks
# ax2.minorticks_on()

# # Set major and minor ticks for y-axis
# ax2.tick_params(axis='y', which='both', length=12, width=2, direction='out')  # Major ticks
# ax2.tick_params(axis='y', which='minor', length=5, width=1.5, direction='out')  # Minor ticks

# ax2.set_yticks(np.arange(-5, 9, 2))

#--------------------------------------------------------------------------------
# # Create a twin x-axis sharing the same y-axis
# ax3 = ax.twiny()  # This creates a second x-axis

# # Set the second x-axis variables
# x_var_2 = np.log10(L_FIR)

# # Set the label for the second x-axis
# ax3.set_xlabel(r'$\log\left(L_{8-1000 \,\mu \rm{m}} \,[L_{\odot}]\right)$', color='red')

# # Adjust the tick parameters for the second x-axis
# ax3.tick_params(axis='x', labelcolor='red', which='both', length=12, width=2)
# ax3.tick_params(axis='x', which='minor', length=5, width=1.5)  # Minor ticks

# # Optional: adjust the x-limits for the second x-axis
# x_min2 = 10**(x_min) / (0.79 * 1.7 * 10**(-10))
# x_max2 = 10**(x_max) / (0.79 * 1.7 * 10**(-10))

# ax3.set_xlim(np.log10(x_min2), np.log10(x_max2))

#------------------------------------------------------------------------------------------  
    


# Calculate Pearson's and Spearman's correlation coefficients
corr_coeff, p_value = pearsonr(x_var_filter, y_var_filter)
sper_corr, sper_p_value = spearmanr(x_var_filter, y_var_filter)

# Calculate standard errors for correlation coefficients
n = len(x_var_filter)
pearson_se = np.sqrt((1 - corr_coeff**2) / (n - 2))
spearman_se = 1 / np.sqrt(n - 1)

# Define the linear fitting function
def func(x, a, b):
    return a * x + b

# Perform curve fitting
popt, pcov = curve_fit(func, x_var_filter, y_var_filter)
slope, intercept = popt
slope_err, intercept_err = np.sqrt(np.diag(pcov))

x0 = -1
L_cut_model = func(x0, slope, intercept)

# Propagate error
cov_ab = pcov[0, 1]
L_cut_err = np.sqrt((x0**2) * slope_err**2 + intercept_err**2 + 2 * x0 * cov_ab)

print(f"L_cut = {L_cut_model:.2f} ± {L_cut_err:.2f}")

# Generate x values for the fitted line
x_line = np.linspace(min(x_var_filter), max(x_var_filter), 100)
# Calculate y values using the fitted parameters
y_line = func(x_line, slope, intercept)


# Plot the fitted line
ax.plot(x_line, y_line, '--', color='green', linewidth=2,alpha= 1, label='Fit to "Fiducial"',zorder=1)

# Calculate medians and percentiles for the bins
bins = np.linspace(x_var_filter.min(), x_var_filter.max(), 10)
median_x = np.zeros(len(bins) - 1)
median_y = np.zeros(len(bins) - 1)
per85_x = np.zeros(len(bins) - 1)
per15_x = np.zeros(len(bins) - 1)
per85_y = np.zeros(len(bins) - 1)
per15_y = np.zeros(len(bins) - 1)
for j in range(len(bins) - 1):
    mask = (x_var_filter >= bins[j]) & (x_var_filter < bins[j + 1])
    median_x[j] = np.percentile(x_var_filter[mask], 50)
    per85_x[j] = np.percentile(x_var_filter[mask], 95)
    per15_x[j] = np.percentile(x_var_filter[mask], 5)
    median_y[j] = np.percentile(y_var_filter[mask], 50)
    per85_y[j] = np.percentile(y_var_filter[mask], 95)
    per15_y[j] = np.percentile(y_var_filter[mask], 5)
    
#Fit a linear model (1st-degree polynomial) to your percentile data
slope85, intercept85 = np.polyfit(per85_x, per85_y, 1)
slope15, intercept15 = np.polyfit(per15_x, per15_y, 1)

#Generate fitted y-values
per85_fit_y = slope85 * np.array(per85_x) + intercept85
per15_fit_y = slope15 * np.array(per15_x) + intercept15

#Plot the fitted lines
# ax.plot(per85_x, per85_y, color='red', linestyle='dotted', linewidth=2)
# ax.plot(per15_x, per15_y, color='red', linestyle='dotted', linewidth=2)


# Plot the median line
# ax.plot(median_x, median_y, color='blue', linestyle='--', linewidth=3, label='median')

# Add text annotations for redshift, correlation coefficients, and slope
ax.text(0.98, 0.92, r'(b)', ha='right', color='black',fontsize=35, transform=ax.transAxes)#,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
ax.text(0.96, 0.30, f'z=0.0', ha='right', color='red', transform=ax.transAxes,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
# ax.text(0.96, 0.35, f'$r_{{\mathrm{{sp}}}}={sper_corr:.2f}\pm {spearman_se:.2f}$', ha='right',color='green', transform=ax.transAxes,fontsize=20)
# ax.text(0.96, 0.30, f'$r_{{\mathrm{{p}}}}={corr_coeff:.2f}\pm {pearson_se:.2f}$', ha='right',color='green', transform=ax.transAxes,fontsize=20)
# ax.text(0.96, 0.25, f'Slope$ ={a:.2f} \pm {slope_error:.2f}$', ha='right', color='green',transform=ax.transAxes,fontsize=20)

# Customize tick parameters
# Enable minor ticks
ax.minorticks_on()

# Set major and minor ticks for x-axis
ax.tick_params(axis='x', which='both', bottom=True, top=True, length=12, width=2, direction='in')  # Major ticks
ax.tick_params(axis='x', which='minor', length=5, width=1.5, direction='in')  # Minor ticks

# Set major and minor ticks for y-axis
ax.tick_params(axis='y', which='both', length=12, width=2, direction='in')  # Major ticks
ax.tick_params(axis='y', which='minor', length=5, width=1.5, direction='in')  # Minor ticks

####################################################################
# Plotting Taba+16 data points for z = 0.001 -----------------------------------------------------------------------

# Log transformation
log_sfr = np.log10(sfr_obs)
log_sfr_err = sfr_err_obs/(sfr_obs*np.log(10))
log_values = np.log10(values)
log_uncertainties = uncertainties / (values * np.log(10))  # relative error in log 

log_sfr2 = np.log10(sfr_obs2)
log_sfr3 = np.log10(sfr_obs3)

# log_sfr_err2 = sfr_err_obs/(sfr_obs*np.log(10))
log_values2 = logL_obs2
# log_uncertainties = uncertainties / (values * np.log(10))  # relative error in log 

# mask = 10**(log_sfr)> 0.01 # as in T+17, spiral galaxies have SFR > 0.01

log_values_obs_all = (np.log10(values[mask17]))# , np.log10(values_T16[mask16]))
log_values_obs_err_all = (uncertainties[mask17] / (values[mask17] * np.log(10)))#,
                                  #uncertainties_T16[mask16] / (values_T16[mask16] * np.log(10)))

log_sfr_obs_all = (np.log10(sfr_obs[mask17]))#, np.log10(sfr_obs_T16[mask16]))
log_sfr_obs_err_all = (sfr_err_obs[mask17]/(sfr_obs[mask17]*np.log(10)))#,
                                #sfr_err_obs_T16[mask16]/(sfr_obs_T16[mask16]*np.log(10)))

log_values_obs_total = np.concatenate([
    np.log10(values[mask17]),
    logL_obs2,logL_obs3
])

log_sfr_obs_total = np.concatenate([
    np.log10(sfr_obs[mask17]),
    np.log10(sfr_obs2), np.log10(sfr_obs3)
])
# Calculate Pearson's and Spearman's correlation coefficients
corr_coeff_obs, p_value_obs = pearsonr(log_sfr_obs_total, log_values_obs_total)
sper_corr_obs, sper_p_value_obs = spearmanr(log_sfr_obs_total, log_values_obs_total)

# Calculate standard errors for correlation coefficients
n = len(log_sfr_obs_total)
pearson_se_obs = np.sqrt((1 - corr_coeff_obs**2) / (n - 2))
spearman_se_obs = 1 / np.sqrt(n - 1)

# ax.text(0.96, 0.17, f'$r_{{\mathrm{{sp}}}}={sper_corr:.2f}\pm {spearman_se:.2f}$', ha='right',color='blue', transform=ax.transAxes,fontsize=20)
# ax.text(0.96, 0.12, f'$r_{{\mathrm{{p}}}}={corr_coeff:.2f}\pm {pearson_se:.2f}$', ha='right',color='blue', transform=ax.transAxes,fontsize=20)

        

# Plot data with error bars (spiral galaxies)
ax.scatter(x=log_sfr3, y=logL_obs3, c='white', alpha=1.0, s=50,marker='P',edgecolors='black',label="Spiral (Bell+2003)",zorder=4)

ax.errorbar(log_sfr[mask17], log_values[mask17],xerr =log_sfr_err[mask17],yerr=log_uncertainties[mask17], fmt=' ',markersize=8, label='T17',color='black', ecolor='black', capsize=4, capthick=1, alpha=0.8,zorder=2)

sc2=ax.scatter(x=log_sfr[mask17], y=log_values[mask17], c=np.log10(sSFR_obs[mask17]), alpha=1.0, s=80, cmap="rainbow",norm=norm,marker='h',edgecolors='black', label="Spiral (T17)",zorder=3)
ax.scatter(x=log_sfr2, y=logL_obs2, c=np.log10(sSFR_obs2), alpha=1.0, s=80, cmap="rainbow",norm=norm,marker='s',edgecolors='black', label="Spiral (Liu+2015)",zorder=4)

# ax.errorbar(np.log10(sfr_obs_T16),np.log10(values_T16) ,xerr = sfr_err_obs_T16/(sfr_obs_T16*np.log(10)),
#             yerr=uncertainties_T16 / (values_T16 * np.log(10)),
#             fmt=' ',markersize=8, label='T+16',color='black', ecolor='black', capsize=4, capthick=1, alpha=0.8)
# sc3 = ax.scatter(x = np.log10(sfr_obs_T16[mask16]), y = np.log10(values_T16[mask16]), c = np.log10(sSFR_obs_T16[mask16]),
#           alpha=1.0, s=100, cmap="rainbow", norm=norm, marker='P', edgecolors='black', label="Spiral (T+16)",zorder=4)



# Plot Irr-galaxies with a different marker (Irregular galaxies)
ax.errorbar(log_sfr[~mask17],log_values[~mask17],xerr =log_sfr_err[~mask17], fmt=' ',markersize=8,color='black', ecolor='black', capsize=4, capthick=1, alpha=0.8)
ax.scatter(x=log_sfr[~mask17], y=log_values[~mask17], c=np.log10(sSFR_obs[~mask17]), 
           alpha=1.0, s=100, cmap="rainbow", norm=norm, marker='v', edgecolors='black', label="Irr/E/S0/SA0 (T17)",zorder=5)

# ax.scatter(x = np.log10(sfr_obs_T16[~mask16]), y = np.log10(values_T16[~mask16]), c = np.log10(sSFR_obs_T16[~mask16]),
#           alpha=1.0, s=100, cmap="rainbow", norm=norm, marker='*', edgecolors='black', label="Irregular (T+16)",zorder=6)





# Define linear function for fitting
def linear_model(x, a, b):
    return a * x + b



# Perform the weighted least squares fit
popt_obs, pcov_obs = curve_fit(linear_model, log_sfr_obs_total, log_values_obs_total)#, sigma=log_sfr_err)
slope_obs, intercept_obs = popt_obs
slope_err_obs = np.sqrt(pcov_obs[0, 0])
intercept_err_obs = np.sqrt(pcov_obs[1, 1])


# x0 = -2
L_cut_obs = func(x0, slope_obs, intercept_obs)

# Propagate error
cov_ab_obs = pcov_obs[0, 1]
L_cut_err_obs = np.sqrt((x0**2) * slope_err_obs**2 + intercept_err_obs**2 + 2 * x0 * cov_ab_obs)

print(f"L_cut_obs = {L_cut_obs:.2f} ± {L_cut_err_obs:.2f}")


# Create the line fit
x_fit = np.linspace(min(log_sfr_obs_total), max(log_sfr_obs_total), 100)
y_fit = linear_model(x_fit, slope_obs, intercept_obs)

# Plot the fit line
ax.plot(x_fit, y_fit, '--', color='blue',lw=2,alpha= 1,label='Fit to "Spiral (T17, L15)"',zorder=7 )

# Display the fit parameters with uncertainties
# ax.text(0.96, 0.07,     f'Slope = {slope:.2f} ± {slope_err:.2f}',#\nintercept (Taba+17) = {intercept:.2f} ± {intercept_err:.2f}
#  ha='right', transform=ax.transAxes, color='blue')

############################### Information Table ##################################################################
# # Define table data
# table_data = [
#     [r"$r_{\mathrm{sp}}$", r"$r_{\mathrm{p}}$", "Slope"],
#     [f"{sper_corr:.2f} ", f"{corr_coeff:.2f} ", f"{slope:.2f}"],
#     [f"{sper_corr_obs:.2f} ± {spearman_se_obs:.2f}", f"{corr_coeff_obs:.2f} ± {pearson_se_obs:.2f}", f"{slope_obs:.2f} ± {slope_err_obs:.2f}"]
# ]

# # Create table in the plot
# table = ax.table(cellText=table_data,
#                  colLabels=None,
#                  loc="bottom right",
#                  cellLoc="center",
#                  bbox=[0.55, 0.04, 0.44, 0.2],
#                  cellColours=[['white'] * len(table_data[0])] * len(table_data)) # bbox=[x, y, width, height]


# # Adjust font size
# table.auto_set_font_size(False)
# table.set_fontsize(15)  # Increase or decrease this value

# # Scale the table
# table.scale(1.2, 1.2)  # Adjust width and height scaling factors

# # # Change row colors
# # row_colors = ["#d3d3d3", "#90ee90", "#add8e6"]  # Gray (header), Light Green, Light Blue
# # for i, color in enumerate(row_colors):
# #     for j in range(len(table_data[i])):
# #         table[i, j].set_facecolor(color)

        
        
# # Set font colors
# font_colors = ["black", "green", "blue"]  # Header (black), Row 2 (green), Row 3 (blue)
# for i, font_color in enumerate(font_colors):
#     for j in range(len(table_data[i])):
#         table[i, j].get_text().set_color(font_color)
# Define table data
table_data = [
    [r"$r_{\mathrm{sp}}$", r"$r_{\mathrm{p}}$", r"$\gamma$",r"$\log\left(L_{0,1.4}\right)$"],
    [f"{sper_corr:.2f} ", f"{corr_coeff:.2f} ", f"{slope:.2f}", f"{L_cut_model:.2f}"],
    [f"{sper_corr_obs:.2f} ± {spearman_se_obs:.2f}", f"{corr_coeff_obs:.2f} ± {pearson_se_obs:.2f}", f"{slope_obs:.2f} ± {slope_err_obs:.2f}",f"{L_cut_obs:.2f} ± {L_cut_err_obs:.2f}"]
]

# Create table in the plot
table = ax.table(cellText=table_data,
                 colLabels=None,
                 loc="bottom right",
                 cellLoc="center",
                 bbox=[0.37, 0.032, 0.62, 0.18],
                 alpha=0.6,  # Table container alpha
                 zorder=8)


# Adjust font size
table.auto_set_font_size(False)
table.set_fontsize(17)  # Increase or decrease this value

# Scale the table
table.scale(1.2, 1.2)  # Adjust width and height scaling factors

# # Change row colors
# row_colors = ["#d3d3d3", "#90ee90", "#add8e6"]  # Gray (header), Light Green, Light Blue
# for i, color in enumerate(row_colors):
#     for j in range(len(table_data[i])):
#         table[i, j].set_facecolor(color)

        
        
# Set font colors
font_colors = ["black", "green", "blue"]  # Header (black), Row 2 (green), Row 3 (blue)
for i, font_color in enumerate(font_colors):
    for j in range(len(table_data[i])):
        table[i, j].get_text().set_color(font_color)
        
# Ensure the table background is visible
for key, cell in table.get_celld().items():
    cell.set_facecolor("white")  # Opaque background
    cell.set_alpha(0.5)  # Slight transparency

# Prevent the table from being clipped
table.set_clip_on(False)

####################################### Histogram plots #######################################################
#---------------------------------------------------------
# function for minimum bins for observed data:

def min_no_bins(min_points_per_bin, data, max_bin = 100):
    bin_ini = 0
    count_array = []
    # loops over trial bins to check minimum data counts per bin
    for m in range(1,max_bin): 
        bin_trial = bin_ini + m
        count,bins = np.histogram(data, bins = bin_trial)
        # condition for minimum data count
        if np.min(count) > min_points_per_bin:
            # append all possible num of bins
            count_array.append(len(count))
    # maxmum num of bins possible with that minimum counts        
    max_bins_counts = np.max(count_array)
    return max_bins_counts

#--------------------------------------------------------

# Y-axis marginal plot (right side)
ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax)

# Remove inf, -inf, and NaN values
y_var_clean = y_var[np.isfinite(y_var)]

counts_model_y, bins_model_y = np.histogram(y_var_clean, bins=200)
counts_model_y = counts_model_y / counts_model_y.max()
bin_centers_model_y = 0.5 * (bins_model_y[:-1] + bins_model_y[1:])

ax_marg_y.barh(
    bin_centers_model_y,
    counts_model_y,
    height=np.diff(bins_model_y),
    color='green',
    alpha=0.5
)
        
    
bin1 = min_no_bins(3, log_values_obs_all)
bins_counts = np.max([bin1,5])


counts_obs_y, bins_obs_y = np.histogram(log_values_obs_total, bins=bins_counts)
# print(counts_obs_y)
counts_obs_y = counts_obs_y / counts_obs_y.max()
bin_centers_obs_y = 0.5 * (bins_obs_y[:-1] + bins_obs_y[1:])

ax_marg_y.barh(
    bin_centers_obs_y,
    counts_obs_y,
    height=np.diff(bins_obs_y),
    color='blue',
    alpha=0.5
)

# Match y-limits with the main plot
ax_marg_y.set_ylim(ax.get_ylim())

# Set x-label to indicate normalized counts
ax_marg_y.set_xlabel("Norm Counts", fontsize=18)

# Set x-ticks from 0 to 1 (normalized)
ax_marg_y.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_marg_y.set_xticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=13)

# Optionally add legend
# ax_marg_y.legend(loc="upper right", fontsize=12, frameon=True)
# Show ticks on both left and right sides
ax_marg_y.xaxis.set_ticks_position('both')
ax_marg_y.tick_params(axis='x', length=5, direction='in')

# Remove y-axis (only keep density axis visible)
ax_marg_y.yaxis.set_visible(False)



# X-axis marginal plot (top side)
ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax)

# Compute normalized counts for model data
counts_model, bins_model = np.histogram(x_var, bins=200)
counts_model = counts_model / counts_model.max()  # normalize by max count
bin_centers_model = 0.5 * (bins_model[:-1] + bins_model[1:])

ax_marg_x.bar(
    bin_centers_model,
    counts_model,
    width=np.diff(bins_model),
    color='green',
    alpha=0.5,
    label='model_data'
)

bin2 = min_no_bins(3, log_sfr_obs_all)
bins_counts2 = np.max([bin2,5])

# Compute normalized counts for observed data
counts_obs, bins_obs = np.histogram(log_sfr_obs_total, bins=bins_counts2)
counts_obs = counts_obs / counts_obs.max()  # normalize by max count
bin_centers_obs = 0.5 * (bins_obs[:-1] + bins_obs[1:])

ax_marg_x.bar(
    bin_centers_obs,
    counts_obs,
    width=np.diff(bins_obs),
    color='blue',
    alpha=0.5,
    label='obs_data (Spiral)'
)

# Match x-limits with the main plot
ax_marg_x.set_xlim(ax.get_xlim())

# Set y-label to indicate normalized counts
ax_marg_x.set_ylabel("Norm Counts", fontsize=18)

# Legend
ax_marg_x.legend(loc="upper left", fontsize=18, frameon=True)

# Set y-ticks from 0 to 1 (normalized)
ax_marg_x.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_marg_x.set_yticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=13)
# Show ticks on both left and right sides
ax_marg_x.yaxis.set_ticks_position('both')
ax_marg_x.tick_params(axis='y', length=5, direction='in')

# Remove x-axis (only keep density axis visible)
ax_marg_x.xaxis.set_visible(False)

################################################ Plot Labels ##################################################


# Set axis labels
ax.set_xlabel(r'$\log\left(\rm{SFR}\,[\rm{M}_{\odot}\,\rm{yr}^{-1}]\right)$')
ax.set_ylabel(r'$\log\left(L_{1.4} \,[\rm{W} \,\rm{Hz}^{-1}]\right)$')

# Add legend ------------------------------------------------------------------------
# Get existing legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Define the desired order (modify as needed)
order = [1,2,3,4,5,0]  # Change indices to reorder legends

# Apply reordered legend
ax.legend(
    [handles[i] for i in order], [labels[i] for i in order], 
    frameon=True, fontsize=20, loc="upper left", facecolor='white', edgecolor='black'
)

colorbar_height = 0.02  # Height of the colorbar
colorbar_pad = 0.01  # Padding between the subplots and the colorbar

fig.subplots_adjust(bottom=0.18)  # Leave space for the colorbar at the bottom

# Set a common colorbar for all subplots at the bottom
cbar_ax = fig.add_axes([0.13, 0.05, 0.35, colorbar_height])  # [left, bottom, width, height]
fig.colorbar(sc, cax=cbar_ax, orientation="horizontal", label=r'$\log\left(\rm{sSFR}\,[\rm{yr}^{-1}]\right)$')


# plot title
# fig.suptitle('J24 (4.8GHz)',x=0.55,y=0.85, fontsize=20,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3',alpha=0.5))
fig.suptitle(r'(1.4 GHz)',x=0.329,y=0.685, fontsize=20,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3',alpha=0.5))

# fig.text(0.345,0.365,r'$L_{I,1.4\,\rm GHz}^{\rm nt}=L_{0,1.4\,\rm GHz} \left[\frac{\rm SFR}{0.1\rm M_{\odot}\,yr^{-1}}\right]^{\gamma}$', fontsize=20,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3',alpha=0.5))
# fig.text(
#     0.58, 0.925, r'J24',
#     ha='right',
#     fontsize=20,
#     fontfamily='monospace',     # 'serif', 'sans-serif', 'monospace', 'cursive', 'fantasy'
#     fontweight='bold',      # 'normal', 'bold', 'light', 'ultralight', 'heavy', 'black'
#     transform=ax.transAxes,
#     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.7)
# )

fig.text(
    0.625, 0.925, r'Fiducial',
    ha='right',
    fontsize=20,
    fontfamily='monospace',     # 'serif', 'sans-serif', 'monospace', 'cursive', 'fantasy'
    fontweight='bold',      # 'normal', 'bold', 'light', 'ultralight', 'heavy', 'black'
    transform=ax.transAxes,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.7)
)

# plot title
# fig.suptitle('Total gas model (4.8GHz)',x=0.72,y=0.87, fontsize=20,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3',alpha=0.5))
# fig.suptitle(r'(1.4 GHz)',x=0.562,y=0.80, fontsize=20,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3',alpha=0.5))
# fig.text(0.96,0.85,r'(active)',ha='right',transform=ax.transAxes)
# fig.text(0.96,0.85,r'(quenched)',ha='right',transform=ax.transAxes)

# Save and display the plot
plt.savefig('/media/sukanta/New Volume1/PhD_Works/Research_works/Projects/Global_scaling_relations_I/plots/luminosity/Lum_I_SFR_sSFR_Total_Gas_1.4GHz_histo.png', bbox_inches='tight')
#plt.show()

