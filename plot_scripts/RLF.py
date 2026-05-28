# This code computes Radio Luminosity Functions and plots its redshift evolution -----------------------!
#-------------------------------------------------------------------------------------------------------!

# Libraries ----------------------------------------------------------------------!
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
import RLF_obs_data as RLFdata


# Constants ----------------------------------------------------------------------!
K      = 0.81          # rad m^−2 cm^3 μG^−1 pc^−1
L_sun  = 3.828*10**26  # W Hz^-1

# unit conversions ---------------------------------------------------------------!
kpc_cm = 3.086 * 10**21   # cm
Mpc_cm = 3.086 * 10**24   # cm
Mpc_m  = 3.086*  10**22   # m
cm_km  = 10**(-5)         # km
GHz_Hz = 10**9            # Hz
muG_G  = 10**(-6)         # muG
GeV_erg= 0.00160218       # GeV
Jy_mJy = 10**3            # mJy
Jy_cgs = 10**(-23)        # erg s^-1 cm^-2 Hz^-1
mJy_si = 10**(-29)        # W m^-2 Hz^-1


# Reading the radio-luminosity data at different redshifts -----------------------!

# Initialize DataFrame to hold data
data_sam = pd.DataFrame()
data_sam_J24 = pd.DataFrame()

# Galform_model name:
galform_model = 'Lacey14_new'
galform_model_J24 = 'Lacey14'

# Magnetizer_model name:
mag_model_J24     = "CJ24fiducial" # J24 model
mag_model      ="G25_fb0.8_Rk0.3_vtmod2_no_alp_sq_SFRD_L16_Fiducial_mod1" # Fiducial model of G26

# The PDFs are constructed using these file numbers
filenos = np.r_[1:11, 13:30] # files for G26 model
filenos_J24 = range(2,11,1)  # files for J24 model

# Read data from .txt files and store into a DataFrame
data_sam = pd.DataFrame()
data = pd.DataFrame()
data_z = pd.DataFrame()
data_sam_z = pd.DataFrame()

# Redshifts for which we need to plot the PDF
redshifts = [0.0,0.25,0.5,0.7,0.9,1.15,1.45,1.8,2.25]#,2.9,4.0]

# Define a variable to store all scatter plots
all_sc = []

# Loop through each redshift
for z in redshifts:
    # Loop through each file number to read data for G26 model
    for fno in filenos:
        # Read the data from the corresponding .txt file
        filename = f"/media/sukanta/New Volume1/New folder/magnetizer_cj24/Bprops/props_pola_trial_total_sfr_L16_mod1_{galform_model}_{mag_model}_fno{fno}_z{z:.1f}.txt"
        df = pd.read_csv(filename, header=None, sep=' ')

        # Define column names
        df.columns = ['galid','rmax','Bmax','Bavg','B_rms','Beq_rms','B_total_avg','B_bar_avg', 
                      'Mhalo', 'Mstar','Mbulge', 'SFR_new', 'Mgas','M_diffuse_n','num_density',
                      'S_I','Lum','R_25','Ur_avg','v_t']

        df['redshift'] = z

        # Concatenate the data
        data_sam = pd.concat([data_sam, df])


    # Loop through each file number to read data for J24 model
    for fno1 in filenos_J24:
        # Read the data from the corresponding .txt file
        filename_J24 = f"/media/sukanta/New Volume1/New folder/magnetizer_cj24/Bprops/props_pola_{galform_model_J24}_{mag_model_J24}_fno{fno1}_z{z:.1f}.txt"
        df2 = pd.read_csv(filename_J24, header=None, sep=' ')

        # Define column names
        df2.columns = ['galid','rmax','Bmax','Bavg','B_rms','Beq_rms','B_total_avg','B_bar_avg', 
                      'Mhalo', 'Mstar','Mbulge', 'SFR_new', 'Mgas','M_diffuse_n','num_density',
                       'S_I','Lum','R_25','Ur_avg','v_t']

        df2['redshift'] = z

        # Concatenate the data
        data_sam_J24 = pd.concat([data_sam_J24, df2])
        
# Applying Filters to the data -------------------------------------------------------!

# Mbulge filter
apply_Mbulge_cutoff = False
Mbulge_cutoff_value = 10**(9) # M_star

# sSFR filter 
apply_sSFR_cutoff = False
sSFR_cutoff_value = -10.5 # M_sun/yr (-10.5 for z = 0 & -9.3 for z = 1.5)

# SFR filter 
apply_SFR_cutoff = False
SFR_cutoff_value = 0.001

# Luminosity filter (needed if observed data have selection effects)
apply_L_cutoff = False
L_cutoff_value = 20

# Bulge to Total stellar mass ratio (B/T) filter
apply_BT_cutoff = False
BT_cutoff_value = 0.2

# R_25 radius filter
apply_R_25_cutoff = False
R_25_cutoff_value = 6 # kpc


# Condition for filtering the data
condition = np.ones(len(data_sam), dtype=bool)

# Applying the filters
if apply_Mbulge_cutoff:
    condition &= (data_sam['Mstar'] > Mbulge_cutoff_value)

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
    

# Apply the combined filter to G26 model only 
data_sam = data_sam[condition]

#------------------------------------------------------------------------------------------------!

#**************************************** Plotting **********************************************
#************************************************************************************************

# Update font size for plots
plt.rcParams.update({'font.size': 28})
rcParams['font.family'] = 'serif'

#Setting Figure Size, Fonts and Properties
fig, axs = plt.subplots(nrows=3, ncols=3,figsize=(30, 30),sharex=True,sharey=True,)
axs = axs.flatten()
fig.subplots_adjust(hspace=0.0)
fig.subplots_adjust(wspace=0.0)

# LOOP THROUGH EACH REFDHISTS ------------------------------------------------------------------!
for i, z in enumerate(redshifts):
    
    # Scatter Plot for the selected redshift
    
    # G26 model ---------------------------------------!
    data_sam_z = data_sam[data_sam['redshift'] == z] 
    df1 = pd.DataFrame(data_sam_z)
    data_z = df1  
    
    # J24 model ---------------------------------------!
    data_sam_z_J24 = data_sam_J24[data_sam_J24['redshift'] == z]
    df1_J24 = pd.DataFrame(data_sam_z_J24)
    data_z_J24 = df1_J24 


    # PLOT VARIABLES ----------------------------------!
    # Galform Parameters :
    Mstar       = np.array((data_z['Mstar']))
    bulge_mass  = np.array((data_z['Mbulge']))
    Mgas        = np.array((data_z['Mgas']))
    Mdiffuse_n  = np.array(data_z['M_diffuse_n'])
    Mhalo       = np.array((data_z['Mhalo']))
    SFR         = data_z['SFR_new']
    B_T_ratio   = bulge_mass/Mstar
    sSFR        = SFR/Mstar
    
    # Magnetizer Parameters :
    B_0         = data_z['B_rms']
    B_total_avg = data_z['B_total_avg']
    B_bar_avg   = data_z['B_bar_avg']
    v_rot_avg   = data_z['Ur_avg']
    R_25        = data_z['R_25']
    L_FIR       = SFR/(0.79*1.7*10**(-10)) # unit: L_sun
    
    # Convert to 1.4 GHz using alpha_nt = 1
    L           = data_z['Lum']*10**(-7)*((1.4)**(-1))/((4.8)**(-1)) # unit : W Hz^(-1) 
    L_J24       = data_z_J24['Lum']*10**(-7)*((1.4)**(-1))/((4.8)**(-1)) # unit : W Hz^(-1)

    
    # Computing Luminosity Function --------------------------------------!
    logL = np.log10(L)
    
    mask0 = (bulge_mass/Mstar <=0.4)
    mask1 = (bulge_mass/Mstar <=0.2)
    mask2 = (bulge_mass/Mstar <=0.5)

    logL_mask0 = L[mask0]
    logL_mask1 = L[mask1]
    logL_mask2 = L[mask2]
    
    logL_J24 = np.log10(L_J24)

    # Create bins (e.g., from 20 to 28 in steps of 0.5 dex)
    bins = np.arange(19, 27, 0.3)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = 0.3 # delta log L

    # Count galaxies in each bin
    counts_all, _ = np.histogram(np.log10(logL_mask0), bins=bins)
    counts_mask1, _ = np.histogram(np.log10(logL_mask1), bins=bins)
    counts_mask2, _ = np.histogram(np.log10(logL_mask2), bins=bins)

    counts_J24, _ = np.histogram(logL_J24, bins=bins)

    # Normalize by Volume and Bin Width
    # Comoving volume of the box
    v_singlebox = 244140.625  #if all the sub-volumes are equal 
    V_box     = v_singlebox * len(filenos)
    V_box_J24 = v_singlebox * len(filenos_J24)

    # Luminosity Function
    phi_mask0 = counts_all / (V_box * bin_width)
    phi_mask1 = counts_mask1 / (V_box * bin_width)
    phi_mask2 = counts_mask2 / (V_box * bin_width)

    phi_J24 = counts_J24 / (V_box_J24 * bin_width)

    
    
    
    sc = axs[i].plot(bin_centers, np.log10(phi_mask0), c='darkorange',label=r'Fiducial' if i == 0 else None, lw=3)#, yerr=errors, fmt='o', capsize=3)
    
    sc0 = axs[i].fill_between(bin_centers, np.log10(phi_mask1), np.log10(phi_mask2),color='darkorange', alpha=0.3,label=r'$0.2\leq B/T \leq 0.5$' if i == 0 else None)#, yerr=errors, fmt='o', capsize=3)

#     sc1 = axs[i].plot(bin_centers, np.log10(phi[mask1]), c='black',label=r'Fiducial', lw=3)#, yerr=errors, fmt='o', capsize=3)

    
    sc2= axs[i].plot(bin_centers, np.log10(phi_J24), c='black',ls='--',label=r'J24' if i == 0 else None, lw=2)#, yerr=errors, fmt='o', capsize=3)
       
    
    
    # Store scatter plot for each subplot..............................
    all_sc.append(sc) 
    
    # Observational Data -----------------------------------------------------!
    # z = 0.0
    if np.isclose(z, 0.0):
        y_err = [RLFdata.RLF_Condon02_err_minus, RLFdata.RLF_Condon02_err_plus]
        axs[i].errorbar(RLFdata.LogL_Condon02, RLFdata.RLF_Condon02, yerr= y_err,fmt='d',c='cyan', ms=8,label = 'Condon+2002'if i == 0 else None)

        y_err1 = [RLFdata.RLF_Condon19_err_minus, RLFdata.RLF_Condon19_err_plus]
        axs[i].errorbar(RLFdata.LogL_Condon19, RLFdata.RLF_Condon19, yerr= y_err1,fmt='s',c='black', ms=8,label = 'Condon+2019'if i == 0 else None)

        y_err2 = [RLFdata.RLF_Mauch07_err_minus, RLFdata.RLF_Mauch07_err_plus]
        axs[i].errorbar(RLFdata.LogL_Mauch07, RLFdata.RLF_Mauch07, yerr= y_err2,fmt='o',c='red', ms=8,label = 'Mauch+2007'if i == 0 else None)

        y_err3 = [RLFdata.RLF_Best05_err_minus, RLFdata.RLF_Best05_err_plus]
        axs[i].errorbar(RLFdata.LogL_Best05, RLFdata.RLF_Best05, yerr= y_err3,fmt='*',c='lime', ms=10,label = 'Best+2007'if i == 0 else None)

    # z = 0.25
    if np.isclose(z, 0.25):
        x_err1 = [RLFdata.LogL_z0p25_vanderVlught22_err_minus, RLFdata.LogL_z0p25_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z0p25_vanderVlught22_err_minus, RLFdata.RLF_z0p25_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z0p25_vanderVlught22, RLFdata.RLF_z0p25_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue', ms=8, label='van der Vlugt+2022' if i == 1 else None)

        x_err2 = [RLFdata.LogL_z0p25_Novak17_err_minus, RLFdata.LogL_z0p25_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z0p25_Novak17_err_minus, RLFdata.RLF_z0p25_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z0p25_Novak17, RLFdata.RLF_z0p25_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green', ms=8,label='Novak+2017'if i == 1 else None)
         
        y_err3 = RLFdata.RLF_z0p25_Smol_err
        axs[i].errorbar(RLFdata.LogL_z0p25_Smol, RLFdata.RLF_z0p25_Smol, yerr=y_err3, fmt='d', c='deeppink', ms=8,label='Smolcic+2009'if i == 1 else None)
         

    # z = 0.5
    if np.isclose(z, 0.5):
        x_err1 = [RLFdata.LogL_z0p5_vanderVlught22_err_minus, RLFdata.LogL_z0p5_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z0p5_vanderVlught22_err_minus, RLFdata.RLF_z0p5_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z0p5_vanderVlught22, RLFdata.RLF_z0p5_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue', ms=8,label='vanderVlught+22'if i == 1 else None)

        x_err2 = [RLFdata.LogL_z0p5_Novak17_err_minus, RLFdata.LogL_z0p5_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z0p5_Novak17_err_minus, RLFdata.RLF_z0p5_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z0p5_Novak17, RLFdata.RLF_z0p5_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green',ms=8, label='Novak+17'if i == 1 else None)

        y_err3 = RLFdata.RLF_z0p5_Smol_err
        axs[i].errorbar(RLFdata.LogL_z0p5_Smol, RLFdata.RLF_z0p5_Smol, yerr=y_err3, fmt='d', c='deeppink',ms=8, label='Smolcic+09'if i == 1 else None)
         
    # z = 0.7
    if np.isclose(z, 0.7):
        x_err1 = [RLFdata.LogL_z0p7_vanderVlught22_err_minus, RLFdata.LogL_z0p7_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z0p7_vanderVlught22_err_minus, RLFdata.RLF_z0p7_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z0p7_vanderVlught22, RLFdata.RLF_z0p7_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue',ms=8, label='vanderVlught+22'if i == 1 else None)

        x_err2 = [RLFdata.LogL_z0p7_Novak17_err_minus, RLFdata.LogL_z0p7_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z0p7_Novak17_err_minus, RLFdata.RLF_z0p7_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z0p7_Novak17, RLFdata.RLF_z0p7_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green',ms=8,  label='Novak+17'if i == 1 else None)
        
        y_err3 = RLFdata.RLF_z0p75_Smol_err
        axs[i].errorbar(RLFdata.LogL_z0p75_Smol, RLFdata.RLF_z0p75_Smol, yerr=y_err3, fmt='d', c='deeppink', ms=8, label='Smolcic+09'if i == 1 else None)
         

    # z = 0.9
    if np.isclose(z, 0.9):
        x_err1 = [RLFdata.LogL_z0p9_vanderVlught22_err_minus, RLFdata.LogL_z0p9_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z0p9_vanderVlught22_err_minus, RLFdata.RLF_z0p9_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z0p9_vanderVlught22, RLFdata.RLF_z0p9_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue',ms=8,  label='vanderVlught+22'if i == 1 else None)

        x_err2 = [RLFdata.LogL_z0p9_Novak17_err_minus, RLFdata.LogL_z0p9_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z0p9_Novak17_err_minus, RLFdata.RLF_z0p9_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z0p9_Novak17, RLFdata.RLF_z0p9_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green', ms=8, label='Novak+17'if i == 1 else None)


    # z = 1.15
    if np.isclose(z, 1.15):
        x_err1 = [RLFdata.LogL_z1p15_vanderVlught22_err_minus, RLFdata.LogL_z1p15_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z1p15_vanderVlught22_err_minus, RLFdata.RLF_z1p15_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z1p15_vanderVlught22, RLFdata.RLF_z1p15_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue', ms=8, label='vanderVlught+22'if i == 1 else None)

        x_err2 = [RLFdata.LogL_z1p15_Novak17_err_minus, RLFdata.LogL_z1p15_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z1p15_Novak17_err_minus, RLFdata.RLF_z1p15_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z1p15_Novak17, RLFdata.RLF_z1p15_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green',ms=8,  label='Novak+17'if i == 1 else None)


    # z = 1.45
    if np.isclose(z, 1.45):
        x_err1 = [RLFdata.LogL_z1p45_vanderVlught22_err_minus, RLFdata.LogL_z1p45_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z1p45_vanderVlught22_err_minus, RLFdata.RLF_z1p45_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z1p45_vanderVlught22, RLFdata.RLF_z1p45_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue',ms=8,  label='vanderVlught+22'if i == 1 else None)

        x_err2 = [RLFdata.LogL_z1p45_Novak17_err_minus, RLFdata.LogL_z1p45_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z1p45_Novak17_err_minus, RLFdata.RLF_z1p45_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z1p45_Novak17, RLFdata.RLF_z1p45_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green',ms=8,  label='Novak+17'if i == 1 else None)


    # z = 1.8
    if np.isclose(z, 1.8):
        x_err1 = [RLFdata.LogL_z1p8_vanderVlught22_err_minus, RLFdata.LogL_z1p8_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z1p8_vanderVlught22_err_minus, RLFdata.RLF_z1p8_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z1p8_vanderVlught22, RLFdata.RLF_z1p8_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue', ms=8, label='vanderVlught+22'if i == 1 else None)

        x_err2 = [RLFdata.LogL_z1p8_Novak17_err_minus, RLFdata.LogL_z1p8_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z1p8_Novak17_err_minus, RLFdata.RLF_z1p8_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z1p8_Novak17, RLFdata.RLF_z1p8_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green',ms=8,  label='Novak+17'if i == 1 else None)


    # z = 2.25 
    if np.isclose(z, 2.25):
        x_err1 = [RLFdata.LogL_z2p25_vanderVlught22_err_minus, RLFdata.LogL_z2p25_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z2p25_vanderVlught22_err_minus, RLFdata.RLF_z2p25_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z2p25_vanderVlught22, RLFdata.RLF_z2p25_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue', ms=8, label='vanderVlught+22')

        x_err2 = [RLFdata.LogL_z2p25_Novak17_err_minus, RLFdata.LogL_z2p25_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z2p25_Novak17_err_minus, RLFdata.RLF_z2p25_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z2p25_Novak17, RLFdata.RLF_z2p25_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green',ms=8,  label='Novak+17'if i == 1 else None)

    # z = 2.9
    if np.isclose(z, 2.9):
        x_err1 = [RLFdata.LogL_z2p9_vanderVlught22_err_minus, RLFdata.LogL_z2p9_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z2p9_vanderVlught22_err_minus, RLFdata.RLF_z2p9_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z2p9_vanderVlught22, RLFdata.RLF_z2p9_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue', ms=8, label='vanderVlught+22')

        x_err2 = [RLFdata.LogL_z2p9_Novak17_err_minus, RLFdata.LogL_z2p9_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z2p9_Novak17_err_minus, RLFdata.RLF_z2p9_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z2p9_Novak17, RLFdata.RLF_z2p9_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green',ms=8,  label='Novak+17'if i == 1 else None)

        
    # z = 2.9
    if np.isclose(z, 4.0):
        x_err1 = [RLFdata.LogL_z3p95_vanderVlught22_err_minus, RLFdata.LogL_z3p95_vanderVlught22_err_plus]
        y_err1 = [RLFdata.RLF_z3p95_vanderVlught22_err_minus, RLFdata.RLF_z3p95_vanderVlught22_err_plus]
        axs[i].errorbar(RLFdata.LogL_z3p95_vanderVlught22, RLFdata.RLF_z3p95_vanderVlught22,
                        xerr=x_err1, yerr=y_err1, fmt='o', c='blue', ms=8, label='vanderVlught+22')

        x_err2 = [RLFdata.LogL_z3p95_Novak17_err_minus, RLFdata.LogL_z3p95_Novak17_err_plus]
        y_err2 = [RLFdata.RLF_z3p95_Novak17_err_minus, RLFdata.RLF_z3p95_Novak17_err_plus]
        axs[i].errorbar(RLFdata.LogL_z3p95_Novak17, RLFdata.RLF_z3p95_Novak17,
                        xerr=x_err2, yerr=y_err2, fmt='d', c='green',ms=8,  label='Novak+17'if i == 1 else None)

#     if np.isclose(z, 2.25):
#         x_err1 = [RLFdata.LogL_z2p25_vanderVlught22_err_minus, RLFdata.LogL_z2p25_vanderVlught22_err_plus]
#         y_err1 = [RLFdata.RLF_z2p25_vanderVlught22_err_minus, RLFdata.RLF_z2p25_vanderVlught22_err_plus]
#         axs[i].errorbar(RLFdata.LogL_z2p25_vanderVlught22, RLFdata.RLF_z2p25_vanderVlught22, xerr= x_err1, yerr= y_err1,fmt='o', c='blue', label = 'vanderVlught+22')
        
#         x_err2 = [RLFdata.LogL_z2p25_Novak17_err_minus, RLFdata.LogL_z2p25_Novak17_err_plus]
#         y_err2 = [RLFdata.RLF_z2p25_Novak17_err_minus, RLFdata.RLF_z2p25_Novak17_err_plus]
#         axs[i].errorbar(RLFdata.LogL_z2p25_Novak17, RLFdata.RLF_z2p25_Novak17, xerr= x_err2, yerr= y_err2,fmt='d', c='lime', label = 'Novak+17')

################################################################################################

################################################################################################   
    # Axis Limits......................................................
    x_min, x_max =19,25.6 #-1, 2
    y_min, y_max = -8,-0.5 #7.5, 11.5
    axs[i].set_xlim(x_min,x_max)
    axs[i].set_ylim(y_min,y_max)
    
    # Remove the first and last ticks for nice look....................
    axs[i].set_yticks(np.arange(-8,0,1)) 
    axs[i].set_xticks(np.arange(19,26,1))
    

    # Subplot title and correlation coefficients...................................
#     axs[i].text(0.25, 0.15, f'z={z}', ha='right', color='red', transform=axs[i].transAxes,fontsize=25,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    axs[i].text(0.90, 0.85, f'z={z}', ha='right', color='red', transform=axs[i].transAxes,fontsize=25,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Customize tick parameters
    # Enable minor ticks
    axs[i].minorticks_on()

    # Set major and minor ticks for x-axis
    axs[i].tick_params(axis='x', which='major', bottom=True, top=True, length=10, width=2, direction='in')  # Major ticks
    axs[i].tick_params(axis='x', which='minor',bottom=True, top=True, length=5, width=1.5, direction='in')  # Minor ticks

    # Set major and minor ticks for y-axis
    axs[i].tick_params(axis='y', which='major', length=10, width=2, direction='in')  # Major ticks
    axs[i].tick_params(axis='y', which='minor', length=5, width=1.5, direction='in')  # Minor ticks
    
    axs[0].legend(frameon=True, fontsize=22, loc="lower left", facecolor='white', edgecolor='black')
    axs[1].legend(frameon=True, fontsize=22, loc="lower left", facecolor='white', edgecolor='black')

    
#     axs[i].text(0.52,1.5,r'(4.8 GHz)', fontsize=28,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3',alpha=0.5))
###################################################################################################

    #Set xtics.........................................................
#     axs[i].tick_params(axis='x', which='both', bottom=True, top=True,length=6,direction='in')
#     axs[i].tick_params(axis='y',length=10,direction='in')
    # Set common x-axis label
    fig.supxlabel(r'$\log_{10}\left(L_{1.4} \,[\rm{W} \,\rm{Hz}^{-1}]\right)$', x=0.5, y=0.05)

    # Set common y-axis label
    fig.supylabel(r'$\log_{10}(\Phi\,[\rm{Mpc}^{-1}\rm{dex}^{-1}])$', x=0.08, y=0.51)


# Save the figure................................................
plt.savefig('/media/sukanta/New Volume1/PhD_Works/Research_works/Projects/Global_scaling_relations_I/plots/luminosity/RLF_redshift_evolve.png', bbox_inches='tight')

