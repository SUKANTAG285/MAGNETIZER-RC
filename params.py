'''
This includes all the run parameters: switches, which redshifts/sub-volumes to process, 
the synchrotron model parameters, model names, and file paths.
'''

# ----------------------------------------------------------------------
# Run-time switches
# ----------------------------------------------------------------------

# Assumptions of no-z approximation: 
'''
'True': as exponential vertical average
'False': as mid-plane value
'''
vertical_avg_Bbar = True

# SFR-correction
''' 'True': if the SFRD is modified in MAGNETIZER based on BH06, MD15, or T25
    'False': if no correction is applied, i.e, same as Lacey+16 model
'''
SFR_correction    = False

# Which f_b (large-scale tangling factor) prescription to use.
# 1 -> step function of beta0_rms
# 2 -> step function of SFR
# 3 -> continuous function of beta0_rms
# 4 -> constant value
f_b_profile = 4

# ----------------------------------------------------------------------
# Synchrotron model parameters
# ----------------------------------------------------------------------
#s    = 3.0    # spectral index of relativistic electron energy spectrum


# Different alpha_nt models
# 1 -> redshift dependent
# 2 ->  sSFR dependent
# 3 -> constant spectral index (s = 3)
alpha_nt_model = 3


E    = 8.0    # GeV : threshold energy of relativistic electron spectrum
k_cr = 100.0  # ratio of energy densities of relativistic protons/electrons
nu   = 4.8    # GHz : rest frame frequency 

# Assumed distance used for the mock flux-density calculation
d_assumed = 10.0  # Mpc

# ----------------------------------------------------------------------
# Redshift for which the galaxy properties computed
''' In utils.py, the nearest redshift snapshot is figure out.'''
z_ay    = [0.001]  # , 0.5, 1.0, 1.5, 2.0, 3.0]
# Number of sub-volume 
filenos = range(1, 2, 1)

# ----------------------------------------------------------------------
# Paths & model names
# ----------------------------------------------------------------------
# MAGNETIZER output file path
magnetizer_op_dir = '/media/sukanta/New Volume1/GITHUB_projects/files/example/example_input/MAGNETIZER_op/'
# GALFORM output file path
galform_op_dir    = '/media/sukanta/New Volume1/GITHUB_projects/files/example/example_input/GALFORM_op/'
# Processed output file path
processed_op_dir  = "/media/sukanta/New Volume1/GITHUB_projects/files/example/"

# GALFORM model name
galform_model = 'Lacey14_new'
# MAGNETIZER model name
mag_model     = "G25_fb0.8_Rk0.3_vtmod2_no_alp_sq_SFRD_L16_Fiducial_mod1"

# Optional SFR-correction lookup table (only read if SFR_correction is True)
sfr_ratio_file_path = "/home/sukanta/galform_output/Lacey14_SFR_z_offset_T25.txt"
