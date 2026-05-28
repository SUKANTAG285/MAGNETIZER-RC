'''
This code reads radial-dependent magnetizer properties and computes total synchrotron luminosity 
along with other global galaxy properties.
'''
import h5py
import numpy as np
import math
import random
from utils import *
from scipy.integrate import quad
from scipy import integrate
from astropy.cosmology import WMAP9 as cosmo

#volume - stores total cosmological volume of files
#filenos - file numbers of galform/magnetizer outputs to be used
#v_ay - array for storing cosmological volumes of each file
global volume, filenos, v_ay

###################################################
# SG:unit conversion & constants for calculating diffuse gas mass
M_sun  = 1.989*(10**30) # unit: kg
km_kpc = 3.08567758*(10**16)
cm_kpc = 3.08567758*(10**21)
g_kg   = 10**3
m_km   = 10**3
m_cm   = 10**(-2)
V0     = 25 #unit: km/s
m_H    = 1.67372*10**(-24)# unit: g
# calculating zeta
fb     = 1
xi     = (3/2)*fb**2
eps    = 1
zeta   =(1+xi+xi*eps)/3 # taking from LC's "pressure" note
erg_j  = 10**(-7)
pi     = 3.14156295358
###################################################
#The number density or PDFs are constructed at these redshifts specified by "z_ay"
z_ay     =[0.001]#,0.5,1.0,1.5,2.0,3.0]#,1.5,2.0]#, 3.0]

#The PDFs are constructed using these file numbers 
filenos      = range(5,6,1)

#Input File paths of galform  magnetizer and observable runs 
magnetizer_op_dir  ='/home/sukanta/B_op/'
galform_op_dir     = '/home/sukanta/galform_output/'

#outputs of this code will be stored below
processed_op_dir     = "/home/sukanta/MAGNETIZER/processed_op_dir/"

#Names of galform, magnetizer and observables models. 
#The filename will be like *galform_model_mag_model*hdf5

# GALFORM model name
galform_model  = 'Lacey14'

# MAGNETIZER model name
mag_model      ="G25_fb0.9_Rk1.5_no_alp_sq_SFRD_L16"#'G25_fb0.9_Rk1.5_no_alp_sq_SFRD_MD14'#'FidmRk1.5SFRcorMHB06'
#mag_model      = "CJ24_Rmol0.0_Rk1.0_fb0.8"#"SG_R19"#_floor_seed_frac0.01"#"CJ24_R19seed.frac_seed0.0001"# "CJ24_R19seed"
#mag_model       = "CJ24fiducial_with_all_trial1"#_cfloor1_sg"

##########################
#calculations begin here##
##########################

#Reading first galform file to get list of output redshift
#galform_file= galform_op_dir+ galform_model + "_2_corr.hdf5"
#reading_gfile= h5py.File(galform_file,"r")
galf_z =get_galf_z_output_list(galform_op_dir, galform_model,filenos[0])

for fno in filenos:
   #opening the magnetizer output hdf5 file with a given file number
    Bfile_name = magnetizer_op_dir + galform_model + "_" + \
               mag_model+ "_{0}_corr.hdf5".format(fno)
    Bfile      = h5py.File(Bfile_name, "r")
    print ("Magnetizer prop names ->", Bfile["Output/"].keys() )
   
   #To obtain the total volume and individual volume of galform outputs
    volume, v_ay = find_tot_volume( galform_op_dir, galform_model, [fno])

   #opening the galform output hdf5 file with a given file number
    gfile_name = galform_op_dir + galform_model + "_{0}_corr.hdf5".format(fno)
    gfile      = h5py.File(gfile_name, "r")
    print ("Galform prop names ->", gfile["Input/"].keys() )
   #reading SFR corr ratio file
    #SFR_ratio_file = np.loadtxt("/home/sukanta/MAGNETIZER/python_script/Lacey14_SFRcor_offset_HB06.txt")
    #SFR_ratio_file = np.loadtxt("/home/sukanta/MAGNETIZER/python_script/Lacey14_SFRcor_offset_HB06.txt")
    SFR_ratio_file = np.loadtxt("/home/sukanta/galform_output/Lacey14_SFR_z_offset_T25.txt")
    SFR_disk_to_total = np.loadtxt("/home/sukanta/MAGNETIZER/python_script/Lacey14_SFR_z_offset_disk_to_total.txt")
    print ("SFR_ratio_shape",SFR_ratio_file.shape)
   #
    for redshift in z_ay:
        iz , zgf = find_nearest(galf_z, redshift) 
        print ('redshift =',redshift)
        print ('closest redshift =', zgf)

    #Reading galform properties
      # For below variables
      # First index -> galaxy number
      # Second index -> redshift
        Mstar = gfile["Input/"+"Mstars_disk"][:,iz] + gfile["Input/"+"Mstars_bulge"][:,iz]
        Mbulge= gfile["Input/"+"Mstars_bulge"][:,iz]
        SFR   = gfile["Input/"+"SFR"][:,iz] 
        Mgas  = gfile["Input/"+"Mgas_disk"][:,iz]
        Mhalo   = gfile["Input/"+"Mhalo"][:,iz]
        central = gfile["Input/"+"central"][:,iz]
      # SFR correction 
        SFR_offset = SFR_ratio_file[iz]
        SFR_D_T    = SFR_disk_to_total[iz]
        #SFR_new    = SFR*SFR_offset*SFR_D_T
        SFR_new    = SFR*SFR_D_T


    #Reading Magnetizer properties
      # For below variables
      # First index -> galaxy number
      # Second index -> redshift
        Bavg   = Bfile["Output/"+"Bavg"][:,iz]
        Bmax   = Bfile["Output/"+"Bmax"][:,iz]
        rmax   = Bfile["Output/"+"rmax"][:,iz]
     
     # For below variables
     # First index -> galaxy number
     # Second index -> radial dependence of variable
     # Third index -> redshift
        r    = Bfile["Output/r"][:,:,iz]
        h    = Bfile["Output/h"][:,:,iz]
        v    = Bfile["Output/v"][:,:,iz]# km/s
        n    = Bfile["Output/n"][:,:,iz]#cm^-3
        Br_avg   = Bfile["Output/Br"][:,:,iz]
        Bp_avg   = Bfile["Output/Bp"][:,:,iz]
        Bz_avg   = Bfile["Output/Bzmod"][:,:,iz]
        Beq  = Bfile["Output/"+"Beq"][:,:,iz]
        P    = Bfile["Output/P"][:,:,iz]
        Omega= Bfile["Output/Omega"][:,:,iz]
        Ur   = Omega*r # SG: radial velocity :: unit: km/s
        
     #Constructing an array of galaxy ids 
        galid = np.arange(0, len(Mstar),1 )

        print(galid)

        print("Number of galform galaxies=", len(Mstar) )
        print("Number of magnetizer galaxies=", len(Br_avg) )
      
     #mask -> to select galaxies with a magnetic field        
        mask   = ((rmax > 0.0))# &(central == 1))# & (SFR_new>0.001)) # & (Mstar>10**10.5) )#SG: for disk galaxies (Mbulge_star/Mtotal_star)<0.5

     #Applying the above mask to select galaxies with magnetic fields
        Mstar = Mstar[mask]
        Mbulge= Mbulge[mask]
        Mgas  = Mgas[mask] 
        SFR_new   = SFR_new[mask] 
        Mhalo = Mhalo[mask]
#
        Bavg  = Bavg[mask] 
        Bmax  = Bmax[mask]
        rmax  = rmax[mask]
        Beq   = Beq[mask] 
        r     = r[mask]
        h     = h[mask]
        Br_avg    = Br_avg[mask] 
        Bp_avg    = Bp_avg[mask] 
        Bz_avg    = Bz_avg[mask]
        P     = P[mask]
        Ur    = Ur[mask]
        v     = v[mask]
        n     = n[mask]
        galid = galid[mask]

        print("Number of galform galaxies after masking=", len(Mstar) )
        print("Number of magnetizer galaxies after masking=", len(Br_avg) )

        # Number of galaxies after masking
        no_gals = len(Bavg) 
        
        # Br and Bphi are the vertical avg. (between -h to h) magnetic field strengths 
        # for exponentially decaying profile
        # The mid-plane values are given below
        exp_factor = 1/(1-np.exp(-1))
        Br = exp_factor * Br_avg
        Bp = exp_factor * Bp_avg
        Bz = exp_factor * Bz_avg

#Sqaure of large scale and random fields as a function r
        B_sq       = (Br**2.0 + Bp**2.0 + Bz**2.0) 
        Beq_sq     = (Beq)**2.0
        Br_sq      = Br**2.0
        Bp_sq      = Bp**2.0
        B_total_sq = B_sq + Beq_sq #(0.8)**2*Beq_sq
     #  pitch_sq   = (Br/Bp)**2.0
#Arrays to store rms value of large scale and equipartition field where disc radius is 2.7 rhalf
#In Magnetizer, the galaxy properties are stored as a function of r up to r= 2.7 rhalf
        B_rms       = np.zeros( len(B_sq) )
        Beq_rms     = np.zeros( len(B_sq) )
        beta0_rms   = np.zeros( len(B_sq) )
        M_diffuse_P = np.zeros( len(B_sq) )
        M_diffuse_n = np.zeros( len(B_sq) )
        B_total_avg = np.zeros( len(B_sq) )
        B_bar_avg   = np.zeros( len(B_sq) )
        b_avg       = np.zeros( len(B_sq) )
        pitch_avg   = np.zeros( len(B_sq) )
        deg_pol     = np.zeros( len(B_sq) )
        Spi         = np.zeros( len(B_sq) )
        Si          = np.zeros( len(B_sq) ) 
        S_I         = np.zeros(len(B_sq))
        P_ava       = np.zeros(len(B_sq))
        v_t         = np.zeros(len(B_sq))
        Lum         = np.zeros(len(B_sq))
        Rm1          = np.zeros( len(B_sq) )
        num_density  = np.zeros(len(B_sq))
        SFR_11       = np.zeros(len(B_sq))
#Arrays to store rms value of large scale and equipartition field where disc radius is rhalf
        B_rms_rhalf     = np.zeros( len(B_sq) )
        Beq_rms_rhalf   = np.zeros( len(B_sq) )
        beta0_rms_rhalf = np.zeros( len(B_sq) )
        Ur_R            = np.zeros( len(B_sq) )
        Ur_R1           = np.zeros( len(B_sq) )
        Ur_R2           = np.zeros( len(B_sq) )
        Ur_avg          = np.zeros( len(B_sq) )
        R_25            = np.zeros( no_gals)
        r_half          = np.zeros( no_gals)
        R_200           = np.zeros( no_gals)
        
        # Random seed for randomize the galaxy inclination 
        np.random.seed(42)
        
        for  igal in range( no_gals ):
            print('igal--->', igal)
#Compute B_rms and Beq_rms by  averaging B^2 upto 2.7 rhalf
            Br1        = ( Br[igal,:][1:] + Br[igal,:][:-1] )/2.0
            Bp1        = ( Bp[igal,:][1:] + Bp[igal,:][:-1] )/2.0
            Bz1        = ( Bz[igal,:][1:] + Bz[igal,:][:-1] )/2.0
            B_sq1      = ( B_sq[igal,:][1:] + B_sq[igal,:][:-1] )/2.0
            Beq_sq1    = ( Beq_sq[igal,:][1:]+ Beq_sq[igal,:][:-1] )/2.0
            B_total_sq1= ( B_total_sq[igal,:][1:] + B_total_sq[igal,:][:-1] )/2.0
#         pitch_sq1  = ( pitch_sq[igal,:][1:] + pitch_sq[igal,:][:-1] )/2.0
            Br_sq1     = ( Br_sq[igal,:][1:] + Br_sq[igal,:][:-1] )/2.0
            Bp_sq1     = ( Bp_sq[igal,:][1:] + Bp_sq[igal,:][:-1] )/2.0
            r1         = ( r[igal,:][1:] + r[igal,:][:-1] )/2.0
            h1         = ( h[igal,:][1:] + h[igal,:][:-1] )/2.0
            dr1        = r[igal,:][1:] - r[igal,:][:-1]
            P1         = ( P[igal,:][1:] + P[igal,:][:-1] )/2.0
            v1         = ( v[igal,:][1:] + v[igal,:][:-1] )/2.0
            n1         = ( n[igal,:][1:] + n[igal,:][:-1] )/2.0
            beta0_sq1  = B_sq1/Beq_sq1
            pitch_sq1  = Br_sq1/Bp_sq1
#        B_rms[igal]   = ( np.sum( B_sq1*h1*r1*dr1 ) / np.sum(r1*h1*dr1) )**0.5
#       
            B_rms[igal]     = ( np.trapz(B_sq1*h1*r1, r1)/np.trapz(h1*r1, r1) )**0.5
            Beq_rms[igal]   = ( np.trapz(Beq_sq1*h1*r1, r1)/np.trapz(h1*r1, r1) )**0.5
            beta0_rms[igal] = ( np.trapz(beta0_sq1*h1*r1, r1)/np.trapz(h1*r1, r1) )**0.5
            SFR_11[igal]= SFR_new[igal]
#---------------- computing the factor f_b ----------------------------------------------------

            f_b_profile = 4
    
            if f_b_profile == 1: # Large-scale tangling : model 1
                if beta0_rms[igal] < 0.1:
                    f_b = 0.1
                else:
                    f_b = 1
            
            elif f_b_profile == 2: # Large-scale tangling : model 2
                if SFR_11[igal] > 1:
                    f_b = 1.5
                elif SFR_11[igal] > 0.1:
                    f_b = 1.2
                else:
                    f_b = 1
                    
            elif f_b_profile == 3: # Large-scale tangling: model 3 (continuous profile)
                    if beta0_rms[igal] <= 1:
                        f_b = beta0_rms[igal]
                    else:
                        f_b = 1
                    
            elif f_b_profile == 4: # constant profile
                f_b = 0.8
                
            else:
                print("choose f_b profile")
            
# -----------------------------------------------------------------------------------------            
            
            
         # Calculating total synchrotron intensities
            
            B_bar_avg[igal]   = (np.trapz((B_sq1/B_total_sq1)*h1*r1*(1-np.exp(-2)),r1)/np.trapz(2*h1*r1,r1))**0.5
#             B_total_avg[igal] =(np.trapz((B_total_sq1)*(1-np.exp(-2))*h1*r1,r1)/np.trapz(2*h1*r1,r1))**0.5
            B_total_avg[igal] =(np.trapz((np.sqrt(B_total_sq1)**4)*(1-np.exp(-4))*h1*r1,r1)/np.trapz(2*h1*r1,r1))**(1/4)

#             B_total_avg[igal] =(np.trapz((np.sqrt(B_total_sq1)**5)*(1-np.exp(-5))*h1*r1,r1)/np.trapz((np.sqrt(B_total_sq1)**4)*(1-np.exp(-4))*h1*r1,r1))
#             B_bar_avg[igal]   = (np.trapz((np.sqrt(B_sq1))*h1*r1*(1-np.exp(-2)),r1)/np.trapz(2*h1*r1,r1))**0.5
#             (np.trapz((np.sqrt(B_total_sq1))**2*(np.sqrt(B_sq1))**3*2*np.pi*r1*h1,r1)/ \
#                                 np.trapz((np.sqrt(B_total_sq1))**2*(np.sqrt(B_sq1))**2*2*np.pi*r1*h1,r1))
            pitch_avg[igal]   = (np.trapz((np.sqrt(B_total_sq1))**2*(np.sqrt(B_sq1))**2*np.sqrt(pitch_sq1)*2*np.pi*r1*h1,r1)/  \
                                np.trapz((np.sqrt(B_total_sq1))**2*(np.sqrt(B_sq1))**2*2*np.pi*r1*h1,r1))
                   
            # Volume avg number density
            num_density[igal] = ( np.trapz(n1*r1, r1)/np.trapz(r1, r1) )

    
                
            deg_pol[igal]     = (np.trapz((np.sqrt(B_total_sq1))**2*(np.sqrt(B_sq1))**2*2*np.pi*r1*h1,r1)/ \
                                np.trapz((np.sqrt(B_total_sq1))**4*2*np.pi*r1*h1,r1)) # degree of polarozation 
                                                 
            Si[igal]          = np.trapz(((np.sqrt(B_total_sq1))**2)*((np.sqrt(B_total_sq1))**2)*2*np.pi*r1*h1/1000,r1)
            

###############################################################################################################################################
######################################## SYNCHROTRON LUMINOSITY & INTENSITY ###################################################################
            # constants (in CGS units):
            e       = 4.8032 *10**(-10)  # cm^(3/2) g^(1/2) s^(−1) : electron charge in statcolumb
            m_e     = 9.1094 * 10**(-28) # g : mass of electron
            c       = 2.9979 * 10**10    # cm/s : speed of light
            H_0     = 69.32              # km/s/Mpc : Hubble Constant
            Omega_m = 0.32               # mass density parameter (Planck2016)
            
            # unit conversions :
            kpc_cm = 3.086 * 10**21   # cm
            Mpc_cm = 3.086 * 10**24   # cm
            cm_km  = 10**(-5)         # km
            GHz_Hz = 10**9            # Hz
            muG_G  = 10**(-6)         # muG
            GeV_erg= 0.00160218       # GeV
            Jy_mJy = 10**3            # mJy
            Jy_cgs = 10**(-23)        # erg s^-1 cm^-2 Hz^-1
            
            # parameters:
            s    = 3.0   # spectral index of relativistic electron energy spectrum in the Solar vicinity (ss21 fig 10.10)
            E    = 8.0   # GeV: minimum or threshold energy of relativistic electron energy spectrum in the Solar vicinity (ss21 fig 10.10)
            k_cr = 100.0 # the ratio of the energy densities of the relativistic protons and electrons
            nu   = 4.8   # GHz: frequency at which synchrotron intensity is measured
            
            # Galaxy Parameters
            rd = np.max(r1)       # Disk radius
            hd = np.max(h1)/1000  # Maximum scale hight[h(rd)]
            h_min = np.min(h1)/1000  # Maximum scale hight[h(rd)]

            # Choosing random inclination angle(between LoS and normal of the galactic disk) between 0 to pi of the galaxies
            i = np.arccos(1 - np.random.uniform(0.0, 1))#np.arcsin(random.uniform(0.0, 1))           
            
            # Absolute Magnitude Distance
            d    = 10.0  # Mpc: distance of the galaxy
            
            # Luminosity Distance
            d_L1  = cosmo.luminosity_distance(0.02)# from Astropy  
            d_L   = (2*c*cm_km/H_0/Omega_m**2)*(Omega_m*redshift + \
                    (Omega_m-2)*(np.sqrt(Omega_m*redshift +1)-1)) # Mpc
#             print("distance",d_L1)
            
            # visible area of the galaxy at an inclination i
            gal_area = np.pi*(np.max(r1))**2*np.cos(i)
            # Solid angle created on the telescope
            sigma = gal_area/d**2
            
            # mean magnetic field as a function of r
            B_bar = np.sqrt(B_sq1)
            # random magnetic field as a function of r
            b_rms = f_b*np.sqrt(Beq_sq1)
            # total magnetic field as a function of r 
#             B_T   = np.sqrt( b_rms**2(B_total_sq1) # muG 
            B_T   = np.sqrt( B_bar**2+b_rms**2) # muG 

            
            K_E   = ((s-2)/(8*np.pi*k_cr)) * (B_T*muG_G)**2 * (E*GeV_erg)**(s-2) # particles cm^-3 erg^(s-1): B and E are in muG & GeV respectively 

            # without equipartition
#             K_E   = ((s-2)/(k_cr)) * (E*GeV_erg)**(s-2) # particles cm^-3 erg^(s-1): B and E are in muG & GeV respectively 
            
            
#             print("K_E=",K_E)
            a_s   = (np.sqrt(3)/(4*np.pi*(s+1))) * math.gamma((3*s -1)/12) * math.gamma((3*s +19)/12)

# 
            # Radial derivatives of Br(r) and h(r) to compute Bz
            dBr1_dr1 = np.gradient(Br1, r1)        # d(Br1)/d(r1)
            dh1_dr1  = np.gradient(h1/1000, r1)    # d(h1)/d(r1)

            # Total Emissivity Function (x,y,z)
            def synchrotron_emissivity(phi, j, z):

                r_clamped = np.clip(r1[j], np.min(r1), np.max(r1))

                # defining variables
                h_r          = h1[j] / 1000
                Br_r         = Br1[j]
                Bp_r         = Bp1[j]
                Bz_r         = Bz1[j]

                n_r          = n1[j]
                Beq_sq_r     = Beq_sq1[j]
                delBr_delr_r = dBr1_dr1[j]
                delh_delr_r  = dh1_dr1[j]

                # Create φ–z grid
                PHI, Z = np.meshgrid(phi, z)

                # LoS components
                nr   = -np.sin(i) * np.sin(PHI)
                nphi = -np.sin(i) * np.cos(PHI)
                nz   = np.cos(i)

                sign_z = np.sign(Z)
                sign_z[Z == 0] = 1

                Br_rz = Br_r * np.exp(-np.abs(Z) / h_r)
                Bp_rz = Bp_r * np.exp(-np.abs(Z) / h_r)
                # computing Bz profile from Del. B = 0 (for exponentially decaying profile of Br)
                Bz_rz = sign_z * ((Br_r/r_clamped + delBr_delr_r) 
                                  * h_r * (np.exp(-np.abs(Z) / h_r) - 1)
                                  + Br_r * delh_delr_r 
                                  * (((np.abs(Z) / h_r) + 1) * np.exp(-np.abs(Z) / h_r) - 1))

                # Total mean-field strength
                B_bar_rz = np.sqrt(Br_rz**2 + Bp_rz**2 + Bz_rz**2)

                # Sky plane components of the mean-field
                Bx = Br_rz*np.cos(PHI) - Bp_rz*np.sin(PHI)

                By = (Br_rz*np.sin(PHI)*np.cos(i) 
                      + Bp_rz*np.cos(PHI)*np.cos(i) 
                      + Bz_rz*np.sin(i))
                # Mean-field component perpendicular to LoS
                B_perp_LoS = np.sqrt(Bx**2 + By**2)
                
#                 # alternative approch to compute the perpendicular comp. of mean-field to LoS
#                 # Parallel component of mean Magnetic field along the LoS
#                 B_bar_parall   = nr * Br1[j] + nphi * Bp1[j] + nz * Bz1[j]
#                 # Perpendicular componant of mean Magnetic field along the LoS
#                 B_bar_perp     = np.sqrt(B_bar[j]**2 - B_bar_parall**2 )

#                 B_perp_LoS = B_bar_perp * np.exp(-np.abs(Z) / h_r)
                
                # random magnetic field strength
                b_rms_rz = f_b * np.sqrt(Beq_sq_r) * np.exp(-np.abs(Z)/2/h_r)
                            
                # random magnetic field componant perpendicuular to LoS
                b_perp_rz = np.sqrt((2.0/3.0) * b_rms_rz**2)

                # total magnetic field strength
                B_T = np.sqrt(B_bar_rz**2 + b_rms_rz**2)

                # Equipartition constant
                K_E = ((s-2)/(8*np.pi*k_cr)) * (B_T * muG_G)**2 * (E * GeV_erg)**(s-2)

                # Synchrotron emissivity
                eps = K_E * a_s * (e**3/m_e/c**2) * (3*e/(4*np.pi*m_e**3*c**5))**((s-1)/2) \
                      * (np.sqrt(B_perp_LoS**2 + b_perp_rz**2) * muG_G)**((s+1)/2) \
                      * (nu * GHz_Hz)**(-(s-1)/2)

                return eps

            # Computing the total synchrotron luminosity
            
            # Azimuthal angle
            phi  = np.linspace(0.0, 2*np.pi, 51)
            # Perform integration over phi for each r
            integrated_values_phi = np.zeros(len(r1))  
            for j in range(len(r1)):
                z = np.linspace(-5*h1[j]/1000,5*h1[j]/1000,501)
                eps = synchrotron_emissivity(phi, j, z)
                integrated_values = np.trapz(eps,phi, axis=1)
                integrated_values_phi[j] = np.trapz(integrated_values,z)

            # performing integration over r
            integrated_value =  np.trapz(integrated_values_phi * r1 , r1)  
             
            # Calculate total synchrotron Luminosity in cgs unit by integrated over total solid angle 
            L   =  integrated_value * kpc_cm**3 *4*np.pi#*abs(np.cos(i))
            
            Lum[igal] = L

            
            # Calculate total synchrotron flux density: S_I = L/(4.pi.d^2): unit: mili-Jansky(mJy)
            S_I [igal] = L*Jy_mJy*(d_L*Mpc_cm)**2/(d*Mpc_cm)**2/Jy_cgs 

            if igal == 3:
#                 print("S_I",S_I[igal])
                print('inclination',i)
                print("Lum",Lum[igal])


##############################################################################################################################################
##############################################################################################################################################      
         


##############################################################################################################################################         
#########################Computing Diffuse Gas Mass(unit: Msun)############################################################################### 
         
            method =2
            if method ==1:#using R19 equn (8): using SI units
                M_diffuse_P[igal] = (np.trapz(P1*2*(np.pi)*h1*r1/1000/v1**2/zeta, r1)*(erg_j*(1/(m_cm**3))*\
                                   (km_kpc*m_km)**3/(m_km)**2/M_sun))
        
            elif method ==2:#directly from diffuse gas number density: using CGS units
                M_diffuse_n[igal] = (np.trapz(n1*m_H*2*(np.pi)*h1*r1/1000, r1)*(cm_kpc)**3/(M_sun*g_kg))
        
            else:
                print("choose any perticular method to compute diffuse gas Mass")


                
###############################################################################################################################################
############################################### Weighted avarage pressure #####################################################################

            P_ava[igal] = ( np.trapz(P1*h1*r1, r1)/np.trapz(h1*r1, r1) )
            v_t[igal] = ( np.trapz(v1*h1*r1, r1)/np.trapz(h1*r1, r1) )





###############################################################################################################################################
###############################################################################################################################################
                
#Compute Brms and Beq_rms by averaging B^2 upto rhalf
            irhalf , rhalf = find_nearest(r[igal,:],  r[igal,:][-1]/2.7 )
            B_sq1      = ( B_sq[igal,:][1:irhalf] + B_sq[igal,:][:irhalf-1])
            Beq_sq1    = ( Beq_sq[igal,:][1:irhalf]+ Beq_sq[igal,:][:irhalf-1] )/2.0
            r1_half    = ( r[igal,:][1:irhalf] + r[igal,:][:irhalf-1] )/2.0
            h1_half    = ( h[igal,:][1:irhalf] + h[igal,:][:irhalf-1] )/2.0
            beta0_sq1  = B_sq1/Beq_sq1

###############################################################################################################################################
######################################## CALCULATIING R25 RADIUS ("Kravtsov2013_ApJ") ##############################################################################
           # half_mass radius
            r_half[igal]      = rhalf #kpc
           # R200 of the galaxy : R_200 = r_half/0.015
            R_200[igal]       = r_half[igal]/0.015 #kpc
           # R25 f the galaxy : R_25 = 0.048 * R_200
            R_25[igal]        = 0.048 * R_200[igal] #kpc
            
            if igal == 51:
                print("R_25",R_25[igal])  
                print('r_half',r_half[igal])

###############################################################################################################################################
###############################################################################################################################################
# SG: Computing the v_rotational at R = 2*rhalf

            iR, R = find_nearest(r[igal,:],  (r[igal,:][-1]*2.0)/2.7 )
            Ur_R[igal]   = Ur[igal, iR] #Ur[igal,:][:iR]

         
# SG: Computing the v_rotational at R = 1.5*rhalf
         
            iR1, R1 = find_nearest(r[igal,:],  (r[igal,:][-1]*1.5)/2.7 )
            Ur_R1[igal]   = Ur[igal, iR1] #Ur[igal,:][:iR]

# SG: Computing the v_rotational at R = 2.5*rhalf

            iR2, R2 = find_nearest(r[igal,:],  (r[igal,:][-1]*2.5)/2.7 )
            Ur_R2[igal]   = Ur[igal, iR2] #Ur[igal,:][:iR]

         # Extract the slice of Ur between 1.5*r_half and 2.5*r_half to compute the value of Ur for the flat part of the rotation curve
            Ur_slice = Ur[igal, iR1:iR2+1]

         # Compute the average of Ur over the flat part of the rotation curve
            Ur_avg1 = np.mean(Ur_slice)

         # Assign the average to Ur_R
            Ur_avg[igal] = Ur_avg1
      
      
      #Brms_orig = np.concatenate( (Brms_orig,B_rms[:,iz]) )
        print('----Done----')
#saving average quantities to a file at a given z and for a given magnetizer output file number
        fname  = processed_op_dir + "Bprops/" + "Obs_total_Lum_SFRD_L16_mod1_" + galform_model + "_" + mag_model + \
                   "_fno{0}_z{1:.1f}.txt".format(fno, redshift)
        np.savetxt(fname, np.transpose([galid,rmax,Bmax,Bavg,B_rms,Beq_rms,B_total_avg,B_bar_avg, \
              Mhalo, Mstar,Mbulge, SFR_11, Mgas,M_diffuse_n,num_density,S_I,Lum,R_25,Ur_avg,v_t]) )
       # fname  = processed_op_dir + "Bprops/" + "volno_pola" + galform_model + "_" + mag_model + \
       #            "_fno{0}_z{1:.1f}.txt".format(fno, z)
       # np.savetxt(fname, np.transpose([ volume, len(galid) ]) )
      
