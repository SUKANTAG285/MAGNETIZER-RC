import numpy as np
import h5py

# Find the nearest number to the number 'value' and its index from an array 'array'
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx), array[idx]

# Reading galform file to get list of output redshift
def get_galf_z_output_list(galform_op_dir, galform_model, fno):
    galfile = galform_op_dir + galform_model + "_{0}_corr.hdf5".format(fno)
    with h5py.File(galfile, "r") as galf:
        galf_z = galf["Input/z"][()]
    return galf_z

def read_galf_z_prop(galform_op_dir, galform_model, fno, prop_name, iz):
    galfile = galform_op_dir + galform_model + "_{0}_corr.hdf5".format(fno)
    with h5py.File(galfile, "r") as galf:
        print(list(galf["Input/"].keys()))
        prop_val = galf["Input/"+prop_name][()]
    return prop_val

# To find total cosmological volume of all galform runs whose filenumbers are given
def find_tot_volume(galform_op_dir, galform_model, filenos):
    volume = 0
    v_ay = np.zeros(len(filenos))
    for ifn, fnum in enumerate(filenos):
        fname = galform_op_dir + galform_model + "_{0}_corr.hdf5".format(fnum)
        with h5py.File(fname, "r") as f:
            v_ay[ifn] = f["Galform Parameters"]["volume"][()]
            volume += f["Galform Parameters"]["volume"][()]
            print('file_no, file_vol, tot_vol', fnum, f["Galform Parameters"]["volume"][()], volume)
    return volume, v_ay

