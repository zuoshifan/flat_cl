import numpy as np
import h5py


fl1 = 'cube_galaxy_smooth.hdf5'
with h5py.File(fl1, 'r') as f:
    ras = f['data'].attrs['ra']
    decs = f['data'].attrs['dec']


fl2 = 'cube_21cm_smooth.hdf5'
with h5py.File(fl2, 'r+') as f:
    del f['data'].attrs['ra']
    del f['data'].attrs['dec']
    f['data'].attrs['ra'] = ras
    f['data'].attrs['dec'] = decs
