import os
from collections import defaultdict
import numpy as np
from scipy import constants as const
import h5py
from astropy.cosmology import Planck13 as cosmo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# foregrounds
with h5py.File('cube_galaxy_smooth.hdf5', 'r') as f:
    fg_data = f['data'][:] # K
    freqs = f['data'].attrs['nu'] # MHz
    ras = f['data'].attrs['ra'] # degree
    decs = f['data'].attrs['dec'] # degree

# 21 cm
with h5py.File('cube_21cm_smooth.hdf5', 'r') as f:
    HI_data = f['data'][:] # K


nf = freqs.shape[0]
nra = ras.shape[0]
ndec = decs.shape[0]

fi = nf / 2

if not os.path.isdir('results'):
    os.mkdir('results')

# plot central slice of foregrounds
plt.figure()
plt.imshow(fg_data[fi, :, :], origin='lower', aspect='equal', extent=[ras[0], ras[-1], decs[0], decs[-1]])
cb = plt.colorbar()
cb.ax.set_ylabel('Brightness temperature / K', fontsize=16)
plt.xlabel('RA / degree', fontsize=16)
plt.ylabel('DEC / degree', fontsize=16)
plt.savefig('results/smooth_galaxy_slice_%d.png' % fi)
plt.close()

# plot central slice of HI
plt.figure()
plt.imshow(HI_data[fi, :, :], origin='lower', aspect='equal', extent=[ras[0], ras[-1], decs[0], decs[-1]])
cb = plt.colorbar()
cb.ax.set_ylabel('Brightness temperature / K', fontsize=16)
plt.xlabel('RA / degree', fontsize=16)
plt.ylabel('DEC / degree', fontsize=16)
plt.savefig('results/smooth_21cm_slice_%d.png' % fi)
plt.close()
