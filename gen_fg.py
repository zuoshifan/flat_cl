import os
import numpy as np
from scipy import interpolate as interp
from scipy.ndimage import gaussian_filter
from scipy import constants as const
import h5py
import healpy as hp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# with h5py.File('sim_galaxy_256_700_800_4.hdf5', 'r') as f:
with h5py.File('/public/zuoshifan/workspace/sky_map/sim_galaxy_512_700_800_256.hdf5', 'r') as f:
    hmap = f['map'][:, 0, :]
    freqs = f['index_map/freq'][:] # MHz

freqs = freqs['centre'] # and width

nf = len(freqs)

# theta0 = 2*np.pi / 3 # [0, pi]
# phi0 = np.pi # [0, 2*pi]
theta0 = np.pi / 4 # [0, pi]
phi0 = np.pi # [0, 2*pi]

vec0 = hp.ang2vec(theta0, phi0)

nside = hp.npix2nside(hmap.shape[1])

pis = hp.query_disc(nside, vec0, np.radians(20.0), inclusive=True)

ra0 = np.degrees(phi0 - np.pi) # degree
dec0 = np.degrees(np.pi/2 - theta0) # degree
print ra0, dec0

theta, phi = hp.pix2ang(nside, pis)
ra = np.degrees(phi - np.pi)
dec = np.degrees(np.pi/2 - theta)

# # scatter plot of ra, dec
# plt.figure()
# plt.scatter(ra, dec)
# plt.plot([ra0], [dec0], 'ro')
# plt.savefig('ra_dec_scatter.png')
# plt.close()

fg_cube = np.zeros((nf, 256, 256), dtype=hmap.dtype) # to save the cube

points = np.vstack([ra, dec]).T

ra_low = ra0 - 10.0
ra_high = ra0 + 10.0
dec_low = dec0 - 10.0
dec_high = dec0 + 10.0
grid_ra, grid_dec = np.mgrid[ra_low:ra_high:256j, dec_low:dec_high:256j]
for fi in range(nf):
    interp_map = interp.griddata(points, hmap[fi, pis], (grid_ra, grid_dec), method='cubic')
    fg_cube[fi, :, :] = interp_map


D = 100.0 # m
# lmbda = 0.4 # m
lmbda = const.c / (1.0e6 * freqs) # m
fwhm = np.degrees(1.22 * lmbda / D) # degree
sigma = fwhm / np.sqrt(8.0 * np.log(2)) # degree
sigma = sigma / (20.0 / 256) # number of grids
print sigma.shape

cube1 = np.zeros_like(fg_cube)
for fi in range(nf):
    cube1[fi] = gaussian_filter(fg_cube[fi], sigma[fi])

cube2 = cube1[:, 64:-64, 64:-64]
print cube2.shape

# save data
with h5py.File('cube_galaxy_smooth.hdf5', 'w') as f:
    f.create_dataset('data', data=cube2)
    f['data'].attrs['dims'] = '(nu, ra, dec)'
    f['data'].attrs['nu'] = freqs
    f['data'].attrs['ra'] = np.linspace(ra0 - 5.0, ra0 + 5.0, 128)
    f['data'].attrs['dec'] = np.linspace(dec0 - 5.0, dec0 + 5.0, 128)



fi = nf / 2

# plot several slices
plt.figure()
plt.imshow(fg_cube[fi, :, :], origin='lower', aspect='equal', extent=[ra_low, ra_high, dec_low, dec_high])
plt.colorbar()
plt.savefig('galaxy_slice_%d.png' % fi)
plt.close()

# plot several slices
plt.figure()
plt.imshow(cube2[fi, :, :], origin='lower', aspect='equal', extent=[ra_low, ra_high, dec_low, dec_high])
plt.colorbar()
plt.savefig('smooth_galaxy_slice_%d.png' % fi)
plt.close()
