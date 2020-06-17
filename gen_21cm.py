import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import constants as const
import h5py
from cora.signal import corr21cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


corr = corr21cm.Corr21cm()
corr.x_width = 20.0 # degree
corr.y_width = 20.0 # degree
corr.x_num = 256
corr.y_num = 256
corr.nu_lower = 700.0 # MHz
corr.nu_upper = 800.0 # MHz
corr.nu_num = 256

freqs = np.linspace(corr.nu_lower, corr.nu_upper, corr.nu_num) # MHz

cube = corr.getfield() # (nu, ra, dec)
print cube.shape

D = 100.0 # m
# lmbda = 0.4 # m
lmbda = const.c / (1.0e6 * freqs) # m
fwhm = np.degrees(1.22 * lmbda / D) # degree
sigma = fwhm / np.sqrt(8.0 * np.log(2)) # degree
sigma = sigma / (corr.x_width / corr.x_num) # number of grids
print sigma.shape

cube1 = np.zeros_like(cube)
for fi in range(corr.nu_num):
    cube1[fi] = gaussian_filter(cube[fi], sigma[fi])

cube2 = cube1[:, 64:-64, 64:-64]
print cube2.shape

# save data
with h5py.File('cube_21cm_smooth.hdf5', 'w') as f:
    f.create_dataset('data', data=cube2)
    f['data'].attrs['dims'] = '(nu, ra, dec)'
    f['data'].attrs['nu'] = freqs
    f['data'].attrs['ra'] = np.linspace(-5, 5, 128)
    f['data'].attrs['dec'] = np.linspace(-5, 5, 128)



# plt.figure()
# plt.imshow(cube[128], origin='lower', aspect='equal')
# plt.colorbar()
# plt.savefig('21cm_slice_128.png')
# plt.close()

# plt.figure()
# plt.imshow(cube1[128], origin='lower', aspect='equal')
# plt.colorbar()
# plt.savefig('smooth_21cm_slice_128.png')
# plt.close()