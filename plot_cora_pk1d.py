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

freq0 = 1420.4 # MHz
z1 = freq0 / corr.nu_upper - 1.0 # lower redshifts
z2 = freq0 / corr.nu_lower - 1.0 # higher redshifts

k = np.linspace(0.01, 2, 1000) # Mpc^-1

Pk1d = corr.powerspectrum_1D(k, z1, z2, 256)
print Pk1d.shape

# plot Pk1d
plt.figure()
plt.loglog(k, 1.0e6*Pk1d)
plt.savefig('cora_pk1d.png')
plt.close()

exit()

freqc = freqs[nf/2] # central frequency, MHz
zc = freq0 / freqc - 1.0 # central redshifts
rc = cosmo.comoving_distance(zc).value # Mpc
rcp = const.c * (1 + zc)**2 / (1.0e6*freq0 * 1.0e3*cosmo.H(zc).value) # Mpc/Hz, cosmo.H(z) in unit km/s/Mpc
Omega = (ras[-1] - ras[0]) * (decs[-1] - decs[0]) # this is true approximately for a small flat sky area




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
