from collections import defaultdict
import numpy as np
from scipy import constants as const
import h5py
from astropy.cosmology import Planck13 as cosmo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


with h5py.File('cube_21cm_smooth.hdf5', 'r') as f:
    # data = f['data'][:] # K
    freqs = f['data'].attrs['nu'] # MHz
    ras = f['data'].attrs['ra'] # degree
    decs = f['data'].attrs['dec'] # degree


ras = np.radians(ras) # radians
decs = np.radians(decs) # radians

nf = freqs.shape[0]
nra = ras.shape[0]
ndec = decs.shape[0]

Tsys = 100 # K
omega = 0.1
npol = 2
nfeed = 96
nbl = nfeed * (nfeed - 1) / 2
dfreq = 1.0e6*(freqs[1] - freqs[0]) # Hz
dt = 2.0 * 365 * 24 * 3600 / (nra * (2*np.pi / (ras[-1] - ras[0]))) # s
sigma = Tsys * omega / np.sqrt(npol * nbl * dfreq * dt)
print sigma

# generate noise
noise = np.random.normal(loc=0.0, scale=sigma, size=(nf, nra, ndec)) # K
data = noise

freq0 = 1420.4 # MHz
freqc = freqs[nf/2] # central frequency, MHz
zc = freq0 / freqc - 1.0 # central redshifts
rc = cosmo.comoving_distance(zc).value # Mpc
rcp = const.c * (1 + zc)**2 / (1.0e6*freq0 * 1.0e3*cosmo.H(zc).value) # Mpc/Hz, cosmo.H(z) in unit km/s/Mpc
Omega = (ras[-1] - ras[0]) * (decs[-1] - decs[0]) # this is true approximately for a small flat sky area


data_fft2 = np.zeros((nf, nra, ndec), dtype=np.complex128)
for fi in range(nf):
    data_fft2[fi] = Omega * np.fft.fftshift(np.fft.ifft2(data[fi])) # K^2
    # # for check
    # print np.allclose(np.fft.fft2(np.fft.ifftshift(data_fft2[fi])).real / Omega, data[fi])

Ux = np.fft.fftshift(np.fft.fftfreq(nra, d=ras[1] - ras[0]))
Uy = np.fft.fftshift(np.fft.fftfreq(ndec, d=decs[1] - decs[0]))

lmodes = defaultdict(list)
for xi in range(nra):
    for yi in range(ndec):
        U = (Ux[xi]**2 + Uy[yi]**2)**0.5
        l = np.int(np.around(2*np.pi * U))
        # lmodes[l].append(data_fft2[:, xi, yi])
        # lmodes[l/20 * 20 + 10].append(data_fft2[:, xi, yi])
        # lmodes[l/10 * 10 + 5].append(data_fft2[:, xi, yi])
        lbin = 15
        lmodes[l/lbin * lbin + lbin/2].append(data_fft2[:, xi, yi])


dfreq = 1.0e6*(freqs[1] - freqs[0]) # Hz
dfreqs = dfreq * np.arange(nf) # Hz
# k_para = np.logspace(-1, np.log10(1.5), 100) # Mpc^-1
# k_para = np.linspace(0.1, 1.5, 100) # Mpc^-1
k_para = np.linspace(0.1, 1.0, 200) # Mpc^-1
# k_para = np.fft.fftfreq(nf, d=dfreq * rcp / (2*np.pi))[:nf/2] # Mpc^-1
k_perp = []
Pk2 = []
ls = np.sort(np.array(lmodes.keys()))
for l in ls:
    if l < 150 or l >= 2300:
        continue

    lm = np.array(lmodes[l]).T
    Cl = np.dot(lm, lm.T.conj()).real / lm.shape[1] / Omega # K^2
    # print np.allclose(Cl, Cl.T)

    # plot Cl
    plt.figure()
    plt.imshow((l*(l+1)/(2*np.pi)) * 1.0e6*Cl, origin='lower', aspect='auto') # mK^2
    plt.colorbar()
    plt.savefig('Cl_noise/Cl_%04d.png' %l)
    plt.close()
