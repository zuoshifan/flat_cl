from collections import defaultdict
import numpy as np
from scipy import constants as const
from scipy import linalg as la
import h5py
from astropy.cosmology import Planck13 as cosmo
import spca
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


ras = np.radians(ras) # radians
decs = np.radians(decs) # radians

nf = freqs.shape[0]
nra = ras.shape[0]
ndec = decs.shape[0]

# noise
Tsys = 100 # K
omega = 0.1 # Ae omega = lambda^2
npol = 2
nfeed = 96
nbl = nfeed * (nfeed - 1) / 2
dfreq = 1.0e6*(freqs[1] - freqs[0]) # Hz
dt = 2.0 * 365 * 24 * 3600 / (nra * (2*np.pi / (ras[-1] - ras[0]))) # s
sigma = Tsys * omega / np.sqrt(npol * nbl * dfreq * dt)
print sigma

# generate noise
noise = np.random.normal(loc=0.0, scale=sigma, size=(nf, nra, ndec)) # K


data = fg_data + HI_data + noise


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
# k_para = np.logspace(-2, np.log10(2.0), 200) # Mpc^-1
# k_para = np.linspace(0.1, 1.5, 100) # Mpc^-1
# k_para = np.linspace(0.01, 2.0, 28) # Mpc^-1
# k_perp = []
# Pk2 = []
# ls = np.sort(np.array(lmodes.keys()))
# for l in ls:
for i, l in enumerate([202, 997, 2002]):
    # if l < 150 or l >= 2300:
    #     continue


    lm = np.array(lmodes[l]).T
    Cl = np.dot(lm, lm.T.conj()).real / lm.shape[1] / Omega # K^2
    # print np.allclose(Cl, Cl.T)

    # plot eigen-values of Cl
    e, U = la.eigh(Cl)

    plt.figure()
    plt.semilogy(e[::-1][:12], 'ro')
    plt.xlim(-0.5, 12)
    plt.xlabel('Egen-modes', fontsize=16)
    plt.ylabel('Egenvalues', fontsize=16)
    plt.savefig('results/eigval_%04d.png' % l)
    plt.close()


    # decomp Cl
    L, S = spca.decompose(Cl, rank=4, S=None, lmbda=None, threshold='hard', max_iter=100, tol=1.0e-8, debug=False)

    # plot L, S

    plt.figure()
    plt.imshow((l*(l+1)/(2*np.pi)) * 1.0e6*L, origin='lower', aspect='auto', extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]]) # mK^2
    plt.colorbar()
    plt.xlabel(r'$\nu$ / MHz', fontsize=16)
    plt.ylabel(r'$\nu \, {}^\prime$ / MHz', fontsize=16)
    plt.savefig('results/L_%04d.png' % l)
    plt.close()

    plt.figure()
    plt.imshow((l*(l+1)/(2*np.pi)) * 1.0e6*S, origin='lower', aspect='auto', extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]]) # mK^2
    plt.colorbar()
    plt.xlabel(r'$\nu$ / MHz', fontsize=16)
    plt.ylabel(r'$\nu \, {}^\prime$ / MHz', fontsize=16)
    plt.savefig('results/S_%04d.png' % l)
    plt.close()

    plt.figure()
    plt.imshow((l*(l+1)/(2*np.pi)) * 1.0e6*(Cl - L - S), origin='lower', aspect='auto', extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]]) # mK^2
    plt.colorbar()
    plt.xlabel(r'$\nu$ / MHz', fontsize=16)
    plt.ylabel(r'$\nu \, {}^\prime$ / MHz', fontsize=16)
    plt.savefig('results/N_%04d.png' % l)
    plt.close()

    plt.figure()
    plt.imshow((l*(l+1)/(2*np.pi)) * 1.0e6*(Cl - L), origin='lower', aspect='auto', extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]]) # mK^2
    plt.colorbar()
    plt.xlabel(r'$\nu$ / MHz', fontsize=16)
    plt.ylabel(r'$\nu \, {}^\prime$ / MHz', fontsize=16)
    plt.savefig('results/S+N_%04d.png' % l)
    plt.close()
