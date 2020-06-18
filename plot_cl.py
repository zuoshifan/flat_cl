from collections import defaultdict
import numpy as np
from scipy import constants as const
from scipy.interpolate import interp1d
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


freq0 = 1420.4 # MHz
freqc = freqs[nf/2] # central frequency, MHz
zc = freq0 / freqc - 1.0 # central redshifts
rc = cosmo.comoving_distance(zc).value # Mpc
rcp = const.c * (1 + zc)**2 / (1.0e6*freq0 * 1.0e3*cosmo.H(zc).value) # Mpc/Hz, cosmo.H(z) in unit km/s/Mpc
Omega = (ras[-1] - ras[0]) * (decs[-1] - decs[0]) # this is true approximately for a small flat sky area


fg_data_fft2 = np.zeros((nf, nra, ndec), dtype=np.complex128)
HI_data_fft2 = np.zeros((nf, nra, ndec), dtype=np.complex128)
for fi in range(nf):
    fg_data_fft2[fi] = Omega * np.fft.fftshift(np.fft.ifft2(fg_data[fi])) # K^2
    HI_data_fft2[fi] = Omega * np.fft.fftshift(np.fft.ifft2(HI_data[fi])) # K^2

Ux = np.fft.fftshift(np.fft.fftfreq(nra, d=ras[1] - ras[0]))
Uy = np.fft.fftshift(np.fft.fftfreq(ndec, d=decs[1] - decs[0]))

fg_lmodes = defaultdict(list)
HI_lmodes = defaultdict(list)
for xi in range(nra):
    for yi in range(ndec):
        U = (Ux[xi]**2 + Uy[yi]**2)**0.5
        l = np.int(np.around(2*np.pi * U))
        # lmodes[l].append(data_fft2[:, xi, yi])
        # lmodes[l/20 * 20 + 10].append(data_fft2[:, xi, yi])
        # lmodes[l/10 * 10 + 5].append(data_fft2[:, xi, yi])
        lbin = 15
        fg_lmodes[l/lbin * lbin + lbin/2].append(fg_data_fft2[:, xi, yi])
        HI_lmodes[l/lbin * lbin + lbin/2].append(HI_data_fft2[:, xi, yi])


dfreq = (freqs[1] - freqs[0]) # MHz
dfreqs = dfreq * np.arange(nf) # MHz
dfreqs1 = dfreqs.copy()
dfreqs1[0] += 1.0e-2 # for log plot
# ls = np.sort(np.array(lmodes.keys()))
fg_Cldf = [[], [], []]
fg_std = [[], [], []]
HI_Cldf = [[], [], []]
HI_std = [[], [], []]
# for l in ls:
for i, l in enumerate([202, 997, 2002]):

    fg_lm = np.array(fg_lmodes[l]).T
    fg_Cl = np.dot(fg_lm, fg_lm.T.conj()).real / fg_lm.shape[1] / Omega # K^2

    HI_lm = np.array(HI_lmodes[l]).T
    HI_Cl = np.dot(HI_lm, HI_lm.T.conj()).real / HI_lm.shape[1] / Omega # K^2
    # print np.allclose(Cl, Cl.T)

    # plot Cl
    plt.figure()
    plt.imshow((l*(l+1)/(2*np.pi)) * 1.0e6*fg_Cl, origin='lower', aspect='auto', extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]]) # mK^2
    cb = plt.colorbar()
    cb.ax.set_ylabel(r'$C_l(\nu_1, \, \nu_2)$ / mK${}^2$', fontsize=16)
    plt.xlabel(r'$\nu_1$ / MHz', fontsize=16)
    plt.ylabel(r'$\nu_2$ / MHz', fontsize=16)
    plt.savefig('results/fg_Cl_%04d.png' % l)
    plt.close()

    plt.figure()
    plt.imshow((l*(l+1)/(2*np.pi)) * 1.0e6*HI_Cl, origin='lower', aspect='auto', extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]]) # mK^2
    cb = plt.colorbar()
    cb.ax.set_ylabel(r'$C_l(\nu_1, \, \nu_2)$ / mK${}^2$', fontsize=16)
    plt.xlabel(r'$\nu_1$ / MHz', fontsize=16)
    plt.ylabel(r'$\nu_2$ / MHz', fontsize=16)
    plt.savefig('results/HI_Cl_%04d.png' % l)
    plt.close()


    for k in range(nf):
        fg_Cldf[i].append(np.diag(fg_Cl, k=k).mean()) # Cl is symmetric
        fg_std[i].append(np.diag(fg_Cl, k=k).std()) # Cl is symmetric
        HI_Cldf[i].append(np.diag(HI_Cl, k=k).mean()) # Cl is symmetric
        HI_std[i].append(np.diag(HI_Cl, k=k).std()) # Cl is symmetric


    # plot Cldf
    plt.figure()
    # interpolate before plot
    x = np.linspace(dfreqs[0], dfreqs[-1], 20000)
    y = interp1d(dfreqs, np.array(fg_Cldf[i]), kind='cubic')(x)
    # plt.semilogx(dfreqs, 1.0e6*np.array(fg_Cldf[i]))
    plt.semilogx(x, 1.0e6*y)
    plt.errorbar(dfreqs1, 1.0e6*np.array(fg_Cldf[i]), yerr=1.0e6*np.array(fg_std[i]), fmt='ro', ecolor='r')
    plt.xlim(0.009, 100)
    plt.xlabel(r'$\Delta \nu$ / MHz', fontsize=16)
    plt.ylabel(r'$C_l(\Delta \nu)$ / mK${}^2$', fontsize=16)
    plt.savefig('results/fg_Cldf_%d.png' % i)
    plt.close()

    plt.figure()
    # interpolate before plot
    x = np.linspace(dfreqs[0], dfreqs[-1], 20000)
    y = interp1d(dfreqs, np.array(HI_Cldf[i]), kind='cubic')(x)
    # plt.semilogx(dfreqs, 1.0e6*np.array(HI_Cldf[i]))
    plt.semilogx(x, 1.0e6*y)
    plt.errorbar(dfreqs1, 1.0e6*np.array(HI_Cldf[i]), yerr=1.0e6*np.array(HI_std[i]), fmt='ro', ecolor='r')
    plt.xlim(0.009, 100)
    plt.xlabel(r'$\Delta \nu$ / MHz', fontsize=16)
    plt.ylabel(r'$C_l(\Delta \nu)$ / mK${}^2$', fontsize=16)
    plt.savefig('results/HI_Cldf_%d.png' % i)
    plt.close()
