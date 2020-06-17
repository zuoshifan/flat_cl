from collections import defaultdict
import numpy as np
from scipy import constants as const
import h5py
from astropy.cosmology import Planck13 as cosmo
from cora.signal import corr21cm
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

# theoretical power spectrum from cora
corr = corr21cm.Corr21cm()
z1 = freq0 / freqs[-1] - 1.0 # lower redshifts
z2 = freq0 / freqs[0] - 1.0 # higher redshifts
# k_vec = np.linspace(0.01, 2, 2000) # Mpc^-1
k_vec = np.logspace(-2, np.log10(2.0), 2000) # Mpc^-1
Pk1d = corr.powerspectrum_1D(k_vec, z1, z2, 256)


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
k_para = np.linspace(0.008, 2.0, 2000) # Mpc^-1
k_perp = []
Pk2 = []
ls = np.sort(np.array(lmodes.keys()))
lused = []
Cldfs = []
for l in ls:
    # if l < 150 or l >= 2300:
    #     continue

    lused.append(l)

    lm = np.array(lmodes[l]).T
    Cl = np.dot(lm, lm.T.conj()).real / lm.shape[1] / Omega # K^2
    # print np.allclose(Cl, Cl.T)

    # decomp Cl
    L, S = spca.decompose(Cl, rank=3, S=None, lmbda=None, threshold='hard', max_iter=100, tol=1.0e-8, debug=False)


    Cl = Cl - L

    # compute pk
    k_perp.append(l / rc) # Mpc^-1
    Cldf = []
    for k in range(nf):
        Cldf.append(np.diag(Cl, k=k).mean()) # Cl is symmetric

    Cldf = np.array(Cldf) # K^2
    # blackman = np.blackman(len(Cldf))
    # Cldf *= blackman

    Cldfs.append(Cldf)


    Pk2.append( rc**2 * rcp * dfreq * np.dot(np.exp(-1.0J * rcp * np.outer(k_para, dfreqs)), 1.0e6*Cldf).real ) # mK^2 Mpc^3



k_perp = np.array(k_perp)
Pk2 = np.array(Pk2)
print k_perp.shape, Pk2.shape

# # compute Pk
# nbin = 100
# k_bounds = np.logspace(np.log10(0.02), np.log10(1.0), nbin+1)
# k_starts = k_bounds[:-1]
# k_ends = k_bounds[1:]
# k_centers = 0.5*(k_starts + k_ends)
# Pk = np.zeros_like(k_centers)
# Pki = np.zeros_like(k_centers, dtype=np.int)
# for xi, kx in enumerate(k_perp):
#     for yi, ky in enumerate(k_para):
#         k = (kx**2 + ky**2)**0.5
#         for si in range(nbin):
#             if k >= k_bounds[si] and k < k_bounds[si+1]:
#                 Pki[si] += 1
#                 Pk[si] += Pk2[xi, yi]

#         # if k <= k_bounds[0] and k >= k_bounds[-1]:
#         #     continue
#         # si = np.searchsorted(k_bounds, k) - 1
#         # # if si == 20:
#         # #     import pdb; pdb.set_trace()
#         # Pki[si] += 1
#         # Pk[si] += Pk2[xi, yi]

# Pk = Pk / Pki

kmodes = defaultdict(list)
for xi, kx in enumerate(k_perp):
    for yi, ky in enumerate(k_para):
        k = (kx**2 + ky**2)**0.5
        kmodes[np.around(k, 2)].append(Pk2[xi, yi])

ks = np.sort(np.array(kmodes.keys()))
Pk = np.array([ np.mean(kmodes[k]) for k in ks ])
Pk = Pk[ks<0.5]
ks = ks[ks<0.5]
D2 = ks**3 * Pk / (2 * np.pi**2)

plt.figure()
# plt.plot(ks, Pk)
# plt.semilogy(ks, Pk)
# plt.semilogy(ks, D2)
# plt.loglog(ks, D2)
plt.loglog(k_vec, 1.0e6*Pk1d, 'r', label='theoretical')
plt.loglog(ks, Pk, 'g', label='recovered')
# plt.loglog(k_centers, Pk)
plt.legend(fontsize=15)
plt.xlim(0.01, 1.0)
plt.ylim(0.1, 1000.0)
plt.xlabel(r'$k$ / Mpc${}^{-1}$', fontsize=16)
plt.ylabel(r'$P(k)$ / mK${}^2$Mpc${}^{3}$', fontsize=16)
plt.savefig('Pk1d_decomp1.png')
plt.close()
