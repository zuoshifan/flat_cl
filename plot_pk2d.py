from collections import defaultdict
import numpy as np
from scipy import constants as const
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
k_para = np.linspace(0.01, 2.0, 28) # Mpc^-1
k_perp = []
Pk2 = []
ls = np.sort(np.array(lmodes.keys()))
lused = []
Cldfs = []
for l in ls:
    if l < 150 or l >= 2300:
        continue

    lused.append(l)

    lm = np.array(lmodes[l]).T
    Cl = np.dot(lm, lm.T.conj()).real / lm.shape[1] / Omega # K^2
    # print np.allclose(Cl, Cl.T)

    # decomp Cl
    L, S = spca.decompose(Cl, rank=4, S=None, lmbda=None, threshold='hard', max_iter=100, tol=1.0e-8, debug=False)


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

    # Cldf_fft = np.fft.fft(Cldf)[:nf/2]
    # # print Cldf_fft
    # # import pdb; pdb.set_trace()
    # Pk2.append( rc**2 * rcp * 1.0e6 * Cldf_fft.real )


# plot Cldf
plt.figure()
# import pdb; pdb.set_trace()
# plt.pcolormesh(np.array(lused), dfreqs, np.array(Cldfs))
ndf = 25
dfs = 1.0e-6 * dfreqs[:ndf+1] # MHz
dfs[0] += 0.01
dfs = np.log10(dfs)
lused = np.log10(np.array(lused))
plt_data = 1.0e6*np.array(Cldfs)[:, :ndf].T # mK^2
# plt._data = np.where(plt_data > 0, plt_data, 0.1 * plt_data[plt_data>0].min())
# plt_data[plt_data<=0] = plt_data[plt_data>0].min()
# print plt_data.shape, plt_data.min()
# plt.pcolormesh(np.array(lused), dfs, np.log10(plt_data), vmin=-0.0e-13, vmax=2.0e-13)
# plt.pcolormesh(np.array(lused), dfs, np.log10(plt_data))
# plt.pcolormesh(lused, dfs, plt_data, vmax=8.0e-13)
plt.pcolormesh(lused, dfs, plt_data)
cb = plt.colorbar()
cb.set_label(r'$C_l(\Delta \nu)$ / mK${}^2$$', fontsize=16)
plt.xlim(lused[0], 3.2)
plt.ylim(dfs[0], 1.0)
ax = plt.gca()
ax.set_xticks([np.log10(200.0), np.log10(500.0), np.log10(1000.0), np.log10(1500.0)])
ax.set_xticklabels(['200', '500', '1000', '1500'])
ax.set_yticks([-2.0, -1.0, 0.0, 1.0])
ax.set_yticklabels(['0.01', '0.1', '1', '10'])
plt.xlabel(r'$l$', fontsize=16)
plt.ylabel(r'$\Delta \nu$ / MHz', fontsize=16)
plt.savefig('results/Cldf.png')
plt.close()


k_perp = np.array(k_perp)
Pk2 = np.array(Pk2)
print k_perp.shape, Pk2.shape

# # plot Pk2
# plt.figure()
# # plt.pcolormesh(k_perp, k_para, Pk2.T, vmin=0, vmax=10)
# # plt.pcolormesh(k_perp, k_para, Pk2.T, vmin=0, vmax=10)
# # plt.pcolormesh(k_perp, k_para, Pk2.T, vmin=-10, vmax=100)
# # Pklog = np.where(Pk2>0, np.log10(Pk2), 0)
# Pk2_1 = Pk2.copy() # mK^2 Mpc^3
# Pk2_1[Pk2_1<=0] = Pk2_1[Pk2_1>0].min()
# Pklog = np.log10(Pk2_1)
# # plt.pcolormesh(k_perp, k_para, Pklog.T)
# k_perp_log = np.log10(k_perp)
# k_para_log = np.log10(k_para)
# # plt.pcolormesh(k_perp_log, k_para_log, Pk2.T, vmin=-10, vmax=50)
# plt.pcolormesh(k_perp_log, k_para_log, Pklog.T, vmin=-2, vmax=3)
# cb = plt.colorbar()
# cb.set_ticks([-2, -1, 0, 1, 2, 3])
# cb.set_ticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', ])
# cb.set_label(r'$P(k_\perp, \, k_\parallel)$ / mK${}^2$Mpc${}^3$', fontsize=16)
# plt.xlim(-1.22, -0.18)
# plt.ylim(-2.0, 0.2)
# ax = plt.gca()
# ax.set_xticks([np.log10(0.1), np.log10(0.2), np.log10(0.5)])
# ax.set_xticklabels(['0.1', '0.2', '0.5'])
# ax.set_yticks([-2.0, -1.0, 0.0])
# ax.set_yticklabels(['0.01', '0.1', '1'])
# plt.xlabel(r'$k_\perp$ / Mpc${}^{-1}$', fontsize=16)
# plt.ylabel(r'$k_\parallel$ / Mpc${}^{-1}$', fontsize=16)
# plt.savefig('Pk2_decomp1.png')
# plt.close()

# exit()

# # compute Pk
# kmodes = defaultdict(list)
# for xi, kx in enumerate(k_perp):
#     for yi, ky in enumerate(k_para):
#         k = (kx**2 + ky**2)**0.5
#         kmodes[np.around(k, 2)].append(Pk2[xi, yi])

# ks = np.sort(np.array(kmodes.keys()))
# Pk = np.array([ np.mean(kmodes[k]) for k in ks ])
# D2 = ks**3 * Pk / (2 * np.pi**2)

# plt.figure()
# # plt.plot(ks, Pk)
# # plt.semilogy(ks, Pk)
# # plt.semilogy(ks, D2)
# plt.loglog(ks, D2)
# plt.savefig('Pk1d_decomp1.png')
# plt.close()
