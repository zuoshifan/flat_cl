from collections import defaultdict
import numpy as np
from scipy import constants as const
import h5py
from astropy.cosmology import Planck13 as cosmo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


with h5py.File('cube_21cm_smooth.hdf5', 'r') as f:
    data = f['data'][:] # K
    freqs = f['data'].attrs['nu'] # MHz
    ras = f['data'].attrs['ra'] # degree
    decs = f['data'].attrs['dec'] # degree


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


# zs = freq0 / freqs - 1.0 # redshifts

# # get comoving distance
# cd = cosmo.comoving_distance(zs).value # Mpc
# # print cosmo.h
# cd /= cosmo.h # Mpc / h
# # get k_parallel by approximate cd as uniform
# k_paras = np.fft.fftshift(2*np.pi * np.fft.fftfreq(nf, d=(cd[0]-cd[-1])/nf)) # h Mpc^-1
# k_paras = k_paras[nf/2:] # get only positive k_paras


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


# ls = np.array(lmodes.keys())
# print len(ls), ls.min(), ls.max()
# exit()

# ls = []
# ns = []
# for k, v in lmodes.items():
#     # print k, len(v)
#     ls.append(k)
#     ns.append(len(v))

# plt.figure()
# plt.bar(ls, ns)
# plt.savefig('ln_bar.png')
# plt.close()
# exit()

# l = 1975
# l = 805
# l = 85
# l = 750
# print len(lmodes[l])

# print np.array(lmodes[l]).shape

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

    # # plot Cl
    # plt.figure()
    # plt.imshow((l*(l+1)/(2*np.pi)) * 1.0e6*Cl, origin='lower', aspect='auto') # mK^2
    # plt.colorbar()
    # plt.savefig('Cl_%04d.png' %l)
    # plt.close()

    # continue

    # compute pk
    k_perp.append(l / rc) # Mpc^-1
    Cldf = []
    for k in range(nf):
        Cldf.append(np.diag(Cl, k=k).mean()) # Cl is symmetric

    Cldf = np.array(Cldf) # K^2
    # blackman = np.blackman(len(Cldf))
    # Cldf *= blackman

    Pk2.append( rc**2 * rcp * dfreq * np.dot(np.exp(-1.0J * rcp * np.outer(k_para, dfreqs)), 1.0e6*Cldf).real ) # mK^2 Mpc^3

    # Cldf_fft = np.fft.fft(Cldf)[:nf/2]
    # # print Cldf_fft
    # # import pdb; pdb.set_trace()
    # Pk2.append( rc**2 * rcp * 1.0e6 * Cldf_fft.real )

    # # plot Pk2
    # plt.figure()
    # # plt.plot(k_para, Pk2)
    # plt.semilogx(k_para, Pk2)
    # plt.savefig('Pk2.png')
    # plt.close()

k_perp = np.array(k_perp)
Pk2 = np.array(Pk2)
print k_perp.shape, Pk2.shape

# plot Pk2
plt.figure()
# plt.pcolormesh(k_perp, k_para, Pk2.T, vmin=0, vmax=10)
# plt.pcolormesh(k_perp, k_para, Pk2.T, vmin=0, vmax=10)
# plt.pcolormesh(k_perp, k_para, Pk2.T, vmin=-10, vmax=100)
Pklog = np.where(Pk2>0, np.log10(Pk2), 0)
plt.pcolormesh(k_perp, k_para, Pklog.T)
plt.colorbar()
plt.xlim(k_perp[0], 0.3)
plt.ylim(k_para[0], k_para[-1])
plt.savefig('Pk2.png')
plt.close()

# compute Pk
kmodes = defaultdict(list)
for xi, kx in enumerate(k_perp):
    for yi, ky in enumerate(k_para):
        k = (kx**2 + ky**2)**0.5
        kmodes[np.around(k, 2)].append(Pk2[xi, yi])

ks = np.sort(np.array(kmodes.keys()))
Pk = np.array([ np.mean(kmodes[k]) for k in ks ])
D2 = ks**3 * Pk / (2 * np.pi**2)

plt.figure()
# plt.plot(ks, Pk)
# plt.semilogy(ks, Pk)
# plt.semilogy(ks, D2)
plt.loglog(ks, D2)
plt.savefig('Pk1d.png')
plt.close()

exit()


# 3D inverse Fourier transform of cm to get its Fourier modes
# NOTE: approximate the line of sight distances as uniform
cmk3 = np.fft.fftshift((nf*nx*ny) * np.fft.ifftn(cm))
# array to save the 2D Fourier transform in transverse plans of cm
cmk2 = np.zeros_like(cm, dtype=np.complex128)
# array to save kx and ky
kxs = np.zeros((nf, nx)) # h Mpc^-1
kys = np.zeros((nf, ny)) # h Mpc^-1
for fi, z in enumerate(zs):
    # 2D inverse Fourier transform for each z
    cmk2[fi] = np.fft.fftshift((nx * ny) * np.fft.ifft2(cm[fi]))
    # compute kx, ky
    lon_mpc = 1.0e-3 * cosmo.kpc_proper_per_arcmin(z) * (lonra[1] - lonra[0]) * 60
    lon_mpch = lon_mpc / cosmo.h # Mpc / h
    lat_mpc = 1.0e-3 * cosmo.kpc_proper_per_arcmin(z) * (latra[1] - latra[0]) * 60
    lat_mpch = lat_mpc / cosmo.h # Mpc / h
    kxs[fi] = np.fft.fftshift(2*np.pi * np.fft.fftfreq(nx, d=lon_mpch/nx)) # h Mpc^-1
    kys[fi] = np.fft.fftshift(2*np.pi * np.fft.fftfreq(ny, d=lat_mpch/ny)) # h Mpc^-1

# use only central frequency kx, ky to approximate all freqs
kxs = kxs[nf/2]
kys = kys[nf/2]


factor = 0.5
kpbin = int(factor * np.sqrt((ny/2.0)**2 + (nx/2.0)**2)) # bin for k_perp
# only use the central freq ks to bin
k_bins = np.linspace(0, (kpbin+2)/(kpbin+1)*np.sqrt(kxs[0]**2 + kys[0]**2), kpbin+1)
k_perps = np.array([ (k_bins[i] + k_bins[i+1])/2 for i in range(kpbin) ])
# print k_perps
# get the corresponds kx, ky in each bin
kpmodes = defaultdict(list)
for yi in range(ny):
    for xi in range(nx):
        # get the bin index
        bi = np.searchsorted(k_bins, np.sqrt(kxs[xi]**2 + kys[yi]**2))
        # drop (0, 0) mode
        if bi == 0:
            continue
        kpmodes[bi-1].append((yi, xi))


Pk3 = (cmk3 * cmk3.conj()).real # K^2, the true Pk(kz, ky, kx) for comparison
Pk2 = np.zeros((kpbin, nf)) # K^2, to save true Pk(k_perp, k_para)
Pkk = np.zeros((kpbin, nf, nf)) # K^2, to save all Pkk
Pkkd = np.zeros((kpbin, nf)) # K^2, to save diagonal of all Pkk
for bi in range(kpbin):
    # print bi, len(kpmodes[bi])
    Tk2 = np.zeros((nf, len(kpmodes[bi])), dtype=cmk2.dtype)
    Tk3 = np.zeros((nf, len(kpmodes[bi])), dtype=cmk3.dtype)
    for i, (yi, xi) in enumerate(kpmodes[bi]):
        Pk2[bi] += Pk3[:, yi, xi]
        Tk2[:, i] = cmk2[:, yi, xi]
        Tk3[:, i] = cmk3[:, yi, xi]
    Pk2[bi] /= len(kpmodes[bi])

    # compute freq covariance matrix of this bin
    corr2 = np.dot(Tk2, Tk2.T.conj()) / len(kpmodes[bi])

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(corr2.real, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(corr2.imag, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.savefig(out_dir + 'corr2zz_%d.png' % bi)
    # plt.close()


    # compute and check Pk(k_perp, k_para, k'_para)
    corr2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(nf * np.fft.ifft(corr2, axis=0), axes=0), axis=1), axes=1)

    corr3 = np.dot(Tk3, Tk3.T.conj()) / len(kpmodes[bi])

    assert np.allclose(corr2, corr3)

    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(corr2.real, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(corr2.imag, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(corr3.real, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(corr3.imag, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.savefig(out_dir + 'corr2_and_corr3_%d.png' % bi)
    # plt.close()


    # plt.figure()
    # plt.subplot(211)
    # plt.plot(np.diag(corr2.real))
    # plt.subplot(212)
    # plt.plot(np.diag(corr2.imag))
    # plt.savefig(out_dir + 'corr2_and_corr3_diag_%d.png' % bi)
    # plt.close()

    Pkk[bi] = corr2.real
    Pkkd[bi] = np.diag(corr2.real)

# check
assert np.allclose(Pkkd, Pk2)

# plot Pkkd and Pk2
plt.figure()
plt.subplot(121)
# times 1.0e6 to mK^2
# plt.imshow(1000 * Pkkd.T, origin='lower', aspect='auto', vmax=100)
im = 1.0e6 * Pkkd.T[nf/2:, :] # mK^2
m, n = im.shape
plt.pcolormesh(im, vmax=100000)
plt.xlim(0, n)
plt.ylim(0, m)
plt.colorbar()
plt.subplot(122)
plt.imshow(im, origin='lower', aspect='auto', interpolation='nearest', vmax=100000)
plt.colorbar()
plt.savefig(out_dir + 'Pkkd.png')
plt.close()


# bin to get Pk
kbin = int(factor * np.sqrt((nf/2.0)**2 + (ny/2.0)**2 + (nx/2.0)**2))
k_bins = np.linspace(np.sqrt(k_paras[0]**2 + k_perps[0]**2), (kbin+2)/(kbin+1)*np.sqrt(k_paras[-1]**2 + k_perps[-1]**2), kbin+1)
ks = np.array([ (k_bins[i] + k_bins[i+1])/2 for i in range(kbin) ])
# print ks
# get the corresponds k_paras, k_perp in each bin
kmodes = defaultdict(list)
for yi in range(kpbin): # for perp
    for xi in range(nf/2): # for para
        # drop 0 mode of k_paras
        if xi == 0:
            continue
        # get the bin index
        bi = np.searchsorted(ks, np.sqrt(k_perps[yi]**2 + k_paras[xi]**2))
        kmodes[bi].append((yi, xi))


Pk = np.zeros((kbin,)) # K^2, to save all Pk
for bi in range(kbin):
    # print bi, len(kmodes[bi])
    for i, (yi, xi) in enumerate(kmodes[bi]):
        Pk[bi] += Pkkd[yi, xi+nf/2]
    Pk[bi] /= len(kmodes[bi])

# plog Pk
plt.figure()
plt.loglog(ks, 1.0e6 * ks**3 * Pk / (2 * np.pi**2))
plt.xlabel(r'$k \ [h \, \rm{Mpc}^{-1}]$', fontsize=14)
plt.ylabel(r'$\Delta(k)^2 \ [\rm{mK}^2]$', fontsize=14)
plt.savefig(out_dir + 'Pk.png')
plt.close()


# bin input 21cm to get Pk
Pk_input = np.zeros((kbin,)) # to save all input Pk
for bi in range(kbin):
    modes = []
    cnt = 0
    for zi in range(nf/2, nf):
        for yi in range(ny/2, ny):
            for xi in range(nx/2, nx):
                k = np.sqrt(k_paras[zi-nf/2]**2 + kys[yi]**2 + kxs[xi]**2)
                if k_bins[bi] <= k and k < k_bins[bi+1]:
                    cnt += 1
                    Pk_input[bi] += Pk3[zi, yi, xi]
    Pk_input[bi] /= cnt

# check Pk and Pk_input
# plog Pk and Pk_input
plt.figure()
plt.loglog(ks, 1.0e6 * ks**3 * Pk / (2 * np.pi**2), label='Pk')
plt.loglog(ks, 1.0e6 * ks**3 * Pk_input / (2 * np.pi**2), label='Pk_input')
plt.xlabel(r'$k \ [h \, \rm{Mpc}^{-1}]$', fontsize=14)
plt.ylabel(r'$\Delta(k)^2 \ [\rm{mK}^2]$', fontsize=14)
plt.legend()
plt.savefig(out_dir + 'Pk_check.png')
plt.close()