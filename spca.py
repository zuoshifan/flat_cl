import numpy as np
from scipy import linalg as la


# def mad(a):
#     """Median absolute deviation."""
#     return np.median(np.abs(a - np.median(a)))

def mad(a):
    """Median absolute deviation."""

    def madr(x):
        """Median absolute deviation of a real array."""
        return np.median(np.abs(x - np.median(x)))

    if np.isrealobj(a):
        return madr(a)
    else:
        return np.sqrt(madr(a.real)**2 + madr(a.imag)**2)

def MAD(a):
    """Median absolute deviation divides 0.6745."""
    return mad(a) / 0.6745


def l0_norm(a):
    """Return the :math:`l_0`-norm (i.e., number of non-zero elements) of an array."""
    return len(np.where(a.flatten() != 0.0)[0])


def l1_norm(a):
    """Return the :math:`l_1`-norm of an array."""
    return np.sum(np.abs(a))


def truncate(a, lmbda):
    """Hard thresholding operator, which works for both real and complex array."""
    return a * (np.abs(a) > lmbda)


def sign(a):
    """Sign of an array, which works for both real and complex array."""
    if np.isrealobj(a):
        return np.sign(a)
    else:
        return np.exp(1.0J * np.angle(a))


def shrink(a, lmbda):
    """Soft thresholding operator, which works for both real and complex array."""
    return sign(a) * np.maximum(np.abs(a) - lmbda, 0.0) # work for both real and complex


def decompose(M, rank=1, S=None, lmbda=None, threshold='hard', max_iter=100, tol=1.0e-8, debug=False):
    """Stable principal component decomposition a matrix."""

    d1, d2 = M.shape

    # if np.isrealobj(M):
    #     real = True
    # else:
    #     real = False

    if d1 == d2 and np.allclose(M, M.T.conj()):
        hermitian = True
    else:
        hermitian = False

    if lmbda is None:
        fixed_lmbda = False
    else:
        fixed_lmbda = True

    if threshold == 'hard':
        hard  = True
    elif threshold == 'soft':
        hard = False
    else:
        raise ValueError('Unknown thresholding method: %s' % threshold)

    if (S is None) or (S.shape != M.shape):
        # initialize S as zero
        S = np.zeros_like(M)
    else:
        S = S.astype(M.dtype)

    S_old = S
    L_old = np.zeros_like(M)
    MF= la.norm(M, ord='fro')

    for it in xrange(max_iter):
        if hermitian:
            # compute only the largest rank eigen values and vectors, which is faster
            s, U = la.eigh(M - S, eigvals=(d1-rank, d1-1))
            # threshold s to make V0 Hermitian positive semidefinite
            # L = np.dot(U[:, -rank:]*np.maximum(s[-rank:], 0), U[:, -rank:].T.conj())
            L = np.dot(U*np.maximum(s, 0), U.T.conj())
        else:
            U, s, VT = la.svd(M - S, full_matrices=False)
            L = np.dot(U[:, :rank]*s[:rank], VT[:rank])

        # U, s, VT = la.svd(M - S, full_matrices=False)
        # L = np.dot(U[:, :rank]*s[:rank], VT[:rank])
        # # plot s
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.semilogy(s)
        # plt.semilogy(s, 'ro')
        # plt.xlim(-1, len(s)+1)
        # plt.savefig('s_%d.png' % it)
        # plt.close()

        res = M - L

        if not fixed_lmbda: # and it >= 1:
            # the universal threshold: sigma * (2 * log(d*d))**0.5
            th = (2.0 * np.log10(d1 * d2))**0.5 * MAD(res)
            # if real and (not hard): # real, soft-thresholding
            #     lmbda = th
            # elif real and hard: # real, hard-thresholding
            #     lmbda = 2**0.5 * th
            # elif (not real) and (not hard): # complex, soft-thresholding
            #     lmbda = 2**-0.5 * th
            # else: # complex, hard-thresholding
            #     lmbda = th
            if hard: # hard-thresholding
                lmbda = 2**0.5 * th
            else: # soft-thresholding
                lmbda = th

            if debug:
                print 'lmbda:', lmbda

        # compute new S
        if hard:
            S = truncate(res, lmbda)
        else:
            S = shrink(res, lmbda)

        tol1 = (la.norm(L - L_old, ord='fro') + la.norm(S - S_old, ord='fro')) / MF
        if tol1 < tol:
            if debug:
                print 'Converge when iteration: %d with tol: %g < %g' % (it, tol1, tol)
            break

        L_old = L
        S_old = S

    else:
        print 'Exit with max_iter: %d, tol: %g >= %g' % (it, tol1, tol)

    return L, S