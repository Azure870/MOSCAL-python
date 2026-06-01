r"""
    Decompose the J(\omega) in the frequency domain using the AAA algorithm.
    The result is a set of poles (\gamma_k) and residues (\eta_k) that approximate the J(\omega) = \sum_k \eta_k / (\gamma_k - i * \omega) in the frequency domain.
"""

import numpy as np
from scipy import linalg as LA
from scipy import sparse


def prz(z, f, w):
    r"""
    Compute the rational function at the frequency points.
    Output:
        expn: 1D array of poles, \gamma_k.
        etal: 1D array of residues, \eta_k.
    Reference:
        This part of code is rewritten based on the MATLAB code provided by:
            The AAA Algorithm for Rational Approximation. DOI: 10.1137/16M1106122.
    Note:
        this part of code should only be used in the aaa.py.
    """
    B = np.eye(len(w) + 1)
    B[0, 0] = 0
    E = np.zeros_like(B)
    for i in range(len(z)):
        E[i + 1, i + 1] = z[i]
        E[i + 1, 0] = 1
        E[0, i + 1] = w[i]
    pol = LA.eigvals(E, B)
    pol = pol[np.isfinite(pol)]
    dz = 1e-5 * np.exp(2.0j * np.pi * np.arange(1, 5) / 4)
    res = np.repeat(pol[:, np.newaxis], repeats=[4], axis=1)[..., :] + dz
    zv = (res.T).flatten()
    CC = 1 / (np.repeat(zv[:, np.newaxis], repeats=[len(z)], axis=1)[..., :] - z)
    r = (CC @ (w * f)) / (CC @ w)

    # # for ii in np.where(np.isinf(r))[0]:
    # for ii in np.where(np.abs(r) > 100)[0]:
    #     print(ii)
    #     r[ii] = f[np.where(zv[ii] == z)]

    r = r.reshape(np.shape(res)[1], np.shape(res)[0]).T
    res = r @ dz.T / 4

    # E = np.zeros_like(B)
    # for i in range(len(z)):
    #     E[i + 1, i + 1] = z[i]
    #     E[i + 1, 0] = 1
    #     E[0, i + 1] = w[i] * f[i]
    # zer = LA.eigvals(E, B)
    # zer = zer[np.isfinite(zer)]

    etal_aaa = -1.0j * res
    expn_aaa = 1.0j * pol
    print(expn_aaa)
    # only keep the poles with positive imaginary part.
    etal_aaa = etal_aaa[np.real(expn_aaa) > 0]
    expn_aaa = expn_aaa[np.real(expn_aaa) > 0]

    return etal_aaa, expn_aaa


def aaa_fit_freq(
    Z: np.ndarray,
    F: np.ndarray,
    tol: float = 1e-8,
    max_item: int = 100,
):
    r"""
    Decompose the J(\omega) in the frequency domain using the AAA algorithm.
    Input:
        Z: 1D array of frequency points.
        F: 1D array of spectral density, J(\omega).
        tol: Tolerance for the convergence of the AAA algorithm.
        max_item: Maximum number of poles.
            Force to stop even if the error is not smaller than tol.
    Output:
        expn: 1D array of poles, \gamma_k.
        etal: 1D array of residues, \eta_k.
    Reference:
        This part of code is rewritten based on the MATLAB code provided by:
            The AAA Algorithm for Rational Approximation. DOI: 10.1137/16M1106122.
    Note:
        The result may not be paired, and need to be paired manually under the Bosonic bath.
    """
    print(
        r"""
          Decompose the J(\omega) in the frequency domain using the AAA algorithm.
          Please cite the following papers:
            The AAA Algorithm for Rational Approximation. 
            DOI: 10.1137/16M1106122.
            
            Taming Quantum Noise for Efficient Low Temperature Simulations of Open Quantum Systems. 
            DOI: 10.1103/PhysRevLett.129.230601
            
            Efficient low temperature simulations for fermionic reservoirs with the hierarchical equations of motion method: Application to the Anderson impurity model. 
            DOI: 10.1103/PhysRevB.107.195429
        """
    )

    M = len(Z)
    R = np.mean(F)
    SF = sparse.diags(F, 0)
    J = np.arange(M)
    z = []
    f = []
    C = []
    mmax = 2 * max_item
    C = np.zeros((M, 1))
    err_vec = np.zeros((mmax))

    for m in range(mmax):
        j = np.argmax(abs(F - R))
        z.append(Z[j])
        f.append(F[j])
        J = np.delete(J, J == j)
        C[:, m] = 1 / (Z - Z[j])
        C[j, m] = 0
        Sf = np.diag(f)
        A = SF @ C[:, : m + 1] - C[:, : m + 1] @ Sf
        _, _, V = LA.svd(
            A[J, :],
            full_matrices=False,
            lapack_driver="gesvd",
        )
        w = V[m, :]
        N = C[:, : m + 1] @ (w * f)
        D = C[:, : m + 1] @ (w)
        R = F.copy()
        R[J] = N[J] / D[J]
        err_vec[m] = np.linalg.norm(F - R, np.inf)
        if err_vec[m] <= tol * np.linalg.norm(F, np.inf):
            if m % 2 == 0:
                break
        else:
            print(f"{m+1}# of poles, Error = {err_vec[m]}")
        C = np.pad(C, ((0, 0), (0, 1)), mode="constant", constant_values=0)

    return prz(z, f, w)


if __name__ == "__main__":
    # Example
    import matplotlib.pyplot as plt
    from deom.spectrum.aux import gen_jw, sum_expn_etal_freq

    w = np.linspace(-100, 100, 50000)
    print(w)

    jw = gen_jw(w)
    etal, expn = aaa_fit_freq(w, jw, tol=1e-4, max_item=5)
    res_J_aaa = np.zeros(len(w), dtype=complex)
    sum_expn_etal_freq(w, res_J_aaa, expn, 2 * etal)

    print(expn)
    # [1.44467482e+01+4.56217096e-01j 4.51650857e+00+4.79038465e-01j
    # 1.76624911e+00+3.86385660e-01j 8.81170056e-01+2.65973993e-01j
    # 4.52596163e-01+6.98614222e-02j 2.04468564e-01+1.46543178e-02j
    # 9.37813226e-02+3.31557785e-03j 4.59095744e-02+3.75660516e-04j
    # 2.38508269e-02-2.20192005e-05j 7.85440391e-03-3.68312904e-07j]

    print(etal)
    # [ 1.08846005e-03+0.01523157j  7.62895977e-03+0.03652687j
    # 5.26239036e-02+0.08494969j  1.56074880e-01-0.02544657j
    # 2.77326006e-02-0.0673287j   3.77780591e-03-0.02700741j
    # 8.62320642e-04-0.01138094j  1.92857794e-04-0.00487667j
    # -2.96084263e-06-0.00270134j -4.20347728e-07-0.0025008j ]

    # plot the result
    plt.plot(w, jw - res_J_aaa)
    plt.show()
