r"""
    ESPRIT algorithm for exponential fitting.
    The result is a set of poles (\gamma_k) and residues (\eta_k) that approximate C(t) = \sum \eta_k * \exp(-\gamma_k * t) in the time domain. C(t) is the Fourier transform of the J(\omega).
    The exponential fitting can be transformed to the polynomial fitting problem, C(t_n) = \sum \omega_k * z_k^n. The poles (\gamma_k) are related to the nodes, z_k.
"""

import numpy as np
from scipy import linalg as LA

from gen_spectrum.spectrum.aux import sum_expn_etal_time, least_squares


def find_gamma_esprit(
    aim_ft: np.ndarray,
    n: int,
    L: int,
    m: int,
):
    r"""
    Find the node using the ESPRIT algorithm.
    Input:
        aim_ft: 1D array of the target function.
        n: int, the number of sampling points.
        L: int, the number of Hankel matrix rows.
        m: int, the number of nodes.
    Output:
        gamma: 1D array of node, \gamma_k.
    """
    mat_h = np.zeros((2 * n - L, L + 1), dtype=complex)
    for i in range(L + 1):
        for j in range(2 * n - L):
            mat_h[j, i] = aim_ft[j + i]
    _, A_svd, W = LA.svd(
        mat_h,
        full_matrices=False,
        lapack_driver="gesvd",
    )
    print(A_svd[:50] / A_svd[0])
    W_0 = W[:m, :L]
    W_1 = W[:m, 1 : L + 1]
    gamma, _ = LA.eig(LA.pinv(W_0.T) @ W_1.T)
    return gamma


def esprit(
    res_t,
    m,
    n=2500,
    scale=2500,
    L=250,
):
    r"""
    ESPRIT algorithm for exponential fitting.
    Input:
        expn_long: 1D array of poles, \gamma_k.
        etal_long: 1D array of residues, \eta_k.
        m: int or list of int, the number of poles.
        n: int, the number of sampling points.
        scale: float, the scaling factor.
        L: int, the number of Hankel matrix rows.
    Note:
        If m is a list of int, find the gamma using both the real and imaginary parts.
        This is useful when one needs the poles to be complex conjugate.
        If m is an int, find the gamma using the complex one.
    """

    if isinstance(m, int):
        gamma = find_gamma_esprit(res_t, n, L, m)
    elif isinstance(m, list):
        gamma_r = find_gamma_esprit(res_t.real, n, L, m[0])
        gamma_i = find_gamma_esprit(res_t.imag, n, L, m[1])
        gamma = np.append(gamma_r, gamma_i)

    etal_e = least_squares(gamma, res_t)
    expn_e = -(2 * n) * np.log(gamma) / scale
    for i, i_expn in enumerate(expn_e):
        if np.abs((i_expn.imag / np.pi) % 1) < 1e-8:
            expn_e[i] = expn_e[i].real
    return etal_e, expn_e


if __name__ == "__main__":
    # Example
    import matplotlib.pyplot as plt
    from aux import gen_jw, sum_expn_etal_freq

    w = np.linspace(-100, 100, 50000)
    jw = gen_jw(w)
    expn_long = np.array(
        [
            1.44467482e01 + 4.56217096e-01j,
            4.51650857e00 + 4.79038465e-01j,
            1.76624911e00 + 3.86385660e-01j,
            8.81170056e-01 + 2.65973993e-01j,
            4.52596163e-01 + 6.98614222e-02j,
            2.04468564e-01 + 1.46543178e-02j,
            9.37813226e-02 + 3.31557785e-03j,
            4.59095744e-02 + 3.75660516e-04j,
            2.38508269e-02 - 2.20192005e-05j,
            7.85440391e-03 - 3.68312904e-07j,
        ]
    )
    etal_long = np.array(
        [
            1.08846005e-03 + 0.01523157j,
            7.62895977e-03 + 0.03652687j,
            5.26239036e-02 + 0.08494969j,
            1.56074880e-01 - 0.02544657j,
            2.77326006e-02 - 0.0673287j,
            3.77780591e-03 - 0.02700741j,
            8.62320642e-04 - 0.01138094j,
            1.92857794e-04 - 0.00487667j,
            -2.96084263e-06 - 0.00270134j,
            -4.20347728e-07 - 0.0025008j,
        ]
    )
    etal_e, expn_e = esprit(expn_long, 2 * etal_long, 5, n=10000, scale=10000, L=500)
    res_J_p = np.zeros(len(w), dtype=complex)
    sum_expn_etal_freq(w, res_J_p, expn_e, etal_e)
    plt.plot(w, jw - res_J_p.real)
    plt.show()
