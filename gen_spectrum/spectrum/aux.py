import numpy as np

from scipy import linalg as LA


def least_squares(
    gamma: np.ndarray,
    aim_f_t: np.ndarray,
):
    """
    Solve the least squares problem.
    Input:
        gamma: 1D array of poles, \gamma_k.
        aim_f_t: 1D array of the target function.
    Output:
        omega_new: 1D array of the residues, \omega_k.
    """
    n_col = len(aim_f_t)
    n_row = len(gamma)
    gamma_m = np.zeros((2 * n_col, 2 * n_row), dtype=float)
    for i in range(n_row):
        for j in range(n_col):
            gamma_m[j, i] = np.real(gamma[i] ** j)
            gamma_m[n_col + j, n_row + i] = np.real(gamma[i] ** j)
            gamma_m[j, n_row + i] = -np.imag(gamma[i] ** j)
            gamma_m[n_col + j, i] = np.imag(gamma[i] ** j)
    h_m = np.append(np.real(aim_f_t), np.imag(aim_f_t))
    omega_new_temp = (LA.inv(gamma_m.T @ gamma_m) @ gamma_m.T @ h_m).reshape(2, n_row)
    return omega_new_temp[0, :] + 1.0j * omega_new_temp[1, :]


def gen_jw(w):
    """
    define the spectral density
    """
    return 1 / (w**2 + 1) / (1 + np.exp(-w * 400))


def sum_expn_etal_freq(w, res, expn, etal):
    """
    sum the poles and residues to get the spectral density
    """
    for i in range(len(etal)):
        res += etal[i] / (expn[i] - 1.0j * w)
    return res


def sum_expn_etal_time(t, res, expn, etal):
    """
    sum the poles and residues to get the correlation function
    """
    for i in range(len(etal)):
        res += etal[i] * np.exp(-expn[i] * t)
    return res


def fourier(jw, boson_fermi, beta, scale_fft, t_range, t_num, n_fft):
    """
    Fourier transform of the spectral density
    """
    n_rate = int(scale_fft * t_range / (4 * t_num))
    print("Should be any int: ", scale_fft * t_range / (4 * t_num))

    w = np.linspace(0, scale_fft * np.pi, n_fft + 1)[:-1]
    dw = w[1] - w[0]
    jw1 = jw(w)
    jw2 = jw(-w)

    if boson_fermi == "boson":
        cw1 = jw1 / (1 - np.exp(-beta * w))
        cw2 = jw2 * np.exp(-beta * w) / (1 - np.exp(-beta * w))
    elif boson_fermi == "fermi":
        cw1 = jw1 / (1 + np.exp(-beta * w))
        cw2 = jw2 * np.exp(-beta * w) / (1 + np.exp(-beta * w))

    cw1[0] = cw1[1] / 2
    cw2[0] = cw2[1] / 2

    fft_ct = (np.fft.fft(cw1) * dw - np.fft.ifft(cw2) * len(cw2) * dw) / np.pi
    fft_t = 2 * np.pi * np.fft.fftfreq(len(cw1), dw)

    return fft_ct[(t_range >= fft_t) & (fft_t >= 0)][::n_rate]
