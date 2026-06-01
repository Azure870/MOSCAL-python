import os
import subprocess
from functools import reduce

import numpy as np
import sympy as sp
from scipy.linalg import expm as matrix_exp
import pandas as pd

from gen_spectrum.spectrum.pade import spe_pade
from gen_spectrum.spectrum.prony import prony
from gen_spectrum.spectrum.aaa import aaa_fit_freq
from gen_spectrum.spectrum.esprit import esprit
from gen_spectrum.spectrum.aux import sum_expn_etal_time, fourier


def fastread(str):
    try:
        data = pd.read_csv(str, header=None, sep="\s+", dtype=np.float64)
    except:
        data = pd.read_csv(str, header=None, sep="\s+", skipfooter=1, dtype=np.float64)
    return data


def direct_product_2d(a, b):
    c = np.zeros((len(a) * len(b), len(a) * len(b)))
    for i in range(len(a)):
        for j in range(len(a)):
            for k in range(len(b)):
                for l in range(len(b)):
                    c[i * len(b) + k, j * len(b) + l] = a[i, j] * b[k, l]
    return c


def gen_twsg(expn):
    twsg = []
    syl_nind = 0
    expn_bk = expn.copy()
    syl_nmod = 0
    for i in expn:
        twsg_a = np.where(np.abs(expn_bk - i) < 1e-10)
        if len(twsg_a[0]) > 0:
            twsg.append(twsg_a[0])
            syl_nind += 1
        syl_nmod = max(len(twsg_a[0]), syl_nmod)
        expn_bk[twsg_a] = -100
    for i_twsg in twsg:
        if len(i_twsg) < syl_nmod:
            i_twsg.append([-1] * (syl_nmod - len(i_twsg)))
    twsg = np.array(twsg)
    return twsg, syl_nind, syl_nmod


def append(*args):
    list_ = [i for i in args]
    return reduce(np.append, list_)


def benchmark(file_str, magic_str, dir_str="bose", if_np=True, if_save=False):
    cmd = r"mv {} {}-{}".format(file_str, file_str, magic_str)
    result = subprocess.call(cmd, shell=True)
    if os.path.exists("result/{}/{}-{}.npy".format(dir_str, file_str, magic_str)):
        data1 = np.load("result/{}/{}-{}.npy".format(dir_str, file_str, magic_str))
    else:
        data1 = fastread_np("result/{}/{}-{}".format(dir_str, file_str, magic_str))
    data2 = fastread_np("./{}-{}".format(file_str, magic_str))
    if if_save:
        np.save("result/{}/{}-{}".format(dir_str, file_str, magic_str), data2)
    result = np.sum(np.abs(data1 - data2))
    if float(result) > 1e-6:
        print(result, "FAILED")
        print(result, "!!!!!!!!")
    else:
        print(result, "PASSED")


def thermal_equilibrium(beta, hams):
    return matrix_exp(-beta * hams) / np.trace(matrix_exp(-beta * hams))


def direct_product(a, *args):
    c = a.copy()
    for arg in args:
        if not isinstance(arg, np.ndarray):
            raise TypeError("Input must be numpy.ndarray")
        c = direct_product_2d(c, arg)
    return c


def single_oscillator(omega, beta):
    etal = np.array(
        [1 / (2 * (1 - np.exp(-beta * omega))), -1 / (2 * (1 - np.exp(beta * omega)))],
        dtype=complex,
    )
    etar = np.array(
        [-1 / (2 * (1 - np.exp(beta * omega))), 1 / (2 * (1 - np.exp(-beta * omega)))],
        dtype=complex,
    )
    etaa = np.sqrt(np.abs(etal + etar))
    expn = np.array([1.0j * omega, -1.0j * omega])
    return etal, etar, etaa, expn


def fastread_np(str):
    return fastread(str).to_numpy()


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    elif isinstance(o, np.int32):
        return int(o)
    raise TypeError


def sort_symmetry(etal, expn, if_sqrt=False):
    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]
    expn = expn[expn_imag_sort]
    etal = etal[expn_imag_sort]
    etar = etal[expn_imag_sort]
    expn_val_cc = np.where(expn[expn_imag > 1e-10])[0]
    etaa = np.zeros(len(etal), dtype=float)
    for ii in range(0, len(expn_val_cc), 2):
        even_i = ii
        odd_i = ii + 1
        etar[even_i] = np.conj(etal[odd_i])
        etar[odd_i] = np.conj(etal[even_i])
        etaa[even_i] = np.abs(etal[even_i])
        etaa[odd_i] = np.abs(etal[odd_i])
    for ii in range(len(expn_val_cc), len(expn)):
        even_i = ii
        etar[even_i] = np.conj(etal[even_i])
        etaa[even_i] = np.abs(etal[even_i])
    if if_sqrt:
        etaa = np.sqrt(etaa)
    # print("changelog: change expn, etal, etar, etaa to etal, etar, etaa, expn")
    return etal, etar, etaa, expn


def decompose_spe(
    spe,
    w_sp,
    sp_para_dict,
    para_dict,
    condition_dict,
    npsd,
    boson_fermi="boson",
    method="pade",
    fit_method="pade",
    w_range=(-100, 100),
    w_num=50000,
    t_range=5000,
    t_num=50000,
    esprit_L=250,
    pade_npsd=200,
    n_fft=1000000,
    scale_fft=2000,
):
    r"""
    Decompose spectrum to poles and residues.
    Input:
        spe: spectrum.
        w_sp: symbol of frequency.
        sp_para_dict: parameters of the spectrum.
        para_dict: parameters of the system.
        condition_dict: conditions of the spectrum.
        npsd: number of poles.
        boson_fermi: boson or fermi.
        method: pade, aaa, prony or esprit.
        fit_method: pade, aaa or fourier.
        w_range: range of frequency. No effect in fourier or pade.
        w_num: number of frequency points. No effect in fourier or pade.
        t_range: range of time.
        t_num: number of time points.
        esprit_L: number of poles for esprit.
        pade_npsd: number of poles for pade.
        n_fft: number of fft points.
        scale_fft: scale of fft.
    Note:
        Prony method needs tune the parameters to get the best result.
    """
    beta = para_dict["beta"]
    if sp.cancel(spe).as_real_imag()[1] == 0:
        imag_spe = sp.cancel(spe).as_real_imag()[0]
    else:
        imag_spe = sp.cancel(spe).as_real_imag()[1]
    w = np.linspace(w_range[0], w_range[1], w_num)

    if method == "pade":
        if isinstance(npsd, list):
            print("npsd is a list, using the first element")
            npsd = npsd[0]
        etal, expn = spe_pade(
            imag_spe,
            w_sp,
            sp_para_dict,
            beta,
            condition_dict,
            npsd,
            boson_fermi,
        )
    elif method == "aaa":
        if isinstance(npsd, list):
            print("npsd is a list, using the first element")
            npsd = npsd[0]
        jw = sp.lambdify(w_sp, imag_spe.subs(sp_para_dict))(w) / (
            1 - np.exp(-para_dict["beta"] * w)
        )
        etal, expn = aaa_fit_freq(w, jw, tol=1e-4, max_item=npsd)
    elif method in ["prony", "esprit"]:
        if fit_method == "pade":
            etal_long, expn_long = spe_pade(
                spe,
                w_sp,
                sp_para_dict,
                beta,
                condition_dict,
                pade_npsd,
                boson_fermi,
            )
        elif fit_method == "aaa":
            if boson_fermi == "boson":
                jw = sp.lambdify(w_sp, imag_spe.subs(sp_para_dict))(w) / (
                    1 - np.exp(-para_dict["beta"] * w)
                )
            elif boson_fermi == "fermi":
                jw = sp.lambdify(w_sp, imag_spe.subs(sp_para_dict))(w) / (
                    1 + np.exp(-para_dict["beta"] * w)
                )
            etal_long, expn_long = aaa_fit_freq(w, jw, tol=1e-12, max_item=100)
        elif fit_method == "fourier":
            jw = sp.lambdify(w_sp, imag_spe.subs(sp_para_dict))(w) / (
                1 - np.exp(-para_dict["beta"] * w)
            )
        else:
            raise ValueError("fit_method must be pade, fourier or aaa")

        if method == "prony":
            if fit_method in ["pade", "aaa"]:
                t = np.linspace(0, 1, 2 * t_num + 1)
                res_t = np.zeros(len(t), dtype=complex)
                sum_expn_etal_time(t_range * t, res_t, expn_long, etal_long)
            elif fit_method == "fourier":
                res_t = fourier(jw, boson_fermi, beta, scale_fft, t_range, t_num, n_fft)
            print("check the sample points")
            print(res_t[:10])
            print(res_t[-10:])
            etal, expn = prony(res_t, npsd, t_num, t_range)
        elif method == "esprit":
            if fit_method in ["pade", "aaa"]:
                t = np.linspace(0, 1, 2 * t_num)
                res_t = np.zeros(len(t), dtype=complex)

                sum_expn_etal_time(t_range * t, res_t, expn_long, etal_long)
            print(res_t[:10])
            print(res_t[-10:])
            etal, expn = esprit(res_t, npsd, n=t_num, scale=t_range, L=esprit_L)
    return sort_symmetry(etal, expn)


def complex_2_json(list_input, if_dense=None):
    """
    Convert a complex matrix to a Json format.

    Parameters
    ----------
    if_dense: False, True or None.
        If the input is a dense matrix. If None, then the function will try to determine whether the input is a dense matrix or not.

    Returns
    -------
    json_init: dict
    """
    if if_dense is None:
        if len(np.shape(list_input)) > 1:
            index_list = np.where(abs(list_input.flatten()) > 1e-10)
            if 5 * len(index_list) > len(list_input.flatten()):
                if_dense = True
            else:
                if_dense = False
        else:
            if_dense = True

    if isinstance(list_input, np.ndarray):
        if if_dense:
            return {
                "if_initial": True,
                "real": list(np.real(list_input.flatten())),
                "imag": list(np.imag(list_input.flatten())),
            }
        else:
            index_list = np.where(abs(list_input) > 1e-10)
            json_init = {
                "if_initial": True,
                "if_dense": False,
                "length": len(index_list[-1]),
                "i": list(index_list[-2]),
                "j": list(index_list[-1]),
                "real": list(np.real(list_input[index_list])),
                "imag": list(np.imag(list_input[index_list])),
            }
            if len(index_list) > 2:
                json_init["k"] = list(index_list[-3])
            return json_init
    else:
        return {"real": np.real(list_input), "imag": np.imag(list_input)}


def init_qmd(json_init, qmd1a, qmd1c, mode, nsys, etaa, etal, etar, if_dense=None):
    qmdta_l = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdta_r = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdtc_l = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdtc_r = np.zeros((len(mode), nsys, nsys), dtype=complex)
    for i, i_mod in enumerate(mode):
        qmdta_l[i, :, :] = qmd1a[i_mod, :, :] * np.sqrt(etaa[i])
        qmdta_r[i, :, :] = qmd1a[i_mod, :, :] * np.sqrt(etaa[i])
        qmdtc_l[i, :, :] = qmd1c[i_mod, :, :] * etal[i] / np.sqrt(etaa[i])
        qmdtc_r[i, :, :] = qmd1c[i_mod, :, :] * etar[i] / np.sqrt(etaa[i])
    json_init["qmdta_l"] = complex_2_json(qmdta_l, if_dense=if_dense)
    json_init["qmdta_r"] = complex_2_json(qmdta_r, if_dense=if_dense)
    json_init["qmdtc_l"] = complex_2_json(qmdtc_l, if_dense=if_dense)
    json_init["qmdtc_r"] = complex_2_json(qmdtc_r, if_dense=if_dense)


# Do some normalize thing, you can find more details in the pdf file.
def init_qmd_quad(
    json_init,
    qmd2a,
    qmd2b,
    qmd2c,
    mode,
    nsys,
    nind,
    nmod,
    etaa,
    etal,
    etar,
    if_dense=None,
):
    qmdt2a_l = np.zeros((nind * nind, nsys, nsys), dtype=complex)
    qmdt2a_r = np.zeros((nind * nind, nsys, nsys), dtype=complex)
    qmdt2b_l = np.zeros((nind * nind, nsys, nsys), dtype=complex)
    qmdt2b_r = np.zeros((nind * nind, nsys, nsys), dtype=complex)
    qmdt2c_l = np.zeros((nind * nind, nsys, nsys), dtype=complex)
    qmdt2c_r = np.zeros((nind * nind, nsys, nsys), dtype=complex)
    for i, i_mod in enumerate(mode):
        for j, j_mod in enumerate(mode):
            index_mat = i * nind + j
            qmdt2a_l[index_mat, :, :] = (
                qmd2a[i_mod, j_mod, :, :] * np.sqrt(etaa[i]) * np.sqrt(etaa[j])
            )
            qmdt2a_r[index_mat, :, :] = (
                qmd2a[i_mod, j_mod, :, :] * np.sqrt(etaa[i]) * np.sqrt(etaa[j])
            )
            qmdt2b_l[index_mat, :, :] = (
                qmd2b[i_mod, j_mod, :, :]
                * etal[i]
                / np.sqrt(etaa[i])
                * np.sqrt(etaa[j])
            )
            qmdt2b_r[index_mat, :, :] = (
                qmd2b[i_mod, j_mod, :, :]
                * etar[i]
                / np.sqrt(etaa[i])
                * np.sqrt(etaa[j])
            )
            qmdt2c_l[index_mat, :, :] = (
                qmd2c[i_mod, j_mod, :, :]
                * etal[i]
                * etal[j]
                / np.sqrt(etaa[i])
                / np.sqrt(etaa[j])
            )
            qmdt2c_r[index_mat, :, :] = (
                qmd2c[i_mod, j_mod, :, :]
                * etar[i]
                * etar[j]
                / np.sqrt(etaa[i])
                / np.sqrt(etaa[j])
            )
    json_init["qmdt2a_l"] = complex_2_json(qmdt2a_l, if_dense=if_dense)
    json_init["qmdt2a_r"] = complex_2_json(qmdt2a_r, if_dense=if_dense)
    json_init["qmdt2b_l"] = complex_2_json(qmdt2b_l, if_dense=if_dense)
    json_init["qmdt2b_r"] = complex_2_json(qmdt2b_r, if_dense=if_dense)
    json_init["qmdt2c_l"] = complex_2_json(qmdt2c_l, if_dense=if_dense)
    json_init["qmdt2c_r"] = complex_2_json(qmdt2c_r, if_dense=if_dense)
