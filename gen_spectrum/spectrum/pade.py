import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from gen_spectrum.spectrum.aux import sum_expn_etal_freq


def boson_function(
    x,
    pole,
    resi,
):
    fx = 1 / x + 0.5
    for i in range(len(pole)):
        fx += 2.0 * resi[i] * x / (x**2 + pole[i] ** 2)
    return fx


def fermi_function(
    x,
    pole,
    resi,
):
    fx = 0.5
    for i in range(len(pole)):
        fx += 2.0 * resi[i] * x / (x**2 + pole[i] ** 2)
    return fx


def tseig(
    D,
    E,
):
    mat = np.diag(E, -1) + np.diag(D, 0) + np.diag(E, 1)
    return -np.sort(-np.linalg.eigvalsh(mat))


def matsubara(
    N,
    boson_fermi=1,
):
    """
    Matsubara spectral decomposition
    """
    index_list = np.arange(N)
    if boson_fermi == 1:
        pole = 2 * (index_list + 1) * np.pi
        resi = np.ones(N)
        return pole, resi
    elif boson_fermi == 2:
        pole = (2 * index_list + 1) * np.pi
        resi = np.ones(N)
        return pole, resi


def pade(
    N,
    boson_fermi=1,
):
    """
    Padé spectral decomposition
    """
    pole, resi = [], []
    if N > 0:
        M = 2 * N + 1 // 2
        index_list = np.arange(M - 1)
        if boson_fermi == "boson":
            temp = 3.0
        elif boson_fermi == "fermi":
            temp = 1.0
        else:
            raise ValueError("boson_fermi has wrong value!")
        diag = np.zeros(M, dtype=float)
        doff = 1.0 / np.sqrt(
            (temp + 2.0 * index_list) * (temp + 2.0 * (index_list + 1))
        )
        pole = 2.0 / tseig(diag, doff)[:N]
        pol2 = pole**2

        M -= 1
        index_list = np.arange(M - 1)
        temp += 2.0
        diag = np.zeros(M, dtype=float)
        doff = 1.0 / np.sqrt(
            (temp + 2.0 * index_list) * (temp + 2.0 * (index_list + 1))
        )

        M //= 2
        eig2 = np.power(2.0 / tseig(diag, doff)[:M], 2)
        if boson_fermi == "boson":
            scaling = N * (2.0 * N + 3.0)
        elif boson_fermi == "fermi":
            scaling = N * (2.0 * N + 1.0)

        resi = np.zeros(N, dtype=float)
        for j in range(N):
            if j == N - 1:
                temp = 0.5 * scaling
            else:
                temp = 0.5 * scaling * (eig2[j] - pol2[j]) / (pol2[N - 1] - pol2[j])
            for k in range(M):
                temp *= (eig2[k] - pol2[j]) / (pol2[k] - pol2[j]) if k != j else 1.0
            resi[j] = temp
        return pole, resi


def spe_pade(
    spe,
    w_sp,
    sp_para_dict,
    beta,
    condition_dict,
    npsd,
    boson_fermi="boson",
):
    if sp.cancel(spe.subs(condition_dict)).as_real_imag()[1] == 0:
        imag_part = sp.cancel(spe.subs(condition_dict)).as_real_imag()[0]
    else:
        imag_part = sp.cancel(spe.subs(condition_dict)).as_real_imag()[1]
    numer, denom = sp.cancel(sp.factor(imag_part)).as_numer_denom()
    numer_get_para = (sp.factor(numer)).subs(sp_para_dict)
    denom_get_para = (sp.factor(denom)).subs(sp_para_dict)

    poles = sp.nroots(denom_get_para)
    poles_allplane = np.array(poles, dtype=complex)
    expn = poles_allplane[poles_allplane.imag < 0] * 1.0j
    etal = np.zeros_like(expn, dtype=complex)
    pole, resi = pade(npsd, boson_fermi)
    temp = 1 / beta

    for ii, i_expn in enumerate(expn):
        poles_exclude = poles_allplane[np.abs(poles_allplane + 1.0j * i_expn) > 1e-14]
        etal[ii] = (
            -2.0j
            * complex(sp.N(numer_get_para.subs({w_sp: -1.0j * i_expn})))
            / np.multiply.reduce(-1.0j * i_expn - poles_exclude)
        )
        if boson_fermi == "boson":
            etal[ii] = etal[ii] * boson_function(-1.0j * i_expn / temp, pole, resi)
        elif boson_fermi == "fermi":
            etal[ii] = etal[ii] * fermi_function(-1.0j * i_expn / temp, pole, resi)

    f = numer_get_para / np.multiply.reduce(w_sp - poles_allplane)
    f = sp.lambdify(w_sp, f)

    zomg = -1.0j * pole * temp
    jsum = f(zomg)
    expn = np.append(expn, pole * temp)
    etal = np.append(etal, -2.0j * resi * temp * jsum)

    return etal, expn


def spe_pade_ana(
    spe,
    w_sp,
    sp_para_dict,
    beta,
    condition_dict,
    npsd,
    boson_fermi="boson",
):
    if sp.cancel(spe.subs(condition_dict)).as_real_imag()[1] == 0:
        imag_part = sp.cancel(spe.subs(condition_dict)).as_real_imag()[0]
    else:
        imag_part = sp.cancel(spe.subs(condition_dict)).as_real_imag()[1]
    numer, denom = sp.cancel(sp.factor(imag_part)).as_numer_denom()
    numer_get_para = (sp.factor(numer)).subs(sp_para_dict)
    denom_get_para = (sp.factor(denom)).subs(sp_para_dict)

    poles = sp.nroots(denom_get_para)
    poles_allplane = np.array(poles, dtype=complex)
    expn = poles_allplane[poles_allplane.imag < 0] * 1.0j
    etal = np.zeros_like(expn, dtype=complex)
    pole, resi = pade(npsd, boson_fermi)
    temp = 1 / para_dict["beta"]

    for ii, i_expn in enumerate(expn):
        poles_exclude = poles_allplane[np.abs(poles_allplane + 1.0j * i_expn) > 1e-14]
        etal[ii] = (
            -2.0j
            * complex(sp.N(numer_get_para.subs({w_sp: -1.0j * i_expn})))
            / np.multiply.reduce(-1.0j * i_expn - poles_exclude)
        )
        if boson_fermi == "boson":
            etal[ii] = etal[ii] * boson_function(-1.0j * i_expn / temp, pole, resi)
        elif boson_fermi == "fermi":
            etal[ii] = etal[ii] * fermi_function(-1.0j * i_expn / temp, pole, resi)
    return expn, etal


if __name__ == "__main__":
    beta = 10
    gam, eta = 1, 1
    w_sp, eta_sp, gamma_sp, beta_sp = sp.symbols(
        r"\omega, \eta, \gamma, \beta", real=True
    )
    sp_para_dict = {eta_sp: eta, gamma_sp: gam}
    condition_dict = {}
    para_dict = {"beta": beta}

    # Boson case
    phixx_sp = gamma_sp / (
        gamma_sp * gamma_sp
        - w_sp * w_sp
        - sp.I * w_sp * (eta_sp * gamma_sp / (gamma_sp - sp.I * w_sp))
    )
    etal, expn = spe_pade(
        phixx_sp,
        w_sp,
        sp_para_dict,
        para_dict,
        condition_dict,
        5,
        boson_fermi="boson",
    )

    w = np.linspace(-2.5, 5, 50000)
    jw = sp.lambdify(w_sp, phixx_sp.subs(sp_para_dict))(w) / (1 - np.exp(-beta * w))
    res_J_p = np.zeros(len(w), dtype=complex)
    sum_expn_etal_freq(w, res_J_p, expn, etal)
    plt.plot(w, jw.imag / np.max(jw.imag), "b")
    plt.plot(w, res_J_p.real / np.max(res_J_p.real), "r--")

    # Fermi case
    phixx_sp = eta_sp * gamma_sp**2 / (w_sp**2 + gamma_sp**2)
    etal, expn = spe_pade(
        phixx_sp,
        w_sp,
        sp_para_dict,
        para_dict,
        condition_dict,
        5,
        boson_fermi="fermi",
    )

    jw = sp.lambdify(w_sp, phixx_sp.subs(sp_para_dict))(w) / (1 + np.exp(-beta * w))
    res_J_p = np.zeros(len(w), dtype=complex)
    sum_expn_etal_freq(w, res_J_p, expn, etal)
    plt.plot(w, jw / np.max(jw), "b")
    plt.plot(w, res_J_p.real / np.max(res_J_p.real), "r--")
    plt.show()
    plt.clf()
