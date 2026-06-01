import numpy as np
import sympy as sp
import itertools
import json
import sys
import time
import subprocess
from gen_spectrum import decompose_spe, init_qmd, sort_symmetry
from src.init import load_inputs
from src.deom import FilterCfg
from src.equilibrium import equilibrium_rk4, equilibrium_rk4_current
from src.print import print_moscal_banner

print("Generating model_data.npz...")

for (npsd1, npsd2), ferr, (lamd, omgs, zeta) in itertools.product(
    [(1, 3)],  # npsd
    [1e-10],  # ferr
    [(0.25, 1, 0.5)],  # lamd, omgs, zeta
):
    nsys = 1
    nmod_Q = 2
    nmod_F = 2

    beta1 = 1
    beta2 = 50

    w_sp, lamd_sp, zeta_sp, omgs_sp, beta_sp = sp.symbols(
        r"\omega , \lambda, \zeta, \Omega_{s}, \beta", real=True)
    spe_vib_sp = lamd_sp * omgs_sp / (
        omgs_sp * omgs_sp - w_sp * w_sp - zeta_sp * sp.I * w_sp)
    sp_para_dict = {lamd_sp: lamd, omgs_sp: omgs, zeta_sp: zeta}
    condition_dict = {}
    para_dict = {'beta': beta1}
    etal1, etar1, etaa1, expn1 = decompose_spe(spe_vib_sp, w_sp, sp_para_dict, para_dict,
                                           condition_dict, npsd1)
    
    etal2 = np.loadtxt("./prony_{}/etal1".format(beta2), dtype=complex) * 1
    expn2 = np.loadtxt("./prony_{}/expn1".format(beta2), dtype=complex)
    etal2, etar2, etaa2, expn2 = sort_symmetry(etal2, expn2)

    etal = np.append(etal1, etal2)
    etar = np.append(etar1, etar2)
    etaa = np.append(etaa1, etaa2)
    expn = np.append(expn1, expn2)

    np.savetxt('etaa1.dat', etaa1)
    np.savetxt('etal1.dat', etal1)
    np.savetxt('etar1.dat', etar1)
    np.savetxt('expn1.dat', expn1)
    np.savetxt('etaa2.dat', etaa2)
    np.savetxt('etal2.dat', etal2)
    np.savetxt('etar2.dat', etar2)
    np.savetxt('expn2.dat', expn2)

    rho0 = np.zeros((nsys, nsys), dtype=complex)
    rho0[0, 0] = 1
    # rho0[1, 1] = 0

    hams = np.zeros((nsys, nsys), dtype=complex)
    # hams[0, 0] = 1
    # hams[0, 1] = 1
    # hams[1, 0] = 1

    Q_list = np.zeros((nmod_Q, nsys, nsys), dtype=complex)
    Q_list[0, 0, 0] = 0.1
    # Q_list[0, 1, 1] = 1
    Q_list[1, 0, 0] = -0.001
    # Q_list[1, 1, 1] = -0.1
    # Q_list[1, 0, 0] = 0.5

    print("Q_list =", Q_list)

    p_list = np.zeros((nmod_F, len(expn)), dtype=complex)
    # p_list[0] = np.ones(len(expn), dtype=complex)
    p_list[0, :len(etal1)] = 1.0
    p_list[1, len(etal1):] = 1.0

    print("p_list =", p_list)

    ti = 0
    tf = 100
    dt = 0.02

    np.savez_compressed(
        "./model_data.npz",
        nsys=nsys,
        nind=len(expn),
        lmax=10,
        ti=ti,
        tf=tf,
        dt=dt,
        hams=hams,
        rho0=rho0,
        expn=expn,
        etal=etal,
        etar=etar,
        etaa=etaa,
        Q_list=Q_list,
        p_list=p_list,
    )

cfg, model, eom, time_cfg = load_inputs("model.yaml")

filt = FilterCfg(
    eps_on=1e-10,
    norm="fro",
    never_drop_root=True,
    drop_only_if_order_ge=1,
)

print_moscal_banner()

# equilibrium_rk4(
#     model, eom, time_cfg, filt, rho0_path="rho0.txt", curr_path="curr.txt"
# )

equilibrium_rk4_current(
    model, eom, time_cfg, filt, rho0_path="rho0.txt", curr_path="traces.txt"
)
