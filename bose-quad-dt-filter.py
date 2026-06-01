import time
import numpy as np
import subprocess
import itertools
import json
import sys
import sympy as sp
from deom import benchmark, convert, complex_2_json, init_qmd, init_qmd_quad, decompose_spe


for npsd, lmax, ferr, (lamd, omgs, zeta), dt in itertools.product(
    [1],  # npsd
    [10],  # lmax
    [1e-10],  # ferr
    [(0.3, 1, 1.5)],  # lamd, omgs, zeta
    [0.01]  # dt
):
    nmod = 2
    
    print((zeta / omgs / 2)**2)
    
    temp1 = 1
    beta1 = 1 / temp1
    temp2 = 1
    beta2 = 1 / temp2
    
    OMG = 50

    w_sp, lamd_sp, zeta_sp, omgs_sp, beta_sp = sp.symbols(
        r"\omega , \lambda, \zeta, \Omega_{s}, \beta", real=True)
    spe_vib_sp = omgs_sp / (
        omgs_sp * omgs_sp - w_sp * w_sp - zeta_sp * sp.I * w_sp)
    sp_para_dict = {lamd_sp: lamd, omgs_sp: omgs, zeta_sp: zeta}
    condition_dict = {}
    para_dict = {'beta': beta1}
    etal1, etar1, etaa1, expn1 = decompose_spe(spe_vib_sp, w_sp, sp_para_dict, para_dict,
                                           condition_dict, npsd)
    
    para_dict = {'beta': beta2}
    etal2, etar2, etaa2, expn2 = decompose_spe(spe_vib_sp, w_sp, sp_para_dict, para_dict,
                                           condition_dict, npsd)

    etal = np.append(etal1, etal2)
    etar = np.append(etar1, etar2)
    etaa = np.append(etaa1, etaa2)
    expn = np.append(expn1, expn2)
    
    np.savetxt('expn.dat', expn)
    np.savetxt('etal.dat', etal)
    np.savetxt('etaa.dat', etaa)
    
    np.savetxt('expn1.dat', expn1)
    np.savetxt('expn2.dat', expn2)
    np.savetxt('etal1.dat', etal1)
    np.savetxt('etar1.dat', etar1)
    np.savetxt('etaa1.dat', etaa1)

    nmodmax = np.zeros(len(expn), dtype=float)
    mode = np.zeros(len(expn), dtype=int)
    for i in range(nmod):
        mode[i * len(etal1):(i + 1) * len(etal1)] = i
    
    beta = para_dict['beta']

    rho0 = np.zeros((2, 2), dtype=complex)
    rho0[0, 0] = 1
    rho0[1, 1] = 0

    hams = np.zeros((2, 2), dtype=complex)
    # hams[0, 0] = 1
    # hams[0, 1] = 1
    # hams[1, 0] = 1

    qmds = np.zeros((nmod, 2, 2), dtype=complex)
    # qmds[0, 0, 0] = 1
    # qmds[1, 0, 0] = 1
    # qmds[0, 1, 1] = 1
    
    renormalize = np.zeros((nmod, 2, 2), dtype=complex)
    # renormalize[0, :, :] = etal1.sum() * qmds[0, :, :]
    # renormalize[1, :, :] = etal2.sum() * qmds[1, :, :]
    
    print(np.sum(etal1), np.sum(etal2))

    qmd2 = np.zeros((nmod, nmod, 2, 2), dtype=complex)
    qmd2[0, 1, 0, 0] = 0.5
    qmd2[0, 1, 1, 1] = 0.5
    qmd2[1, 0, 0, 0] = 0.5
    qmd2[1, 0, 1, 1] = 0.5
    # qmd2[1, 0, 0, 0] = 1
    # for i in range(nmod):
        # for j in range(nmod):
            # qmd2[i, j, :, :] = qmds[i, :, :]

    nmax = 200000

    json_init = {
        "nmax": nmax,
        "lmax": lmax,
        "alp0": 0,
        "alp1": 0,
        "alp2": 1,
        "ferr": ferr,
        "nind": len(expn),
        "nmod": nmod,
        "inistate": 0,
        "filter": True,
        "equilibrium": {
            "sc2": False,
            "dt-method": True,
            "ti": 0,
            "tf": 20,
            "dt": dt,
            "backup": True,
        },
        "expn": complex_2_json(expn),
        "ham1": complex_2_json(hams),
        "coef_abs": complex_2_json(etaa),
        "renormalize": complex_2_json(renormalize),
    }

    init_qmd(json_init, qmds, qmds, mode, 2, etaa, etal, etar)
    init_qmd_quad(json_init, qmd2, qmd2, qmd2, mode,
                  2, len(expn), nmod, etaa, etal, etar)

    magic_str = '{}-{}-dt-quad'.format(ferr, lmax)
    with open('input.json', 'w') as f:
        json.dump(json_init, f, indent=4, default=convert)
    cmd = r'export OMP_NUM_THREADS={}'.format(
        4 if len(sys.argv) == 1 else sys.argv[1])
    cmd += '&&' + r'$JEMALLOCPATH  ../code/bose_quad_2.out'
    start_time = time.time()
    with open('out-{}'.format(magic_str), "w") as outfile:
        result = subprocess.call(cmd, shell=True, stdout=outfile)
    np.savetxt('time-{}'.format(magic_str), [time.time() - start_time])
    # benchmark('prop-rho-eq.dat', magic_str, 'bose_quad')
    
    # file_str = 'prop-rho-eq.dat'
    # cmd = r"mv {} {}-{}".format(file_str, file_str, magic_str)
    # result = subprocess.call(cmd, shell=True)
