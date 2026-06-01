# src/rem.py
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Sequence
from .deom import DEOM, Ctrl, Model, EOMSpec
from .algebra import RHSWorkspace
from .kernels_numba import (
    add_drift_subset_nb, add_Q_left_nb, add_Q_right_nb, apply_Fa_once_gather_eta_nb,
    expand_one_hop_list_nb, apply_Phia_once_gather_nb
)


def precompute_n_dot_gamma(c: Ctrl, gamma: np.ndarray) -> np.ndarray:
    """
    weight[id] = sum_k n_k * gamma[k]
    gamma shape: (nind,)
    return shape: (Nado,) complex128
    """
    gamma = np.asarray(gamma)
    if gamma.shape != (c.nind,):
        raise ValueError("gamma shape mismatch")
    # id2n is int32, cast to complex and dot
    # (Nado, nind) @ (nind,) -> (Nado,)
    return (c.id2n.astype(np.complex128) @ gamma.astype(np.complex128))


def expand_one_hop(c: Ctrl, ids_in: np.ndarray, work: RHSWorkspace) -> np.ndarray:
    """
    Return ids_out (sorted increasing) = ids_in ∪ up(ids_in) ∪ dn(ids_in).
    Uses work.mark and work.buf_ids, no Python set.
    """
    # mark = work.mark
    # mark[:] = False

    # up = c.up
    # dn = c.dn

    # # 标记输入与邻居
    # for m in ids_in:
    #   mark[m] = True
    #   for k in range(c.nind):
    #     iu = up[m, k]
    #     if iu != -1:
    #       mark[iu] = True
    #     idn = dn[m, k]
    #     if idn != -1:
    #       mark[idn] = True

    # ids_out = np.flatnonzero(mark).astype(np.int32)
    # return ids_out
    if ids_in.dtype != np.int32:
        ids_in = ids_in.astype(np.int32)

    cnt = expand_one_hop_list_nb(
        ids_in,
        c.up,
        c.dn,
        work.mark,      # bool[Nado]
        work.buf_ids,   # int32[Nado]
    )
    # 注意：build_hop_sequence 需要保留每一层 ids（下一层会覆盖 buf_ids），所以这里要 copy
    return work.buf_ids[:cnt].copy()


def build_hop_sequence(c: Ctrl, ids0: np.ndarray, hop: int, work: RHSWorkspace) -> List[np.ndarray]:
    """
    Build hop sequence [S0, S1, ..., Shop], where S_{t+1} = expand_one_hop(S_t).
    Only depends on topology (up/dn) and hop count; independent of etal/etar.
    """
    seq: List[np.ndarray] = [ids0]
    ids = ids0
    for _ in range(hop):
        ids = expand_one_hop(c, ids, work)
        seq.append(ids)
    return seq


def zero_subset(A: np.ndarray, ids: np.ndarray) -> None:
    A[ids, :, :] = 0.0


def add_drift_subset(input: np.ndarray, out: np.ndarray, deom, model, ids_in: np.ndarray, n_dot_gamma: np.ndarray) -> None:
    hams = model.hams
    for i in ids_in:
        ddo = input[i]
        out[i] += -1j * (hams @ ddo - ddo @ hams)          # -i[H,R]
        out[i] += -(n_dot_gamma[i]) * ddo          # - (Σ n_k γ_k) R


def apply_Fa_once_gather_eta(
    inp: np.ndarray,
    out: np.ndarray,
    deom,
    model,
    a: int,
    ids_in: np.ndarray,   # 输入支持集（X 非零主要在这里）
    ids_out: np.ndarray,  # 输出支持集（结果只需要在这里写）
    eta_coeff: np.ndarray,  # etal 或 etar
    work,
) -> None:
    mark = work.mark
    mark[:] = False
    mark[ids_in] = True

    p = model.p_list[a]      # (nind,)
    c = deom.c
    up = c.up
    dn = c.dn
    id2n = c.id2n
    nind = c.nind
    etaa = model.etaa

    # 2) out 只需要写 ids_out，其余位置不管
    out[ids_out, :, :] = 0.0

    for n in ids_out:
        nn = id2n[n]  # multi-index n_k
        acc = out[n]  # view
        # print("apply F_", a, " on id ", n, " n=", nn  )

        for k in range(nind):
            pk = p[k]
            if pk == 0:
                continue

            # term from n+e_k
            iu = up[n, k]
            if iu != -1 and mark[iu]:
                # print(" up to id ", iu, " n=", id2n[iu])
                acc += pk * (np.sqrt(nn[k] + 1.0) * np.sqrt(etaa[k])) * inp[iu]

            # term from n-e_k
            idn = dn[n, k]
            if idn != -1 and mark[idn] and nn[k] > 0:
                # print(" dn to id ", idn, " n=", id2n[idn])
                acc += pk * \
                    (np.sqrt(nn[k]) * eta_coeff[k] /
                     np.sqrt(etaa[k])) * inp[idn]


def apply_F_seq_gather_with_hops(
    inp: np.ndarray,
    deom,
    model,
    F_seq: Sequence[Tuple[int, int]],   # [(a, power), (b, power2)]，最多两段
    hop_seq: Sequence[np.ndarray],       # S0,S1,...,Shop
    buf1: np.ndarray,
    buf2: np.ndarray,
    eta: np.ndarray,
    work,
) -> Tuple[np.ndarray, np.ndarray]:
    X = inp
    toggle = 0
    step = 0
    for (a, pwr) in F_seq:
        if pwr <= 0:
            raise ValueError("power must be >= 1")

        for _ in range(pwr):
            # print(" step ", step, " apply F_", a)
            ids_in = hop_seq[step]
            ids_out = hop_seq[step + 1]

            if toggle == 0:
                # apply_Fa_once_gather_eta(
                apply_Fa_once_gather_eta_numba(
                    X, buf1, deom, model, a,
                    ids_in=ids_in, ids_out=ids_out,
                    eta_coeff=eta, work=work
                )
                X = buf1
                toggle = 1
            else:
                # apply_Fa_once_gather_eta(
                apply_Fa_once_gather_eta_numba(
                    X, buf2, deom, model, a,
                    ids_in=ids_in, ids_out=ids_out,
                    eta_coeff=eta, work=work
                )
                X = buf2
                toggle = 0

            step += 1

    return X, hop_seq[step]

# -------------------------
# Q kernels (explicit coef)
# -------------------------


def add_Q_left(ids: np.ndarray, Q: np.ndarray, X: np.ndarray, out: np.ndarray) -> None:
    for i in ids:
        out[i] += -1j * (Q @ X[i])


def add_Q_right(ids: np.ndarray, Q: np.ndarray, X: np.ndarray, out: np.ndarray) -> None:
    for i in ids:
        out[i] += 1j * (X[i] @ Q)


def apply_Fa_once_gather_eta_numba(
    inp, out, deom, model, a, ids_in, ids_out, eta_coeff, work
):
    mark = work.mark
    mark[:] = False
    mark[ids_in] = True 
    
    apply_Fa_once_gather_eta_nb(
        inp, out,
        ids_out=ids_out,
        mark=mark,
        p=model.p_list[a],
        up=deom.c.up,
        dn=deom.c.dn,
        id2n=deom.c.id2n,
        eta_coeff=eta_coeff,
        etaa=model.etaa
    )

# -------------------------
# Main RHS generator
# -------------------------

def apply_Phia_once_gather_numba(
    inp, out, deom, model, a, ids_in, ids_out, eta_coeff, work
):
    """
    只用于输出：对输入 inp 施加一次 Phi_a（gather），把结果写到 out。
    """
    mark = work.mark
    mark[:] = False
    mark[ids_in] = True

    apply_Phia_once_gather_nb(
        inp, out,
        ids_out=ids_out,
        mark=mark,
        p=model.p_list[a],
        up=deom.c.up,
        dn=deom.c.dn,
        id2n=deom.c.id2n,
        gamma=model.gamma,
        eta_coeff=eta_coeff,
        etaa=model.etaa,
    )

def rem_gen(
    input: np.ndarray,
    out: np.ndarray,
    deom: DEOM,
    model: Model,
    eom: EOMSpec,
    n_dot_gamma: np.ndarray,
    work: RHSWorkspace,
) -> np.ndarray:
    """
    Compute RHS into `out` using:
      drift: -i[H,·] - (n·gamma)
      terms: -i  * (Q @ rho(;F_seq)_L)  +  i * (rho(;F_seq)_R @ Q)

    left recursion uses eta = model.etal
    right recursion uses eta = model.etar

    Returns ids_total: the maximum support set written this call (useful for stepper update).
    """
    ids0 = deom.active_ids
    # ids0 = np.arange(deom.c.Nado, dtype=np.int32)

    # print("Nactive =", deom.active_ids.size)

    max_hop = 0
    for (q, F_seq_list) in eom.terms:
        hop = sum(int(pwr) for (_, pwr) in F_seq_list)
        if hop > max_hop:
            max_hop = hop

    hop_seq_max = build_hop_sequence(deom.c, ids0, max_hop, work)
    ids_total = hop_seq_max[-1]
    zero_subset(out, ids_total)

    # if eom.include_drift:
    add_drift_subset_nb(input, out, ids0, model.hams, n_dot_gamma)

    for (q, F_seq_list) in eom.terms:
        # print("term q =", q, " F_seq =", F_seq_list)
        Qq = model.Q_list[q]
        F_seq = [(int(a), int(pwr)) for (a, pwr) in F_seq_list]
        if len(F_seq) == 0 or len(F_seq) > 2:
            raise ValueError("only supports 1 or 2 F factors in sequence")

        hop = sum(pwr for (_, pwr) in F_seq)
        hop_seq = hop_seq_max[:hop+1]
        # left: eta = etal
        # print(" left recursion")
        # Xl, ids_term = apply_F_seq_scatter_with_hops(
        Xl, ids_term = apply_F_seq_gather_with_hops(
            input, deom, model, F_seq, hop_seq,
            buf1=deom.ddos_temp1,
            buf2=deom.ddos_temp2,
            eta=model.etal,
            work=work
        )
        # add_Q_left(ids_term, Qq, Xl.copy(), out)
        add_Q_left_nb(ids_term, Qq, Xl, out)
        # print(" right recursion")
        # right: eta = etar (reuse same hop_seq and same ids_term shape)
        # Xr, ids_term2 = apply_F_seq_scatter_with_hops(
        Xr, ids_term2 = apply_F_seq_gather_with_hops(
            input, deom, model, F_seq, hop_seq,
            buf1=deom.ddos_temp1,
            buf2=deom.ddos_temp2,
            eta=model.etar,
            work=work
        )
        add_Q_right_nb(ids_term2, Qq, Xr, out)

    return ids_total
  
def apply_op_seq_with_hops_record_root_trace(
    inp, deom, model,
    op_seq: List[Tuple[str, int]],      # [("Phi", a), ("F", a), ...]
    hop_seq: Sequence[np.ndarray],
    buf1, buf2,
    eta_coeff_for_F: np.ndarray,        # model.etal or model.etar
    eta_coeff_for_Phi: np.ndarray,      # model.etal or model.etar (可与F不同)
    work,
    include_initial: bool = True,
):
    """
    输出用：按 op_seq 的全局顺序执行算符串，每做一步 (F 或 Phi) 就记录一次 root trace。
    """
    rid = deom.c.root_id
    nops = len(op_seq)

    X = inp
    toggle = 0
    step = 0

    traces = np.empty(nops + (1 if include_initial else 0), dtype=np.complex128)
    tpos = 0
    if include_initial:
        traces[tpos] = np.trace(X[rid])
        tpos += 1

    def _F_once(a, ids_in, ids_out):
        nonlocal X, toggle
        if toggle == 0:
            apply_Fa_once_gather_eta_numba(X, buf1, deom, model, a, ids_in, ids_out, eta_coeff_for_F, work)
            X = buf1; toggle = 1
        else:
            apply_Fa_once_gather_eta_numba(X, buf2, deom, model, a, ids_in, ids_out, eta_coeff_for_F, work)
            X = buf2; toggle = 0

    def _Phi_once(a, ids_in, ids_out):
        nonlocal X, toggle
        if toggle == 0:
            apply_Phia_once_gather_numba(X, buf1, deom, model, a, ids_in, ids_out, eta_coeff_for_Phi, work)
            X = buf1; toggle = 1
        else:
            apply_Phia_once_gather_numba(X, buf2, deom, model, a, ids_in, ids_out, eta_coeff_for_Phi, work)
            X = buf2; toggle = 0

    for (op, a) in op_seq:
        ids_in = hop_seq[step]
        ids_out = hop_seq[step + 1]

        if op == "F":
            _F_once(int(a), ids_in, ids_out)
        elif op == "Phi":
            _Phi_once(int(a), ids_in, ids_out)
        else:
            raise ValueError("op must be 'F' or 'Phi'")

        step += 1
        traces[tpos] = np.trace(X[rid])
        tpos += 1

    return X, hop_seq[step], traces


def root_traces_after_op_seq(
    deom, model,
    op_seq: List[Tuple[str, int]],
    work,
    side_F: str = "l",
    side_Phi: str = "l",
    include_initial: bool = True,
):
    """
    便捷接口：自动 build hop_seq，返回 traces（每步后的 root trace）。
    """
    rid = deom.c.root_id
    ids0 = deom.active_ids.astype(np.int32, copy=False)
    if not np.any(ids0 == rid):
        ids0 = np.sort(np.concatenate([ids0, np.array([rid], dtype=np.int32)]))

    hop_total = len(op_seq)
    hop_seq = build_hop_sequence(deom.c, ids0, hop_total, work)

    etaF = model.etal if side_F.lower().startswith("l") else model.etar
    etaP = model.etal if side_Phi.lower().startswith("l") else model.etar

    _, _, traces = apply_op_seq_with_hops_record_root_trace(
        deom.ddos, deom, model,
        op_seq, hop_seq,
        buf1=deom.ddos_temp1,
        buf2=deom.ddos_temp2,
        eta_coeff_for_F=etaF,
        eta_coeff_for_Phi=etaP,
        work=work,
        include_initial=include_initial,
    )
    return traces
