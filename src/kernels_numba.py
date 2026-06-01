# src/kernels_numba.py
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def add_drift_subset_nb(inp, out, ids_in, hams, n_dot_gamma):
    d = hams.shape[0]
    for t in prange(ids_in.shape[0]):
        i = ids_in[t]
        # out[i] += -1j*(H@R - R@H) - (n·gamma)R
        for a in range(d):
            for b in range(d):
                # (H@R)[a,b]
                s1 = 0.0 + 0.0j
                # (R@H)[a,b]
                s2 = 0.0 + 0.0j
                for j in range(d):
                    s1 += hams[a, j] * inp[i, j, b]
                    s2 += inp[i, a, j] * hams[j, b]
                out[i, a, b] += (-1j) * (s1 - s2) - \
                    n_dot_gamma[i] * inp[i, a, b]


@njit(cache=True, parallel=True)
def add_Q_left_nb(ids, Q, X, out):
    nsys = Q.shape[0]
    for t in prange(ids.shape[0]):
        i = ids[t]
        # out[i] += -1j * (Q @ X[i])
        for a in range(nsys):
            for b in range(nsys):
                s = 0.0 + 0.0j
                for j in range(nsys):
                    s += Q[a, j] * X[i, j, b]
                out[i, a, b] += (-1j) * s


@njit(cache=True, parallel=True)
def add_Q_right_nb(ids, Q, X, out):
    nsys = Q.shape[0]
    for t in prange(ids.shape[0]):
        i = ids[t]
        # out[i] += +1j * (X[i] @ Q)
        for a in range(nsys):
            for b in range(nsys):
                s = 0.0 + 0.0j
                for j in range(nsys):
                    s += X[i, a, j] * Q[j, b]
                out[i, a, b] += (1j) * s


@njit(cache=True, parallel=True)
def apply_Fa_once_gather_eta_nb(inp, out, ids_out, mark, p, up, dn, id2n, eta_coeff, etaa):
    """
    只做 gather：每个 n in ids_out 独占写 out[n,:,:]，天然可并行
    out[n] = sum_k p[k] * [ sqrt(nk+1)*sqrt(etaa[k]) * inp[n+e_k]  +  sqrt(nk)*eta[k]/sqrt(etaa[k]) * inp[n-e_k] ]
    注意：是否存在邻居由 up/dn 以及 mark 控制
    """
    d = inp.shape[1]
    nind = p.shape[0]

    # 1) 并行清零 ids_out
    for t in prange(ids_out.shape[0]):
        n = ids_out[t]
        for a in range(d):
            for b in range(d):
                out[n, a, b] = 0.0 + 0.0j

    # 2) 并行计算
    for t in prange(ids_out.shape[0]):
        n = ids_out[t]
        for k in range(nind):
            pk = p[k]
            if pk == 0:
                continue

            nk = id2n[n, k]

            iu = up[n, k]
            if iu != -1 and mark[iu]:
                coef = pk * np.sqrt(nk + 1.0) * np.sqrt(etaa[k])
                for a in range(d):
                    for b in range(d):
                        out[n, a, b] += coef * inp[iu, a, b]

            idn = dn[n, k]
            if idn != -1 and mark[idn] and nk > 0:
                coef = pk * np.sqrt(nk) * (eta_coeff[k] / np.sqrt(etaa[k]))
                for a in range(d):
                    for b in range(d):
                        out[n, a, b] += coef * inp[idn, a, b]


@njit(cache=True)
def expand_one_hop_list_nb(ids_in, up, dn, mark, buf_ids):
    """
    输入: ids_in (int32)
    输出: 把 ids_out 写入 buf_ids[0:cnt]，并在内部排序；返回 cnt
    特点: 不用 flatnonzero，不用全表扫描，靠“首次标记时入队”避免重复。
    """
    # clear marks
    mark[:] = False

    cnt = 0
    nind = up.shape[1]

    for t in range(ids_in.shape[0]):
        m = ids_in[t]

        if not mark[m]:
            mark[m] = True
            buf_ids[cnt] = m
            cnt += 1

        for k in range(nind):
            iu = up[m, k]
            if iu != -1 and (not mark[iu]):
                mark[iu] = True
                buf_ids[cnt] = iu
                cnt += 1

            idn = dn[m, k]
            if idn != -1 and (not mark[idn]):
                mark[idn] = True
                buf_ids[cnt] = idn
                cnt += 1

    # 让输出 deterministic：排序 buf_ids[:cnt]
    buf_ids[:cnt].sort()
    return cnt


@njit(cache=True, parallel=True)
def apply_Phia_once_gather_nb(
    inp, out, ids_out, mark,
    p, up, dn, id2n,
    gamma, eta_coeff, etaa
):
    """
    Phi_a 的一次作用（gather 形式），实现你给的公式：

      (Phi_a X)[n] = sum_k  -gamma[k] * p[k] * [
            sqrt(nk+1)*sqrt(|eta_k|) * X[n+e_k]
          - sqrt(nk) * eta_coeff[k] / sqrt(|eta_k|)       * X[n-e_k]
      ]
    """
    d = inp.shape[1]
    nind = p.shape[0]

    # 1) 并行清零 ids_out
    for t in prange(ids_out.shape[0]):
        n = ids_out[t]
        for a in range(d):
            for b in range(d):
                out[n, a, b] = 0.0 + 0.0j

    # 2) 并行计算
    for t in prange(ids_out.shape[0]):
        n = ids_out[t]
        for k in range(nind):
            pk = p[k]
            if pk == 0:
                continue

            nk = id2n[n, k]

            # up term: (-gamma p)*sqrt(nk+1)*sqrt(|eta|)
            iu = up[n, k]
            if iu != -1 and mark[iu]:
                coef = pk * np.sqrt(nk + 1.0) * np.sqrt(etaa[k]) * (-gamma[k])
                for a in range(d):
                    for b in range(d):
                        out[n, a, b] += coef * inp[iu, a, b]

            # dn term: (-gamma p)*(- nk / sqrt(|eta|)) = (+gamma p)*(nk/eta_sqrt)
            idn = dn[n, k]
            if idn != -1 and mark[idn] and nk > 0:
                coef = pk * np.sqrt(nk) * (eta_coeff[k] / np.sqrt(etaa[k])) * gamma[k]
                for a in range(d):
                    for b in range(d):
                        out[n, a, b] += coef * inp[idn, a, b]
