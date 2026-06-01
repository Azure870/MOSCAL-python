import numpy as np

def check_ctrl_topology(c, ntest=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = c.Nado
    K = c.nind

    # 随机抽一些 id
    ids = rng.integers(0, N, size=min(ntest, N), dtype=np.int32)

    for m in ids:
        nm = c.id2n[m]

        for k in range(K):
            iu = c.up[m, k]
            if iu != -1:
                nu = c.id2n[iu]
                if not np.all(nu == nm + (np.arange(K)==k).astype(nm.dtype)):
                    raise AssertionError(f"up mismatch at m={m},k={k}: nm={nm}, nu={nu}")
                # inverse check
                back = c.dn[iu, k]
                if back != m:
                    raise AssertionError(f"up/dn inverse mismatch: dn[up[m,k],k] != m, m={m},k={k}")

            idn = c.dn[m, k]
            if idn != -1:
                nd = c.id2n[idn]
                if not np.all(nd == nm - (np.arange(K)==k).astype(nm.dtype)):
                    raise AssertionError(f"dn mismatch at m={m},k={k}: nm={nm}, nd={nd}")
                # inverse check
                back = c.up[idn, k]
                if back != m:
                    raise AssertionError(f"dn/up inverse mismatch: up[dn[m,k],k] != m, m={m},k={k}")

    print("Ctrl topology OK")
    
def check_expand_one_hop(c, expand_one_hop, work, ntest=200, seed=1):
    rng = np.random.default_rng(seed)
    N = c.Nado
    K = c.nind

    for _ in range(ntest):
        # 随机一个小集合
        ids_in = np.unique(rng.integers(0, N, size=10, dtype=np.int32))
        out = expand_one_hop(c, ids_in, work)

        # 朴素参考集合
        ref = set(int(x) for x in ids_in)
        for m in ids_in:
            for k in range(K):
                iu = c.up[m,k]
                if iu != -1: ref.add(int(iu))
                idn = c.dn[m,k]
                if idn != -1: ref.add(int(idn))

        ref = np.array(sorted(ref), dtype=np.int32)
        if out.size != ref.size or np.any(out != ref):
            raise AssertionError("expand_one_hop mismatch")
    print("expand_one_hop OK")