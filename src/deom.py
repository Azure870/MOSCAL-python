from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from math import comb

# -------------------------
# Filter configuration
# -------------------------
@dataclass(frozen=True)
class FilterCfg:
  eps_on: float = 1e-10
  norm: str = "fro" # "fro" 或 "maxabs"
  never_drop_root: bool = True
  drop_only_if_order_ge: int = 1 # 默认只剪 |n|>=1
  
# -------------------------
# Hierarchy (A truncation)
# -------------------------
@dataclass
class Ctrl:
  nind: int
  lmax: int
  id2n: np.ndarray # (Nado, K) int32
  n2id: Dict[Tuple[int, ...], int]
  order: np.ndarray # (Nado,) int32, |n| = sum_k n_k
  up: np.ndarray # (Nado, K) int32, id(n+e_k) or -1
  dn: np.ndarray # (Nado, K) int32, id(n-e_k) or -1
  root_id: int = 0

  @property
  def Nado(self) -> int:
    return int(self.id2n.shape[0])

@dataclass(frozen=True)
class TimeStepperCfg:
    ti: float
    tf: float
    dt: float

@dataclass
class DEOM:
  c: Ctrl
  nsys: int
  ddos: np.ndarray # (Nado, d, d) complex128
  ddos1: np.ndarray # (Nado, d, d) complex128
  ddos2: np.ndarray # (Nado, d, d) complex128
  ddos3: np.ndarray # (Nado, d, d) complex128
  ddos4: np.ndarray # (Nado, d, d) complex128
  ddos_temp1: np.ndarray # (Nado, d, d) complex128
  ddos_temp2: np.ndarray # (Nado, d, d) complex128
  mask: np.ndarray # (Nado,) bool
  active_ids: np.ndarray # (Nactive,) int32
  last_ids: np.ndarray
  norm_cache: np.ndarray # (Nado,) float64
  
  @classmethod
  def allocate(cls, c: Ctrl, nsys: int) -> "DEOM":
    ddos = np.zeros((c.Nado, nsys, nsys), dtype=np.complex128)
    ddos1 = np.zeros((c.Nado, nsys, nsys), dtype=np.complex128)
    ddos2 = np.zeros((c.Nado, nsys, nsys), dtype=np.complex128)
    ddos3 = np.zeros((c.Nado, nsys, nsys), dtype=np.complex128)
    ddos4 = np.zeros((c.Nado, nsys, nsys), dtype=np.complex128)
    ddos_temp1 = np.zeros((c.Nado, nsys, nsys), dtype=np.complex128)
    ddos_temp2 = np.zeros((c.Nado, nsys, nsys), dtype=np.complex128)
    mask = np.ones(c.Nado, dtype=bool)
    active_ids = np.arange(c.Nado, dtype=np.int32)
    last_ids = active_ids.copy()
    norm_cache = np.zeros(c.Nado, dtype=np.float64)
    return cls(c=c, nsys=nsys, ddos=ddos, ddos1=ddos1, ddos2=ddos2, ddos3=ddos3, ddos4=ddos4, ddos_temp1=ddos_temp1, ddos_temp2=ddos_temp2, mask=mask, active_ids=active_ids, last_ids=last_ids, norm_cache=norm_cache)
  
  def compute_norms(self, ddos, mode: str = "fro") -> np.ndarray:
    if mode == "fro":
      norms = np.sqrt(np.sum(np.abs(ddos)**2, axis=(1, 2)))
    elif mode == "maxabs":
      norms = np.max(np.abs(ddos), axis=(1, 2))
    else:
      raise ValueError("norm must be 'fro' or 'maxabs'")
    return norms
  
  def update_filter(self, ddos, cfg: FilterCfg) -> None:
    norms = self.compute_norms(ddos, cfg.norm)
    self.norm_cache[:] = norms

    # hysteresis with fixed thresholds
    activate = norms >= cfg.eps_on
    deactivate = norms < cfg.eps_on

    self.mask |= activate
    self.mask &= ~deactivate

    # enforce "only drop order>=X"
    if cfg.drop_only_if_order_ge > 0:
      keep_low = self.c.order < cfg.drop_only_if_order_ge
      self.mask |= keep_low

    if cfg.never_drop_root:
      self.mask[self.c.root_id] = True
  
    self.active_ids = np.flatnonzero(self.mask).astype(np.int32)

  def zero_inactive(self, ddos) -> None:
    ddos[~self.mask, :, :] = 0.0
    
  def update_filter_subset(self, ddos, cfg: FilterCfg, ids_check: np.ndarray) -> np.ndarray:
    ids = ids_check
    if ids.dtype != np.int32:
      ids = ids.astype(np.int32)
    
    X = ddos[ids]
    if cfg.norm == "fro":
      norms = np.sqrt(np.sum(np.abs(X)**2, axis=(1, 2)))
    elif cfg.norm == "maxabs":
      norms = np.max(np.abs(X), axis=(1, 2))
    else:
      raise ValueError("norm must be 'fro' or 'maxabs'")
    
    self.norm_cache[ids] = norms
    old = self.mask[ids].copy()
    new = norms >= cfg.eps_on
    
    if cfg.drop_only_if_order_ge > 0:
      keep_low = self.c.order[ids] < cfg.drop_only_if_order_ge
      new |= keep_low
      
    if cfg.never_drop_root:
      rid = self.c.root_id
      # 如果 root 在 ids_check 里，强制置 True
      # （即使不在 ids_check，也没关系：root 的 mask 保持 True 不变）
      for j in range(ids.shape[0]):
        if ids[j] == rid:
          new[j] = True
          break
      self.mask[rid] = True
      
    deactivate_mask = old & (~new)
    activate_mask   = (~old) & new
    
    deactivate_ids = ids[deactivate_mask]
    activate_ids   = ids[activate_mask]
    
    self.mask[deactivate_ids] = False
    self.mask[activate_ids] = True
    
    a = self.active_ids
    a = a[self.mask[a]]
    
    if activate_ids.size > 0:
      a = np.concatenate([a, activate_ids]).astype(np.int32)
      a.sort()  # 可选：保持有序，方便调试/确定性
    self.active_ids = a
    
    return deactivate_ids

    
    
# -------------------------
# Model: physical parameters + F definitions
# -------------------------
@dataclass
class Model:
  nsys: int
  nind: int
  lmax: int
  hams: np.ndarray # (d,d)
  rho0: np.ndarray # (d,d)
  gamma: np.ndarray # (nind,) complex
  etal: np.ndarray # (nind,) complex
  etar: np.ndarray # (nind,) complex
  etaa: np.ndarray # (nind,) complex
  # interactions: Optional[Interactions] = None
  # F: Dict[str, np.ndarray] = field(default_factory=dict) # name -> p_a[k] (nind,)

  # NEW: list of system operators Q_q (including Q12, ...)
  Q_list: Optional[np.ndarray] = None # (nQ, nsys, nsys)

  # NEW: F channels coefficients p_a,k defining F_a = sum_k p[a,k] f_k
  p_list: Optional[np.ndarray] = None # (nF, nind)

  def __post_init__(self):
    self.hams = np.asarray(self.hams, dtype=np.complex128)
    if self.hams.shape != (self.nsys, self.nsys):
      raise ValueError(f"hams must have shape ({self.nsys},{self.nsys}), got {self.hams.shape}")
    
    self.rho0 = np.asarray(self.rho0, dtype=np.complex128)
    if self.rho0.shape != (self.nsys, self.nsys):
      raise ValueError(f"rho0 must have shape ({self.nsys},{self.nsys}), got {self.rho0.shape}")

    self.gamma = np.asarray(self.gamma, dtype=np.complex128)
    if self.gamma.shape != (self.nind,):
      raise ValueError(f"gamma must have shape (nind,) = ({self.nind},), got {self.gamma.shape}")

    self.etal = np.asarray(self.etal, dtype=np.complex128)
    if self.etal.shape != (self.nind,):
      raise ValueError(f"etal must have shape (nind,) = ({self.nind},), got {self.etal.shape}")

    self.etar = np.asarray(self.etar, dtype=np.complex128)
    if self.etar.shape != (self.nind,):
      raise ValueError(f"etar must have shape (nind,) = ({self.nind},), got {self.etar.shape}")

    self.etaa = np.asarray(self.etaa, dtype=np.complex128)
    if self.etaa.shape != (self.nind,):
      raise ValueError(f"etaa must have shape (nind,) = ({self.nind},), got {self.etaa.shape}")
    
    if self.Q_list is not None:
      self.Q_list = np.asarray(self.Q_list, dtype=np.complex128)
      if self.Q_list.ndim != 3 or self.Q_list.shape[1:] != (self.nsys, self.nsys):
        raise ValueError(f"Q_list must be shape (nQ,{self.nsys},{self.nsys}), got {self.Q_list.shape}")

    if self.p_list is not None:
      self.p_list = np.asarray(self.p_list, dtype=np.complex128)
      if self.p_list.ndim != 2 or self.p_list.shape[1] != self.nind:
        raise ValueError(f"p_list must be shape (nF,{self.nind}), got {self.p_list.shape}")

  @property
  def nQ(self) -> int:
    return 0 if self.Q_list is None else int(self.Q_list.shape[0])

  @property
  def nF(self) -> int:
    return 0 if self.p_list is None else int(self.p_list.shape[0])
    
    
# -------------------------
# EOM specification (declarative)
# -------------------------
FSeqI = List[Tuple[int, int]] # [(a, power), (b, power)] a indexes p_list row

@dataclass
class EOMSpec:
  include_drift: bool = True

  # Each term: Q[q] * rho(; product over F_seq in order)
  # q indexes Q_list, a indexes p_list
  terms: List[Tuple[int, FSeqI]] = field(default_factory=list)
  
  def drift(self, on: bool = True) -> "EOMSpec":
    self.include_drift = on
    return self
  
  def add_term(self, q: int, F_seq: FSeqI) -> "EOMSpec":
    self.terms.append((int(q), [(int(a), int(p)) for a, p in F_seq]))
    return self

  # convenience
  def add_Fpower(self, q: int, a: int, power: int = 1) -> "EOMSpec":
  # single F^power with its matching Q
    return self.add_term(q=q, F_seq=[(a, power)])

  def add_Fmix(self, q: int, a: int, pow_a: int, b: int, pow_b: int) -> "EOMSpec":
  # ordered two-F term with its matching Q (e.g. Q12)
    return self.add_term(q=q, F_seq=[(a, pow_a), (b, pow_b)])
  
  
  
def _fill_compositions_fixed_sum(out: np.ndarray, idx: int, nind: int, total: int) -> int:
  """
  Fill 'out' with all nonnegative integer vectors n of length nind such that sum(n)=total.
  Enumeration order: lexicographic in n[0], then n[1], ... (given fixed total).
  Returns next free row index.
  """
  n = np.zeros(nind, dtype=np.int32)

  def rec(pos: int, rem: int, row: int) -> int:
    if pos == nind - 1:
      n[pos] = rem
      out[row, :] = n
      return row + 1
    for v in range(rem + 1):
      n[pos] = v
      row = rec(pos + 1, rem - v, row)
    return row
    
  return rec(0, total, idx)


def build_ctrl_A(nind: int, lmax: int) -> "Ctrl":
  """
  Build A-truncated hierarchy: all n in N^{nind} with |n|<=lmax.
  Returns Ctrl with id2n/order/up/dn/n2id/root_id populated.
  """
  if nind <= 0 or lmax < 0:
    raise ValueError("nind must be >0 and lmax must be >=0")

  # Total number of multi-indices with sum<=L is C(nind+L, L)
  Nado = comb(nind + lmax, lmax)

  id2n = np.empty((Nado, nind), dtype=np.int32)
  order = np.empty((Nado,), dtype=np.int32)
  
  # Enumerate by total order increasing: 0,1,2,...,lmax
  row = 0
  for L in range(lmax + 1):
    row0 = row
    row = _fill_compositions_fixed_sum(id2n, row, nind, L)
    order[row0:row] = L
  
  if row != Nado:
    raise RuntimeError(f"internal error: filled {row} rows but expected {Nado}")
  
  # Build n2id mapping
  n2id: Dict[Tuple[int, ...], int] = {}
  for i in range(Nado):
    n2id[tuple(int(x) for x in id2n[i])] = i

  # root id (all zeros). With our enumeration, it should be 0.
  root_tuple = (0,) * nind
  root_id = n2id[root_tuple]

  # Neighbor tables
  up = np.full((Nado, nind), -1, dtype=np.int32)
  dn = np.full((Nado, nind), -1, dtype=np.int32)
  
  # Fill up/dn
  # NOTE: we use n2id lookups here only once at construction time.
  for i in range(Nado):
    n = id2n[i]
    L = order[i]

    # up: only possible if L < lmax
    if L < lmax:
      for k in range(nind):
        n_up = n.copy()
        n_up[k] += 1
        # sum is L+1 <= lmax guaranteed
        up[i, k] = n2id[tuple(int(x) for x in n_up)]

    # dn: only if n[k] > 0
    for k in range(nind):
      if n[k] > 0:
        n_dn = n.copy()
        n_dn[k] -= 1
        dn[i, k] = n2id[tuple(int(x) for x in n_dn)]
  
  # Create Ctrl
  c = Ctrl(
    nind=nind,
    lmax=lmax,
    id2n=id2n,
    n2id=n2id,
    order=order,
    up=up,
    dn=dn,
    root_id=root_id,
  )
  return c

def sanity_check_ctrl(c: "Ctrl", n_checks: int = 50, seed: int = 0) -> None:
  """
  Quick sanity checks: verifies neighbor correctness for random (id,k).
  Raises AssertionError if something is inconsistent.
  """
  rng = np.random.default_rng(seed)
  Nado, nind, lmax = c.Nado, c.nind, c.lmax
  
  # root check
  assert c.root_id == c.n2id[(0,) * nind]
  assert np.all(c.id2n[c.root_id] == 0)
  
  for _ in range(n_checks):
    i = int(rng.integers(0, Nado))
    k = int(rng.integers(0, nind))
    n = c.id2n[i]
    L = int(c.order[i])

    iu = int(c.up[i, k])
    if L < lmax:
      assert iu != -1
      n2 = c.id2n[iu]
      assert np.all(n2 == n + np.eye(nind, dtype=np.int32)[k])
      assert int(c.order[iu]) == L + 1
    else:
      assert iu == -1

    idn = int(c.dn[i, k])
    if n[k] > 0:
      assert idn != -1
      n2 = c.id2n[idn]
      e = np.zeros(nind, dtype=np.int32)
      e[k] = 1
      assert np.all(n2 == n - e)
      assert int(c.order[idn]) == L - 1
    else:
      assert idn == -1