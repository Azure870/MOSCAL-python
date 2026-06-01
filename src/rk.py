import numpy as np
from typing import Tuple

from .deom import DEOM, Model, EOMSpec, FilterCfg
from .rem_gen import rem_gen 
from .algebra import RHSWorkspace  

def euler(
  deom: DEOM,
  model: Model,
  eom: EOMSpec,
  n_dot_gamma: np.ndarray,
  dt: float,
  filt: FilterCfg,
  work: RHSWorkspace,
  # rhs_buf: np.ndarray | None = None,
) -> np.ndarray:

  deom.update_filter(deom.ddos, filt)
  deom.zero_inactive(deom.ddos)
  # deom.mask[:] = True
  # deom.active_ids = np.arange(deom.c.Nado, dtype=np.int32)
  ids = rem_gen(deom.ddos, deom.ddos1, deom, model, eom, n_dot_gamma, work)
  deom.ddos[ids] += dt * deom.ddos1[ids]
  return ids

def rk4(
  deom: DEOM,
  model: Model,
  eom: EOMSpec,
  n_dot_gamma: np.ndarray,
  dt: float,
  filt: FilterCfg,
  work: RHSWorkspace,
  # rhs_buf: np.ndarray
) -> np.ndarray:
  
  dt2 = 0.5 * dt
  dt6 = dt / 6.0

  # deom.update_filter(deom.ddos, filt)
  # deom.zero_inactive(deom.ddos)
  
  ids_check = deom.last_ids
  deactivate = deom.update_filter_subset(deom.ddos, filt, ids_check)
  if deactivate.size > 0:
    deom.ddos[deactivate, :, :] = 0.0
  
  # deom.mask[:] = True
  # deom.active_ids = np.arange(deom.c.Nado, dtype=np.int32)

  # ids_acc: 收集 4 个 stage 的并集
  work.marked[:] = False
  
  # ---- stage 1 ----
  ids1 = rem_gen(deom.ddos, deom.ddos1, deom, model, eom, n_dot_gamma, work)  # ddos1 = k1
  work.marked[ids1] = True
  deom.ddos3[ids1] = deom.ddos[ids1] + dt2 * deom.ddos1[ids1]                # y2 = y + dt/2 k1

  # ---- stage 2 ----
  ids2 = rem_gen(deom.ddos3, deom.ddos2, deom, model, eom, n_dot_gamma, work) # ddos2 = k2
  work.marked[ids2] = True
  deom.ddos1[ids2] += 2.0 * deom.ddos2[ids2]                                  # acc += 2 k2
  deom.ddos3[ids2] = deom.ddos[ids2] + dt2 * deom.ddos2[ids2]                 # y3 = y + dt/2 k2

  # ---- stage 3 ----
  ids3 = rem_gen(deom.ddos3, deom.ddos2, deom, model, eom, n_dot_gamma, work) # ddos2 = k3
  work.marked[ids3] = True
  deom.ddos1[ids3] += 2.0 * deom.ddos2[ids3]                                  # acc += 2 k3
  deom.ddos3[ids3] = deom.ddos[ids3] + dt * deom.ddos2[ids3]                  # y4 = y + dt k3

  # ---- stage 4 ----
  ids4 = rem_gen(deom.ddos3, deom.ddos2, deom, model, eom, n_dot_gamma, work) # ddos2 = k4
  work.marked[ids4] = True
  deom.ddos1[ids4] += deom.ddos2[ids4]                                        # acc += k4

  ids_acc = np.flatnonzero(work.marked).astype(np.int32)
  deom.ddos[ids_acc] += dt6 * deom.ddos1[ids_acc]

  deom.last_ids = ids_acc

  return ids_acc

  
  
  
  

