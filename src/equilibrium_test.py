# import numpy as np
# import time

# from .deom import EOMSpec, Model, build_ctrl_A, DEOM, FilterCfg, TimeStepperCfg
# from .rem_gen import precompute_n_dot_gamma, rem_gen
# from .algebra import RHSWorkspace


# def equilibrium_euler(
#   model: Model,
#   eom: EOMSpec,
#   time_cfg: TimeStepperCfg,
#   filt_cfg: FilterCfg,
#   *,
#   log_every: int = 50,   # 每隔多少步打印一次
#   log_first: int = 10,   # 前多少步每步都打印
# ):
#   c = build_ctrl_A(nind=model.nind, lmax=model.lmax)
#   deom = DEOM.allocate(c, nsys=model.nsys)
#   deom.ddos[:] = 0.0
#   deom.ddos[c.root_id] = model.rho0

#   print(c.Nado, "ADO total")

#   n_dot_gamma = precompute_n_dot_gamma(c, model.gamma)
#   work = RHSWorkspace.allocate(c.Nado)

#   ti, tf, dt = time_cfg.ti, time_cfg.tf, time_cfg.dt
#   nstep = int(np.floor((tf - ti) / dt + 1e-12))

#   t_hist = np.zeros(nstep + 1, dtype=np.float64)
#   rho0_hist = np.zeros((nstep + 1, model.nsys, model.nsys), dtype=np.complex128)
#   t_hist[0] = ti
#   rho0_hist[0] = deom.ddos[c.root_id]

#   # --- timing accumulators ---
#   t_filter_sum = 0.0
#   t_rhs_sum = 0.0
#   t_update_sum = 0.0

#   for istep in range(1, nstep + 1):
#     # --- timed Euler step: filter -> RHS -> update ---
#     t0 = time.perf_counter()

#     deom.update_filter(filt_cfg)
#     deom.zero_inactive()

#     t1 = time.perf_counter()

#     # RHS buffer：推荐用 ddos3（避免 ddos1/ddos2 与递归中间缓存冲突）
#     rhs_buf = deom.ddos3
#     ids = rem_gen(deom, model, eom, n_dot_gamma, rhs_buf, work)

#     t2 = time.perf_counter()

#     deom.ddos[ids] += dt * rhs_buf[ids]

#     t3 = time.perf_counter()

#     t_filter_sum += (t1 - t0)
#     t_rhs_sum += (t2 - t1)
#     t_update_sum += (t3 - t2)

#     # logging（注意：print 会显著拖慢）
#     if istep <= log_first or (log_every > 0 and istep % log_every == 0):
#       print(
#         f"step {istep:6d}: Nactive={deom.active_ids.size:6d}  ids_total={ids.size:6d}  "
#         f"t_filter={t1-t0:.3e}s  t_rhs={t2-t1:.3e}s  t_update={t3-t2:.3e}s"
#       )

#     t_hist[istep] = ti + istep * dt
#     rho0_hist[istep] = deom.ddos[c.root_id]

#   if nstep > 0:
#     print(
#       "avg per step: "
#       f"t_filter={t_filter_sum/nstep:.3e}s, "
#       f"t_rhs={t_rhs_sum/nstep:.3e}s, "
#       f"t_update={t_update_sum/nstep:.3e}s"
#     )

#   return t_hist, rho0_hist