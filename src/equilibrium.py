import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from numpy.lib.format import open_memmap


from .deom import EOMSpec, Model, build_ctrl_A, DEOM, FilterCfg, TimeStepperCfg
from .rem_gen import precompute_n_dot_gamma, root_traces_after_op_seq
from .rk import euler, rk4
from .algebra import RHSWorkspace
from .print import collect_second_tier_traces

def equilibrium_euler(
  model: Model,
  eom: EOMSpec,
  time_cfg: TimeStepperCfg,
  filt_cfg: FilterCfg,
):
  c = build_ctrl_A(nind=model.nind, lmax=model.lmax)
  deom = DEOM.allocate(c, nsys=model.nsys)
  deom.ddos[:] = 0.0
  deom.ddos[c.root_id] = model.rho0
  
  # print(c.Nado, "ADO total")
  
  n_dot_gamma = precompute_n_dot_gamma(c, model.gamma)
  work = RHSWorkspace.allocate(c.Nado)
  ti, tf, dt = time_cfg.ti, time_cfg.tf, time_cfg.dt
  nstep = int(np.floor((tf - ti) / dt + 1e-12))
  
  t_hist = np.zeros(nstep + 1, dtype=np.float64)
  rho0_hist = np.zeros((nstep + 1, model.nsys, model.nsys), dtype=np.complex128)
  t_hist[0] = ti
  rho0_hist[0] = deom.ddos[c.root_id]
  
  for istep in range(1, nstep + 1):
    # one Euler step (rhs uses deom.ddos3 by default if you pass it)
    euler(
      deom=deom,
      model=model,
      eom=eom,
      n_dot_gamma=n_dot_gamma,
      dt=dt,
      filt=filt_cfg,
      work=work,
    )

    t_hist[istep] = ti + istep * dt
    rho0_hist[istep] = deom.ddos[c.root_id]
    
  return t_hist, rho0_hist
  
  
def equilibrium_rk4_for_test(
  model: Model,
  eom: EOMSpec,
  time_cfg: TimeStepperCfg,
  filt_cfg: FilterCfg,
):
  c = build_ctrl_A(nind=model.nind, lmax=model.lmax)
  deom = DEOM.allocate(c, nsys=model.nsys)
  deom.ddos[:] = 0.0
  deom.ddos[c.root_id] = model.rho0
  
  # print(c.Nado, "ADO total")
  
  n_dot_gamma = precompute_n_dot_gamma(c, model.gamma)
  work = RHSWorkspace.allocate(c.Nado)
  ti, tf, dt = time_cfg.ti, time_cfg.tf, time_cfg.dt
  nstep = int(np.floor((tf - ti) / dt + 1e-12))
  
  t_hist = np.zeros(nstep + 1, dtype=np.float64)
  rho0_hist = np.zeros((nstep + 1, model.nsys, model.nsys), dtype=np.complex128)
  curr_hist = np.zeros((nstep + 1, model.nind, model.nind), dtype=np.complex128)
  t_hist[0] = ti
  rho0_hist[0] = deom.ddos[c.root_id]
  
  # for istep in range(1, nstep + 1):
  for istep in tqdm(range(1, nstep + 1), total=nstep, desc="DEOM evolve", unit="step"):
    # one Euler step (rhs uses deom.ddos3 by default if you pass it)
    rk4(
      deom=deom,
      model=model,
      eom=eom,
      n_dot_gamma=n_dot_gamma,
      dt=dt,
      filt=filt_cfg,
      work=work,
    )

    t_hist[istep] = ti + istep * dt
    rho0_hist[istep] = deom.ddos[c.root_id]
    curr_hist[istep] = collect_second_tier_traces(deom, model.nind)
    
    
  return t_hist, rho0_hist, curr_hist

def equilibrium_rk4_npy(
  model: Model,
  eom: EOMSpec,
  time_cfg: TimeStepperCfg,
  filt_cfg: FilterCfg,
  out_path: str = "deom_out.npy",
  flush_every: int = 1,   # 1 表示每一步都 flush；如果嫌慢可以改成 10/50/100
):
  c = build_ctrl_A(nind=model.nind, lmax=model.lmax)
  deom = DEOM.allocate(c, nsys=model.nsys)
  deom.ddos[:] = 0.0
  deom.ddos[c.root_id] = model.rho0

  n_dot_gamma = precompute_n_dot_gamma(c, model.gamma)
  work = RHSWorkspace.allocate(c.Nado)
  ti, tf, dt = time_cfg.ti, time_cfg.tf, time_cfg.dt
  nstep = int(np.floor((tf - ti) / dt + 1e-12))

  # --- 关键：创建单文件 memmap（.npy 有 header，可直接 np.load 读）---
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  rec_dtype = np.dtype([
      ("t",   np.float64),
      ("rho0", np.complex128, (model.nsys, model.nsys)),
      ("curr", np.complex128, (model.nind, model.nind)),
  ])

  out = open_memmap(
      filename=str(out_path),
      mode="w+",
      dtype=rec_dtype,
      shape=(nstep + 1,)
  )

  # step 0
  out[0]["t"] = ti
  out[0]["rho0"] = deom.ddos[c.root_id]
  out[0]["curr"] = collect_second_tier_traces(deom, model.nind)

  if flush_every == 1:
      out.flush()

  for istep in tqdm(range(1, nstep + 1), total=nstep, desc="DEOM evolve", unit="step"):
    rk4(
      deom=deom,
      model=model,
      eom=eom,
      n_dot_gamma=n_dot_gamma,
      dt=dt,
      filt=filt_cfg,
      work=work,
    )

    t = ti + istep * dt
    out[istep]["t"] = t
    out[istep]["rho0"] = deom.ddos[c.root_id]
    out[istep]["curr"] = collect_second_tier_traces(deom, model.nind)

    # flush 策略：你要求“每一步都输出到文件”，严格来说 flush_every=1 最符合。
    # 但 flush 会有 IO 开销，如果允许“每步写入但隔几步落盘”，可把 flush_every 调大。
    if flush_every > 0 and (istep % flush_every == 0):
        out.flush()

  out.flush()
  return out_path

def equilibrium_rk4(
    model: Model,
    eom: EOMSpec,
    time_cfg: TimeStepperCfg,
    filt_cfg: FilterCfg,
    rho0_path: str = "rho0.txt",
    curr_path: str = "curr.txt",
):
  
  print("Starting equilibrium_rk4...")
  # print(""
  
  # --- init DEOM ---
  c = build_ctrl_A(nind=model.nind, lmax=model.lmax)
  deom = DEOM.allocate(c, nsys=model.nsys)
  deom.ddos[:] = 0.0
  deom.ddos[c.root_id] = model.rho0

  n_dot_gamma = precompute_n_dot_gamma(c, model.gamma)
  work = RHSWorkspace.allocate(c.Nado)

  ti, tf, dt = time_cfg.ti, time_cfg.tf, time_cfg.dt
  nstep = int(np.floor((tf - ti) / dt + 1e-12))

  # --- helper: write one line exactly like your Julia ---
  # line: t\t  Re  Im  Re  Im  ... \n
  def _write_mat_line(io, t: float, mat: np.ndarray):
    io.write(f"{t}\t")
    flat = np.asarray(mat, dtype=np.complex128).ravel(order="C")
    for z in flat:
      io.write(f"{z.real}  {z.imag}  ")
    io.write("\n")

  # --- open files and stream output ---
  with open(rho0_path, "w") as fr, open(curr_path, "w") as fc:
    # step 0
    t = ti
    rho0 = deom.ddos[c.root_id]
    curr = collect_second_tier_traces(deom, model.nind)

    _write_mat_line(fr, t, rho0)
    _write_mat_line(fc, t, curr)

    # evolve
    for ii in tqdm(range(1, nstep + 1), total=nstep, desc="DEOM evolve", unit="step"):
      rk4(
        deom=deom,
        model=model,
        eom=eom,
        n_dot_gamma=n_dot_gamma,
        dt=dt,
        filt=filt_cfg,
        work=work,
      )
      t = ti + ii * dt
      rho0 = deom.ddos[c.root_id]
      curr = collect_second_tier_traces(deom, model.nind)

      _write_mat_line(fr, t, rho0)
      _write_mat_line(fc, t, curr)

      # 如果你希望“边跑边立刻能看到文件增长”，取消下面两行注释（会慢一点）
      fr.flush()
      fc.flush()

  return None


def equilibrium_rk4_current(
    model: Model,
    eom: EOMSpec,
    time_cfg: TimeStepperCfg,
    filt_cfg: FilterCfg,
    rho0_path: str = "rho0.txt",
    curr_path: str = "traces.txt",   # 建议改名：这里不再是 curr
):
    print("Starting equilibrium_rk4_current...")

    # --- init DEOM ---
    c = build_ctrl_A(nind=model.nind, lmax=model.lmax)
    deom = DEOM.allocate(c, nsys=model.nsys)
    deom.ddos[:] = 0.0
    deom.ddos[c.root_id] = model.rho0

    n_dot_gamma = precompute_n_dot_gamma(c, model.gamma)
    work = RHSWorkspace.allocate(c.Nado)

    ti, tf, dt = time_cfg.ti, time_cfg.tf, time_cfg.dt
    nstep = int(np.floor((tf - ti) / dt + 1e-12))

    # --- helper: write one line exactly like your Julia ---
    def _write_mat_line(io, t: float, mat: np.ndarray):
        io.write(f"{t}\t")
        flat = np.asarray(mat, dtype=np.complex128).ravel(order="C")
        for z in flat:
            io.write(f"{z.real}  {z.imag}  ")
        io.write("\n")

    # ---- 你要的全局顺序：先 Phi 再 F（例：Phi_0, F_1^3）----
    op_seq = [("Phi", 0), ("F", 1), ("F", 1), ("F", 1)]
    # op_seq = [("F", 0), ("F", 0), ("F", 0)]

    # 左右系数选择：按你当前 Phi 实现，Phi 也需要 eta_coeff（etal/etar）
    side_F = "l"
    side_Phi = "l"

    with open(rho0_path, "w") as fr, open(curr_path, "w") as ft:
        # step 0
        t = ti
        rho0 = deom.ddos[c.root_id]
        traces = root_traces_after_op_seq(
            deom, model, op_seq, work,
            side_F=side_F,
            side_Phi=side_Phi,
            include_initial=True,
        )

        _write_mat_line(fr, t, rho0)
        _write_mat_line(ft, t, traces)

        # evolve
        for ii in tqdm(range(1, nstep + 1), total=nstep, desc="DEOM evolve", unit="step"):
            rk4(
                deom=deom,
                model=model,
                eom=eom,
                n_dot_gamma=n_dot_gamma,
                dt=dt,
                filt=filt_cfg,
                work=work,
            )

            t = ti + ii * dt
            rho0 = deom.ddos[c.root_id]

            # 输出：每作用一次 F/Phi 之后的 root trace 序列
            traces = root_traces_after_op_seq(
                deom, model, op_seq, work,
                side_F=side_F,
                side_Phi=side_Phi,
                include_initial=True,
            )

            _write_mat_line(fr, t, rho0)
            _write_mat_line(ft, t, traces)

            fr.flush()
            ft.flush()

    return None

