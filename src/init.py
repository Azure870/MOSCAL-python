import numpy as np
import yaml

from .deom import Model, EOMSpec, TimeStepperCfg

def _scalar(x):
  # np scalar / 0d array -> python scalar
  if isinstance(x, np.ndarray) and x.shape == ():
    return x.item()
  if isinstance(x, np.generic):
    return x.item()
  return x


def load_inputs(model_yaml_path: str) -> tuple[dict, Model, EOMSpec, TimeStepperCfg]:
  with open(model_yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  data = np.load(cfg["data_npz"])

  # scalars
  nsys = int(_scalar(data["nsys"]))
  nind = int(_scalar(data["nind"]))
  lmax = int(_scalar(data["lmax"]))
  ti = float(_scalar(data["ti"]))
  tf = float(_scalar(data["tf"]))
  dt = float(_scalar(data["dt"]))

  # arrays
  hams = np.asarray(data["hams"], dtype=np.complex128)
  rho0 = np.asarray(data["rho0"], dtype=np.complex128)

  gamma = np.asarray(data["expn"], dtype=np.complex128)
  etal = np.asarray(data["etal"], dtype=np.complex128)
  etar = np.asarray(data["etar"], dtype=np.complex128)
  etaa = np.asarray(data["etaa"], dtype=np.complex128)  # 强制 complex，省心

  Q_list = np.asarray(data["Q_list"], dtype=np.complex128)
  p_list = np.asarray(data["p_list"], dtype=np.complex128)

  # model
  model = Model(
    nsys=nsys, nind=nind, lmax=lmax,
    hams=hams,
    rho0=rho0,
    gamma=gamma,
    etal=etal,
    etar=etar,
    etaa=etaa,
  )
  model.Q_list = Q_list
  model.p_list = p_list

  # time
  time_cfg = TimeStepperCfg(ti=ti, tf=tf, dt=dt)

  # eom
  eom_cfg = cfg.get("eom", {})
  eom = EOMSpec()
  eom.include_drift = bool(eom_cfg.get("include_drift", True))
  eom.terms = []
  for t in eom_cfg.get("terms", []):
    q = int(t["q"])
    F_seq = [(int(a), int(pwr)) for a, pwr in t["F_seq"]]
    eom.terms.append((q, F_seq))

  return cfg, model, eom, time_cfg