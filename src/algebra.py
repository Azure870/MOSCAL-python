import numpy as np
from dataclasses import dataclass


@dataclass
class RHSWorkspace:
    mark: np.ndarray          # (Nado,) bool
    marked: np.ndarray          # (Nado,) bool
    buf_ids: np.ndarray       # (Nado,) int32 预留，实际用前 Nvalid

    @classmethod
    def allocate(cls, Nado: int) -> "RHSWorkspace":
        return cls(
            mark=np.zeros(Nado, dtype=bool),
            buf_ids=np.empty(Nado, dtype=np.int32),
            marked=np.zeros(Nado, dtype=bool),
        )
