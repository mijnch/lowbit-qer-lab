"""행렬 하나를 한 가지 방법으로 압축한 Ŵ를 돌려준다 — T2(모든 선형층)에서 층마다 부른다.

method:
  "fp"     : 그대로
  "rtn"    : Q_b(W)
  "qer"    : Q_b(W) + SVD_r(W − Q_b(W))                     (분할의 k = 0)
  "split"  : P_k + Q_b(W − P_k) + SVD_(r−k)(E_k),  k는 k="surrogate"(대리지표) · 정수 · "best"(전수 탐색) ·
             "rho"(ρ_r(W) < rho_threshold 이면 k = r(전부 보존), 아니면 k = 0(QER) — T2의 층별 결과에서 나온 규칙)
"""
import numpy as np

from .budget import bits_per_weight, rank_for_extra_bits
from .lowrank import rho
from .quant import rtn
from .split import reconstruct, split_loss_curve, surrogate_curve


def compress_matrix(w, method, bits=4, group=128, symmetric=False, rank=None, extra_bits=1.0, k="surrogate",
                    rho_threshold=0.5):
    """→ (Ŵ, info). rank가 None이면 extra_bits 예산으로 정한다."""
    w = np.asarray(w, np.float64)
    m, n = w.shape
    info = {"method": method, "shape": [m, n]}
    if method == "fp":
        info.update(bits_per_weight=16.0, rel_error=0.0)
        return w, info
    if method == "rtn":
        w_hat = rtn(w, bits, group, symmetric)
        info.update(bits=bits, rank=0, bits_per_weight=bits_per_weight(bits, m, n, 0, group, symmetric))
    else:
        r = rank_for_extra_bits(extra_bits, m, n) if rank is None else rank
        if method == "qer":
            kk = 0
        elif method == "split":
            if k == "surrogate":
                kk = surrogate_curve(w, bits, r, group, symmetric)["k_star"]
            elif k == "best":
                kk = int(np.argmin(split_loss_curve(w, bits, r, group, symmetric)["loss"]))
            elif k == "rho":
                kk = r if rho(w, r) < rho_threshold else 0
            else:
                kk = int(k)
        else:
            raise ValueError(method)
        w_hat, _ = reconstruct(w, bits, r, kk, group, symmetric)
        info.update(bits=bits, rank=r, k=kk, bits_per_weight=bits_per_weight(bits, m, n, r, group, symmetric))
    info["rel_error"] = float(np.linalg.norm(w - w_hat) / np.linalg.norm(w))
    return w_hat, info
