"""보존-복원 랭크 분할 — 랭크 예산 r을 "먼저 보존할 k" + "양자화 뒤 복원할 r − k"로 나눈다.

  P_k = SVD_k(W)                        상위 k개 성분을 양자화 전에 떼어 둔다
  Q_k = Q_b(W − P_k)                    나머지를 양자화
  E_k = W − P_k − Q_k                   양자화 오차
  C_k = SVD_(r−k)(E_k)                  오차를 저랭크로 복원
  Ŵ(k) = P_k + Q_k + C_k,  L(k) = ‖W − Ŵ(k)‖_F = ‖E_k − C_k‖_F

k = 0이면 P_0 = 0이므로 QER(Ŵ = Q_b(W) + SVD_r(W − Q_b(W)))과 같다.
저장 비용은 k와 무관하다: 저랭크 인자는 합쳐서 랭크 r(P_k와 C_k를 이어 붙임), 정수 행렬은 하나.

대리지표(surrogate): 모든 k에서 E_k를 다시 양자화·SVD하지 않고
  k* = argmin_k ρ_k(W) · ρ_(r−k)(E_probe),   E_probe = W − Q_b(W)  (k = 0의 오차를 탐침으로)
로 고른다. ‖E_k‖² ≈ c·‖W − P_k‖² = c·ρ_k(W)‖W‖² 이고 ρ_(r−k)(E_k) ≈ ρ_(r−k)(E_probe)라는
근사다. 탐침의 선택은 이 저장소의 단순화이며 논문의 정의와 같다고 보장하지 않는다(docs/notes.md).
"""
import numpy as np

from .lowrank import rho_curve, singular_values, svd_p
from .quant import rtn


def reconstruct(w, bits, r, k, group=128, symmetric=False, return_parts=False):
    """Ŵ(k)와 L(k). return_parts면 (P_k, Q_k, C_k)도 준다."""
    w = np.asarray(w, np.float64)
    if not 0 <= k <= r:
        raise ValueError("0 ≤ k ≤ r")
    p_k = svd_p(w, k)
    q_k = rtn(w - p_k, bits, group, symmetric)
    e_k = w - p_k - q_k
    c_k = svd_p(e_k, r - k)
    w_hat = p_k + q_k + c_k
    loss = float(np.linalg.norm(w - w_hat))
    if return_parts:
        return w_hat, loss, (p_k, q_k, c_k)
    return w_hat, loss


def split_loss_curve(w, bits, r, group=128, symmetric=False, ks=None):
    """k = 0..r(또는 ks)에서 L(k). 특이값만으로 ‖E_k − SVD_(r−k)(E_k)‖_F² = Σ_{i>r−k} σ_i(E_k)² 를 쓴다.

    돌려주는 것: {"k": [...], "loss": [...], "quant_error": [‖E_k‖_F ...]}"""
    w = np.asarray(w, np.float64)
    ks = list(range(r + 1)) if ks is None else list(ks)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    losses, qerr = [], []
    for k in ks:
        p_k = (u[:, :k] * s[:k]) @ vt[:k] if k > 0 else 0.0
        e_k = (w - p_k) - rtn(w - p_k, bits, group, symmetric)
        se = singular_values(e_k)
        losses.append(float(np.sqrt(np.sum(se[r - k:] ** 2))))
        qerr.append(float(np.sqrt(np.sum(se ** 2))))
    return {"k": ks, "loss": losses, "quant_error": qerr}


def surrogate_curve(w, bits, r, group=128, symmetric=False, probe=None):
    """대리지표 ρ_k(W)·ρ_(r−k)(E_probe), k = 0..r. probe를 주지 않으면 W − Q_b(W)."""
    w = np.asarray(w, np.float64)
    if probe is None:
        probe = w - rtn(w, bits, group, symmetric)
    rho_w = rho_curve(w)
    rho_e = rho_curve(probe)
    ks = np.arange(r + 1)
    vals = rho_w[np.minimum(ks, len(rho_w) - 1)] * rho_e[np.minimum(r - ks, len(rho_e) - 1)]
    return {"k": ks.tolist(), "surrogate": vals.tolist(), "k_star": int(np.argmin(vals))}


def rtn_loss(w, bits, group=128, symmetric=False):
    """저랭크 없이 RTN만: ‖W − Q_b(W)‖_F."""
    w = np.asarray(w, np.float64)
    return float(np.linalg.norm(w - rtn(w, bits, group, symmetric)))
