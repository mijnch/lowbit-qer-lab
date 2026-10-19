"""저랭크 근사 — SVD_p(A)와 상대 잔차 에너지 ρ_p(A).

Eckart–Young: 프로베니우스 노름에서 랭크 p 최적 근사는 상위 p개 특이성분이고,
  ‖A − SVD_p(A)‖_F² = Σ_{i>p} σ_i².
ρ_p(A) = ‖A − SVD_p(A)‖_F² / ‖A‖_F² = Σ_{i>p} σ_i² / Σ_i σ_i²  (p = 0이면 1, p ≥ rank면 0).
"""
import numpy as np


def svd_p(a, p, return_factors=False):
    """상위 p개 특이성분 근사. p = 0이면 0 행렬. return_factors면 (L, R)도 준다(A_p = L @ R)."""
    a = np.asarray(a, np.float64)
    if p <= 0:
        zero = np.zeros_like(a)
        if return_factors:
            return zero, (np.zeros((a.shape[0], 0)), np.zeros((0, a.shape[1])))
        return zero
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    p = min(p, len(s))
    left = u[:, :p] * s[:p]
    right = vt[:p]
    approx = left @ right
    return (approx, (left, right)) if return_factors else approx


def singular_values(a):
    return np.linalg.svd(np.asarray(a, np.float64), compute_uv=False)


def rho_curve(a=None, s=None):
    """ρ_p를 p = 0..rank 전부 — rho_curve(a)[p] = ρ_p(a). 특이값 s를 주면 그것으로 계산."""
    if s is None:
        s = singular_values(a)
    energy = s ** 2
    total = energy.sum()
    if total == 0:
        return np.zeros(len(s) + 1)
    tail = total - np.concatenate([[0.0], np.cumsum(energy)])
    return np.maximum(tail, 0.0) / total


def rho(a, p):
    """ρ_p(A)."""
    curve = rho_curve(a)
    return float(curve[min(max(p, 0), len(curve) - 1)])


def eckart_young_residual(a, p):
    """‖A − SVD_p(A)‖_F² 를 직접 계산한 값과 Σ_{i>p} σ_i² — 테스트와 검산용으로 둘 다 돌려준다."""
    a = np.asarray(a, np.float64)
    direct = float(np.sum((a - svd_p(a, p)) ** 2))
    s = singular_values(a)
    from_sv = float(np.sum(s[p:] ** 2))
    return direct, from_sv
