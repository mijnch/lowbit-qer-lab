"""합성 가중치 행렬 — 실제 LLM 가중치를 받을 수 없는 환경에서 파이프라인을 돌려 보기 위한 것.
실제 가중치의 분포를 대변한다고 주장하지 않는다(README의 측정/미측정 표).

make(m, n, kind, seed). 모든 종류의 바탕은 iid 가우스 잡음(원소 표준편차 1/√n, 최대 특이값 ≈ 2).
  "gauss"    : 잡음만 — 저랭크 구조가 없다. 분할(k > 0)이 이득을 주지 않아야 한다.
  "lowrank"  : 잡음 + 랭크 rank0 성분(특이값이 잡음의 strength배에서 1배까지 선형 감소) —
               보존할 성분이 뚜렷한 쪽.
  "decay"    : 특이값이 거듭제곱으로 감소하는 행렬(σ_i ∝ i^(−alpha)) + 잡음 — 뚜렷한 틈 없이
               완만하게 줄어드는, 실제 가중치에 더 가까운 모양.
  "outlier"  : 잡음 + 입력 채널 n_outlier개의 크기를 outlier_scale배로 키움 — 그룹 RTN의 scale을
               그 채널이 지배한다.
"""
import numpy as np

NOISE_TOP_SV = 2.0   # m = n 일 때 잡음 행렬의 최대 특이값(Marchenko–Pastur 끝점 ≈ 2)


def make(m, n, kind="lowrank", seed=0, rank0=16, strength=5.0, alpha=0.7, n_outlier=8, outlier_scale=10.0):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((m, n)) / np.sqrt(n)
    if kind == "gauss":
        return noise
    if kind == "lowrank":
        s = NOISE_TOP_SV * np.linspace(strength, 1.0, rank0)
        return noise + _with_spectrum(rng, m, n, s)
    if kind == "decay":
        i = np.arange(1, min(m, n) + 1)
        s = NOISE_TOP_SV * strength * i ** (-alpha)
        return noise + _with_spectrum(rng, m, n, s)
    if kind == "outlier":
        cols = rng.choice(n, n_outlier, replace=False)
        noise[:, cols] *= outlier_scale
        return noise
    raise ValueError(kind)


def _with_spectrum(rng, m, n, s):
    """특이값이 정확히 s인 행렬 U diag(s) Vᵀ (U, V는 무작위 직교)."""
    p = len(s)
    u, _ = np.linalg.qr(rng.standard_normal((m, p)))
    v, _ = np.linalg.qr(rng.standard_normal((n, p)))
    return (u * s) @ v.T
