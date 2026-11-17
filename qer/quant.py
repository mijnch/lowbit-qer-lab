"""RTN(round-to-nearest) 그룹 양자화기 — 가중치 행렬 W(out × in)의 각 행을 입력 축으로 g개씩 묶어
그룹마다 scale(과 zero-point)을 둔다. 양자화 → 역양자화한 float 행렬을 돌려준다(fake quantization).

비대칭(기본): q ∈ [0, 2^b − 1], scale = (max − min)/(2^b − 1), zero = round(−min/scale).
  min·max 는 0 을 포함하도록 넓힌다(min ≤ 0 ≤ max) — 그래야 zero-point 가 [0, 2^b − 1] 안에 들어
  b 비트로 저장되고(budget.py 의 가정), 0 을 포함하지 않는 그룹이 한 단계로 뭉개지지 않는다.
대칭(narrow): q ∈ [−(2^(b−1) − 1), 2^(b−1) − 1], scale = max|w| / (2^(b−1) − 1)

반올림은 np.rint(동률은 짝수 쪽, torch.round와 같음). 논문들이 쓰는 MXINT(공유 지수)와는 다르다.
"""
import numpy as np


def _groups(w, group):
    """(out, in) → (out, n_groups, group). in이 group의 배수가 아니면 마지막 그룹은 0으로 채운다."""
    out, n = w.shape
    pad = (-n) % group
    if pad:
        w = np.concatenate([w, np.zeros((out, pad), w.dtype)], axis=1)
    return w.reshape(out, -1, group), pad


def rtn(w, bits, group=128, symmetric=False):
    """W(out × in, float)를 bits비트 RTN 그룹 양자화했다가 되돌린 float64 행렬."""
    w = np.asarray(w, np.float64)
    if w.ndim != 2:
        raise ValueError("2차원 행렬(out × in)만 다룬다")
    if not 2 <= bits <= 8:
        raise ValueError("bits는 2~8")
    g, pad = _groups(w, group)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = np.abs(g).max(axis=2, keepdims=True) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        q = np.clip(np.rint(g / scale), -qmax, qmax)
        deq = q * scale
    else:
        qmax = 2 ** bits - 1
        lo = np.minimum(g.min(axis=2, keepdims=True), 0.0)   # 0 을 포함하지 않는 그룹(전부 양수·음수)이
        hi = np.maximum(g.max(axis=2, keepdims=True), 0.0)   # 한 단계로 뭉개지던 것을 막는다
        scale = (hi - lo) / qmax
        const = scale == 0                       # 그룹의 값이 전부 같으면 그대로 둔다
        scale = np.where(const, 1.0, scale)
        zero = np.clip(np.rint(-lo / scale), 0, qmax)
        q = np.clip(np.rint(g / scale) + zero, 0, qmax)
        deq = np.where(const, g, (q - zero) * scale)
    deq = deq.reshape(w.shape[0], -1)
    return deq[:, :w.shape[1]] if pad else deq


def rtn_error(w, bits, group=128, symmetric=False):
    """E = W − Q_b(W)."""
    w = np.asarray(w, np.float64)
    return w - rtn(w, bits, group, symmetric)
