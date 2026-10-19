"""RTN 그룹 양자화기 — 오차 한계, 그룹 독립성, 멱등성, 비트 수에 따른 단조성."""
import numpy as np
import pytest

from qer.quant import rtn, rtn_error


@pytest.fixture
def w():
    return np.random.default_rng(0).standard_normal((16, 256))


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_error_within_half_step(w, bits, symmetric):
    q = rtn(w, bits, group=128, symmetric=symmetric)
    g = w.reshape(16, 2, 128)
    if symmetric:
        step = np.abs(g).max(axis=2, keepdims=True) / (2 ** (bits - 1) - 1)
    else:
        step = (g.max(axis=2, keepdims=True) - g.min(axis=2, keepdims=True)) / (2 ** bits - 1)
    err = np.abs((w - q).reshape(16, 2, 128))
    # 비대칭은 zero-point를 반올림하므로 격자가 최대 step/2 만큼 어긋날 수 있다 → 한계는 step
    assert np.all(err <= (step / 2 if symmetric else step) + 1e-12)


def test_asymmetric_uses_full_range_and_keeps_endpoints(w):
    q = rtn(w, 3, group=128)
    g_w, g_q = w.reshape(16, 2, 128), q.reshape(16, 2, 128)
    for i in range(16):
        for j in range(2):
            levels = np.unique(np.rint((g_q[i, j] - g_q[i, j].min()) / ((g_q[i, j].max() - g_q[i, j].min()) / 7)))
            assert len(levels) <= 8
            # 그룹의 최대·최소는 zero-point 반올림 오차(≤ step/2)만큼만 벗어난다
            step = (g_w[i, j].max() - g_w[i, j].min()) / 7
            assert abs(g_q[i, j].max() - g_w[i, j].max()) <= step / 2 + 1e-12
            assert abs(g_q[i, j].min() - g_w[i, j].min()) <= step / 2 + 1e-12


def test_groups_are_independent(w):
    q_full = rtn(w, 4, group=128)
    w2 = w.copy()
    w2[:, 128:] *= 100.0          # 둘째 그룹만 바꾸면
    q2 = rtn(w2, 4, group=128)
    assert np.array_equal(q_full[:, :128], q2[:, :128])   # 첫째 그룹 결과는 그대로


def test_idempotent(w):
    q = rtn(w, 4)
    assert np.allclose(rtn(q, 4), q, atol=1e-12)


def test_more_bits_less_error(w):
    errs = [np.linalg.norm(rtn_error(w, b)) for b in (2, 3, 4, 5, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_group_not_multiple_of_width():
    w = np.random.default_rng(1).standard_normal((4, 300))
    q = rtn(w, 4, group=128)
    assert q.shape == w.shape
    assert np.abs(w - q).max() <= (w.max() - w.min()) / 15 + 1e-12


def test_constant_group_is_exact():
    w = np.full((2, 128), 0.37)
    assert np.allclose(rtn(w, 3), w)
    assert np.allclose(rtn(w, 3, symmetric=True), w)
