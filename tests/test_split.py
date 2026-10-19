"""보존-복원 분할 — k = 0은 QER, 양 끝값, 곡선과 직접 재구성의 일치, 대리지표의 범위."""
import numpy as np
import pytest

from qer import synth
from qer.lowrank import svd_p
from qer.quant import rtn
from qer.split import reconstruct, rtn_loss, split_loss_curve, surrogate_curve

R = 8


@pytest.fixture(params=["gauss", "lowrank", "outlier"])
def w(request):
    return synth.make(64, 256, request.param, seed=7)


def test_k0_equals_qer(w):
    q = rtn(w, 3)
    qer = q + svd_p(w - q, R)
    w_hat, loss = reconstruct(w, 3, R, k=0)
    assert np.allclose(w_hat, qer)
    assert loss == pytest.approx(np.linalg.norm(w - qer))


def test_curve_matches_direct_reconstruction(w):
    curve = split_loss_curve(w, 3, R)
    assert curve["k"] == list(range(R + 1))
    for k in (0, 1, R // 2, R):
        _, loss = reconstruct(w, 3, R, k)
        assert curve["loss"][k] == pytest.approx(loss, rel=1e-9)


def test_lowrank_correction_never_hurts(w):
    # 어떤 k에서도 (P_k + Q_k + C_k)는 같은 비트의 RTN(저랭크 없음)보다 나쁘지 않아야 한다 … 는
    # 보장이 없다(P_k를 뗀 뒤의 양자화 간격이 달라지므로). 보장되는 것은 k = 0: QER ≤ RTN.
    curve = split_loss_curve(w, 3, R)
    assert curve["loss"][0] <= rtn_loss(w, 3) + 1e-12


def test_k_equals_r_means_no_error_correction(w):
    # k = r이면 C_k = SVD_0 = 0이므로 손실은 양자화 오차 ‖E_r‖_F 그대로
    curve = split_loss_curve(w, 3, R)
    assert curve["loss"][R] == pytest.approx(curve["quant_error"][R], rel=1e-9)


def test_surrogate_curve_shape_and_range(w):
    s = surrogate_curve(w, 3, R)
    assert s["k"] == list(range(R + 1))
    assert all(0.0 <= v <= 1.0 for v in s["surrogate"])
    assert 0 <= s["k_star"] <= R


def test_parts_add_up(w):
    w_hat, loss, (p, q, c) = reconstruct(w, 4, R, k=3, return_parts=True)
    assert np.allclose(p + q + c, w_hat)
    assert np.linalg.matrix_rank(p) == 3 and np.linalg.matrix_rank(c) <= R - 3
