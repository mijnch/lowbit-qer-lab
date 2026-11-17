"""Eckart–Young와 ρ_p — 정리가 말하는 값과 직접 계산한 값이 같아야 한다."""
import numpy as np
import pytest

from qer.lowrank import eckart_young_residual, rho, rho_curve, singular_values, svd_p


@pytest.mark.parametrize("shape", [(32, 48), (48, 32), (40, 40)])
@pytest.mark.parametrize("p", [0, 1, 5, 31])
def test_eckart_young_residual_equals_tail_singular_energy(shape, p):
    a = np.random.default_rng(1).standard_normal(shape)
    direct, from_sv = eckart_young_residual(a, p)
    assert direct == pytest.approx(from_sv, rel=1e-10)


def test_rank_p_approx_is_best_among_random_rank_p_matrices():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((30, 20))
    best = np.linalg.norm(a - svd_p(a, 3))
    for _ in range(20):
        other = rng.standard_normal((30, 3)) @ rng.standard_normal((3, 20))
        # 임의의 랭크 3 행렬보다 나쁠 수 없다 (최소제곱으로 맞춘 것도 포함)
        coef, *_ = np.linalg.lstsq(other.reshape(-1, 1), a.reshape(-1), rcond=None)
        assert best <= np.linalg.norm(a - coef[0] * other) + 1e-12


def test_rho_endpoints_and_monotone():
    a = np.random.default_rng(3).standard_normal((25, 35))
    curve = rho_curve(a)
    assert curve[0] == pytest.approx(1.0)
    assert curve[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(curve) <= 1e-12)
    assert rho(a, 4) == pytest.approx(np.sum(singular_values(a)[4:] ** 2) / np.sum(a ** 2))


def test_exact_lowrank_matrix_has_zero_rho_at_its_rank():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((40, 6)) @ rng.standard_normal((6, 50))
    assert rho(a, 6) == pytest.approx(0.0, abs=1e-10)
    assert rho(a, 5) > 1e-3


def test_svd_p_factors_multiply_back():
    a = np.random.default_rng(5).standard_normal((12, 9))
    approx, (left, right) = svd_p(a, 4, return_factors=True)
    assert left.shape == (12, 4) and right.shape == (4, 9)
    assert np.allclose(left @ right, approx)
    assert np.linalg.matrix_rank(approx) == 4
