import pytest

from qer.budget import bits_per_weight, rank_for_extra_bits


def test_example_from_docstring():
    # 2048×2048, r=64, 16비트 인자 → +1.0비트
    assert bits_per_weight(3, 2048, 2048, r=64, group=128, symmetric=True) == pytest.approx(3 + 16 / 128 + 1.0)
    assert rank_for_extra_bits(1.0, 2048, 2048) == 64


def test_asymmetric_group_cost_includes_zero_point():
    sym = bits_per_weight(4, 512, 512, group=128, symmetric=True)
    asym = bits_per_weight(4, 512, 512, group=128, symmetric=False)
    assert asym - sym == pytest.approx(4 / 128)


def test_rank_budget_scales_with_matrix_size():
    assert rank_for_extra_bits(1.0, 576, 576) == 18
    assert rank_for_extra_bits(1.0, 576, 1536) == 26


def test_group_cost_counts_the_padded_last_group():
    # 576 = 4.5 × 128 → 행마다 그룹 5개(마지막은 0 으로 채움), 그룹마다 scale 16비트 + zero 3비트
    assert bits_per_weight(3, 576, 576, group=128) == pytest.approx(3 + 5 * 19 / 576)
    assert bits_per_weight(3, 576, 1536, group=128) == pytest.approx(3 + 12 * 19 / 1536)   # 나누어떨어지면 그대로


def test_rank_is_an_exact_floor():
    # 0.7·720·1440 / (16·2160) = 21 정확히 — 부동소수점 곱으로는 20.999… 이 되어 20 이 나왔다
    assert rank_for_extra_bits(0.7, 720, 1440) == 21
    assert rank_for_extra_bits(0.7, 1440, 2880) == 42
