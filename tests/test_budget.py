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
