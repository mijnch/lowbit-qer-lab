import numpy as np
import pytest

from qer import synth
from qer.apply import compress_matrix
from qer.quant import rtn
from qer.split import reconstruct, surrogate_curve


@pytest.fixture
def w():
    return synth.make(64, 256, "outlier", seed=3)


def test_fp_is_identity(w):
    w_hat, info = compress_matrix(w, "fp")
    assert np.array_equal(w_hat, w) and info["rel_error"] == 0.0


def test_rtn_matches_quantizer(w):
    w_hat, info = compress_matrix(w, "rtn", bits=3, group=64)
    assert np.array_equal(w_hat, rtn(w, 3, 64))
    assert info["bits_per_weight"] == pytest.approx(3 + 19 / 64)


def test_qer_is_split_with_k0(w):
    a, ia = compress_matrix(w, "qer", bits=3, rank=8)
    b, _ = reconstruct(w, 3, 8, 0)
    assert np.allclose(a, b) and ia["k"] == 0 and ia["rank"] == 8


def test_split_surrogate_and_explicit_k(w):
    k_star = surrogate_curve(w, 3, 8)["k_star"]
    a, ia = compress_matrix(w, "split", bits=3, rank=8)
    b, _ = reconstruct(w, 3, 8, k_star)
    assert np.allclose(a, b) and ia["k"] == k_star
    c, ic = compress_matrix(w, "split", bits=3, rank=8, k=5)
    assert ic["k"] == 5 and np.allclose(c, reconstruct(w, 3, 8, 5)[0])
    d, idd = compress_matrix(w, "split", bits=3, rank=8, k="best")
    assert idd["rel_error"] <= ia["rel_error"] + 1e-12


def test_rank_from_budget(w):
    _, info = compress_matrix(w, "qer", bits=3, extra_bits=1.0)
    assert info["rank"] == int(1.0 * 64 * 256 // (16 * (64 + 256)))
