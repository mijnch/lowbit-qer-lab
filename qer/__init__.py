"""lowbit-qer-lab — 저비트 가중치 양자화 오차의 저랭크 복원(Q + LR)을 numpy로 단순화해 확인한다.

W ≈ Q + L R 에서
  QER   : Q = Q_b(W),              LR = SVD_r(W − Q)                      (k = 0)
  분할  : P_k = SVD_k(W), Q_k = Q_b(W − P_k), LR = P_k + SVD_(r−k)(W − P_k − Q_k)
"""
from .quant import rtn, rtn_error                      # noqa: F401
from .lowrank import svd_p, rho, rho_curve, eckart_young_residual   # noqa: F401
from .split import split_loss_curve, surrogate_curve, reconstruct   # noqa: F401
from .budget import bits_per_weight, rank_for_extra_bits            # noqa: F401
