"""비트 예산 — "3비트 + 랭크 r"과 "4비트 RTN"을 같은 저장 비용끼리 비교하기 위한 계산.

가중치 한 개당 비트 = b + (그룹 scale·zero 비용) + (저랭크 인자 비용)
  그룹 비용   : 그룹 g마다 fp16 scale 1개(+ 비대칭이면 zero-point 1개, b비트로 침)
  저랭크 비용 : 랭크 r의 인자 L(m × r), R(r × n)을 16비트로 저장 → 16·r·(m + n) / (m·n)
예: 2048 × 2048, r = 64 → 16·64·4096 / 2048² = +1.0비트. 즉 "3비트 + r64" ≈ "4비트 RTN".
"""
import math
from fractions import Fraction


def bits_per_weight(bits, m, n, r=0, group=128, symmetric=False, factor_bits=16, scale_bits=16):
    """m × n 가중치 행렬 하나의 가중치당 평균 비트.

    그룹 비용은 행마다 실제로 저장하는 그룹 수 ⌈n / group⌉ 로 센다 — n 이 group 의 배수가 아니면
    마지막 그룹은 0 으로 채워 따로 scale·zero 를 둔다(quant._groups). 예: n = 576 이면 그룹 5개.
    """
    per_group = scale_bits + (0 if symmetric else bits)
    group_cost = per_group * math.ceil(n / group) / n
    lowrank_cost = factor_bits * r * (m + n) / (m * n) if r else 0.0
    return bits + group_cost + lowrank_cost


def rank_for_extra_bits(extra_bits, m, n, factor_bits=16):
    """가중치당 extra_bits만큼의 예산으로 살 수 있는 최대 랭크 r = ⌊extra·m·n / (factor_bits·(m+n))⌋.

    분수로 정확히 센다 — 0.7·720·1440 처럼 정확히 나누어떨어지는 값이 부동소수점에서 한 끗 모자라
    랭크가 하나 적게 나오던 것을 막는다.
    """
    return int(Fraction(str(extra_bits)) * m * n // (factor_bits * (m + n)))
