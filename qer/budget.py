"""비트 예산 — "3비트 + 랭크 r"과 "4비트 RTN"을 같은 저장 비용끼리 비교하기 위한 계산.

가중치 한 개당 비트 = b + (그룹 scale·zero 비용) + (저랭크 인자 비용)
  그룹 비용   : 그룹 g마다 fp16 scale 1개(+ 비대칭이면 zero-point 1개, b비트로 침)
  저랭크 비용 : 랭크 r의 인자 L(m × r), R(r × n)을 16비트로 저장 → 16·r·(m + n) / (m·n)
예: 2048 × 2048, r = 64 → 16·64·4096 / 2048² = +1.0비트. 즉 "3비트 + r64" ≈ "4비트 RTN".
"""


def bits_per_weight(bits, m, n, r=0, group=128, symmetric=False, factor_bits=16, scale_bits=16):
    """m × n 가중치 행렬 하나의 가중치당 평균 비트."""
    per_group = scale_bits + (0 if symmetric else bits)
    group_cost = per_group / group
    lowrank_cost = factor_bits * r * (m + n) / (m * n) if r else 0.0
    return bits + group_cost + lowrank_cost


def rank_for_extra_bits(extra_bits, m, n, factor_bits=16):
    """가중치당 extra_bits만큼의 예산으로 살 수 있는 최대 랭크 r = ⌊extra·m·n / (factor_bits·(m+n))⌋."""
    return int(extra_bits * m * n // (factor_bits * (m + n)))
