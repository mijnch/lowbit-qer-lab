"""T1 요약 — 합성 행렬 4종 × 시드 × 비트에서 RTN / QER / 분할(최적·대리) 상대 오차와 같은 예산의 RTN.

    python scripts/t1_summary.py --size 1024 --seeds 0 1 2 --bits 3 4

결과: results/t1_synthetic_summary.json 과 표(stdout, README에 옮겨 적음).
"""
import argparse

import numpy as np

from _common import save_json
from qer import synth
from qer.budget import rank_for_extra_bits
from t1_layer_curve import analyse

KINDS = ("gauss", "decay", "lowrank", "outlier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--extra-bits", type=float, default=1.0)
    ap.add_argument("--group", type=int, default=128)
    a = ap.parse_args()
    m = n = a.size
    r = rank_for_extra_bits(a.extra_bits, m, n)
    out = {"size": [m, n], "rank": r, "extra_bits": a.extra_bits, "group": a.group, "seeds": a.seeds, "rows": []}
    print("| 행렬 | 비트 | RTN | QER (k=0) | 분할 최적 k | 분할 대리 k* | 대리 regret | 같은 예산 RTN |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for kind in KINDS:
        for bits in a.bits:
            rows = []
            for seed in a.seeds:
                w = synth.make(m, n, kind, seed=seed)
                row = analyse(w, bits, r, a.group, False)
                from qer.split import rtn_loss
                row["rtn_same_budget_rel_error"] = rtn_loss(w, bits + int(round(a.extra_bits)), a.group) / np.linalg.norm(w)
                rows.append(row)
            agg = {"kind": kind, "bits": bits, "n_seeds": len(rows)}
            for key in ("rtn", "qer_k0", "split_best", "split_surrogate"):
                vals = [x["rel_error"][key] for x in rows]
                agg[key] = {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals))}
            agg["rtn_same_budget"] = {"mean": float(np.mean([x["rtn_same_budget_rel_error"] for x in rows]))}
            agg["k_best"] = [x["k_best"] for x in rows]
            agg["k_surrogate"] = [x["k_surrogate"] for x in rows]
            agg["surrogate_regret_max"] = float(max(x["surrogate_regret"] for x in rows))
            out["rows"].append(agg)
            print(f"| {kind} | {bits} | {agg['rtn']['mean']:.4f} | {agg['qer_k0']['mean']:.4f} | "
                  f"{agg['split_best']['mean']:.4f} (k={','.join(map(str, agg['k_best']))}) | "
                  f"{agg['split_surrogate']['mean']:.4f} (k*={','.join(map(str, agg['k_surrogate']))}) | "
                  f"≤{100 * agg['surrogate_regret_max']:.2f}% | {agg['rtn_same_budget']['mean']:.4f} ({bits + int(round(a.extra_bits))}비트) |")
    save_json("t1_synthetic_summary.json", out)


if __name__ == "__main__":
    main()
