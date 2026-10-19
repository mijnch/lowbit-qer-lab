"""T1 — 가중치 행렬 하나에서: RTN / QER / 보존-복원 분할의 오차 곡선 L(k), 대리지표 argmin 대 실제 argmin.

    python scripts/t1_layer_curve.py --weight synthetic:decay:1024x1024:0 --bits 3 4 --extra-bits 1.0
    python scripts/t1_layer_curve.py --weight model.safetensors:model.layers.0.self_attn.q_proj.weight

랭크 r은 --rank로 직접 주거나, --extra-bits(가중치당 추가 비트, 16비트 인자 기준)로 정한다.
같은 예산의 비교 상대는 "b비트 + r" 대 "(b + extra)비트 RTN"이다.

결과: results/t1_<이름>.json, results/figures/t1_<이름>.png
"""
import argparse
import re
import time

import numpy as np

from _common import FIGURES, load_weight, save_json
from qer.budget import bits_per_weight, rank_for_extra_bits
from qer.lowrank import rho_curve, singular_values
from qer.split import rtn_loss, split_loss_curve, surrogate_curve


def analyse(w, bits, r, group, symmetric):
    m, n = w.shape
    wnorm = float(np.linalg.norm(w))
    t = time.time()
    curve = split_loss_curve(w, bits, r, group, symmetric)
    sur = surrogate_curve(w, bits, r, group, symmetric)
    k_best = int(np.argmin(curve["loss"]))
    k_star = sur["k_star"]
    row = {
        "bits": bits, "rank": r, "group": group, "symmetric": symmetric,
        "bits_per_weight": bits_per_weight(bits, m, n, r, group, symmetric),
        "rel_error": {                                   # ‖W − Ŵ‖_F / ‖W‖_F
            "rtn": rtn_loss(w, bits, group, symmetric) / wnorm,
            "qer_k0": curve["loss"][0] / wnorm,
            "split_best": curve["loss"][k_best] / wnorm,
            "split_surrogate": curve["loss"][k_star] / wnorm,
            "split_k_eq_r": curve["loss"][r] / wnorm,
        },
        "k_best": k_best, "k_surrogate": k_star,
        "surrogate_regret": curve["loss"][k_star] / curve["loss"][k_best] - 1.0,
        "curve_k": curve["k"],
        "curve_rel_loss": [v / wnorm for v in curve["loss"]],
        "curve_rel_quant_error": [v / wnorm for v in curve["quant_error"]],
        "surrogate": sur["surrogate"],
        "seconds": round(time.time() - t, 1),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help="synthetic:<kind>:<m>x<n>:<seed> 또는 <file>:<tensor>")
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--extra-bits", type=float, default=1.0, help="--rank이 없을 때 랭크를 정하는 추가 비트 예산")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()

    name, w = load_weight(a.weight)
    m, n = w.shape
    r = a.rank if a.rank is not None else rank_for_extra_bits(a.extra_bits, m, n)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    s = singular_values(w)
    out = {"weight": name, "shape": [m, n], "rank": r,
           "spectrum": {"top8": s[:8].tolist(), "rho_curve_at": {str(p): float(rho_curve(s=s)[p]) for p in (1, 4, 16, 64, 256) if p < len(s)}},
           "rows": []}
    print(f"{name}  {m}×{n}  r={r}  ρ_r(W)={rho_curve(s=s)[r]:.3f}")
    for bits in a.bits:
        row = analyse(w, bits, r, a.group, a.symmetric)
        out["rows"].append(row)
        # 같은 예산의 상대: (bits + extra)비트 RTN
        hi = bits + int(round(a.extra_bits)) if a.rank is None else None
        if hi is not None and 2 <= hi <= 8:
            row["rtn_same_budget"] = {"bits": hi, "rel_error": rtn_loss(w, hi, a.group, a.symmetric) / np.linalg.norm(w),
                                      "bits_per_weight": bits_per_weight(hi, m, n, 0, a.group, a.symmetric)}
        e = row["rel_error"]
        print(f"  {bits}비트 r={r}: RTN {e['rtn']:.4f}  QER(k=0) {e['qer_k0']:.4f}  "
              f"분할 최적 k={row['k_best']} {e['split_best']:.4f}  대리 k*={row['k_surrogate']} {e['split_surrogate']:.4f}  "
              f"(regret {100 * row['surrogate_regret']:.2f}%)  {row['seconds']}s"
              + (f"  | {hi}비트 RTN {row['rtn_same_budget']['rel_error']:.4f}" if hi else ""))
    save_json(f"t1_{tag}.json", out)
    if not a.no_figure:
        figure(out, tag)


def figure(out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = out["rows"]
    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 3.8), squeeze=False)
    for ax, row in zip(axes[0], rows):
        k = row["curve_k"]
        ax.plot(k, row["curve_rel_loss"], marker=".", label="L(k)/‖W‖  (split, measured)")
        ax.plot(k, row["curve_rel_quant_error"], ls=":", color="gray", label="‖E_k‖/‖W‖  (before correction)")
        ax.axhline(row["rel_error"]["rtn"], color="C3", ls="--", label=f"RTN {row['bits']}-bit")
        if "rtn_same_budget" in row:
            ax.axhline(row["rtn_same_budget"]["rel_error"], color="C2", ls="--",
                       label=f"RTN {row['rtn_same_budget']['bits']}-bit (same budget)")
        ax.axvline(row["k_best"], color="C0", alpha=0.3)
        ax.axvline(row["k_surrogate"], color="C1", alpha=0.6, ls="-.", label=f"surrogate k*={row['k_surrogate']}")
        ax2 = ax.twinx()
        ax2.plot(k, row["surrogate"], color="C1", alpha=0.5, lw=1)
        ax2.set_ylabel("surrogate ρ_k(W)·ρ_(r−k)(E)", color="C1", fontsize=8)
        ax.set_xlabel("k (preserved rank)  —  r − k restored")
        ax.set_ylabel("relative Frobenius error")
        ax.set_title(f"{out['weight']}\n{row['bits']}-bit, r={row['rank']}, {row['bits_per_weight']:.2f} bits/weight", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = FIGURES / f"t1_{tag}.png"
    fig.savefig(path, dpi=140)
    print(f"→ {path.relative_to(FIGURES.parents[1])}")


if __name__ == "__main__":
    main()
