"""T1 요약 (실제 모델) — safetensors 파일의 여러 층·행렬에서 RTN / QER / 분할(최적·대리)과 같은 예산 RTN.

    python scripts/t1_model_summary.py --file models/SmolLM2-135M/model.safetensors --tag SmolLM2-135M \
        --layers 0 7 15 22 29 --bits 3 4 --extra-bits 1.0
    python scripts/t1_model_summary.py --file models/TinyLlama-1.1B-3T/model.safetensors --tag TinyLlama-1.1B \
        --layers 0 11 --k-step 4

결과: results/t1_<tag>.json (행렬마다 한 줄) + 행렬 종류별 평균 표(stdout).
"""
import argparse
import time

import numpy as np

from _common import read_safetensor, safetensor_names, save_json
from qer.budget import rank_for_extra_bits
from qer.lowrank import rho_curve, singular_values
from qer.split import rtn_loss
from t1_layer_curve import analyse

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--tensors", nargs="+", default=list(PROJ))
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--extra-bits", type=float, default=1.0)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--k-step", type=int, default=1)
    a = ap.parse_args()

    names = safetensor_names(a.file)
    out = {"file": a.file, "tag": a.tag, "layers": a.layers, "extra_bits": a.extra_bits, "group": a.group,
           "k_step": a.k_step, "rows": []}
    for layer in a.layers:
        for proj in a.tensors:
            cands = [n for n in names if f".layers.{layer}." in n and n.endswith(f"{proj}.weight")]
            if len(cands) != 1:
                print(f"층 {layer} {proj}: 이름 {len(cands)}개 — 건너뜀")
                continue
            w = read_safetensor(a.file, cands[0]).astype(np.float64)
            m, n = w.shape
            r = rank_for_extra_bits(a.extra_bits, m, n)
            s = singular_values(w)
            base = {"name": cands[0], "layer": layer, "proj": proj, "shape": [m, n], "rank": r,
                    "rho_r": float(rho_curve(s=s)[r]),
                    "col_absmax_max_over_median": float(np.abs(w).max(0).max() / np.median(np.abs(w).max(0))),
                    "row_absmax_max_over_median": float(np.abs(w).max(1).max() / np.median(np.abs(w).max(1)))}
            for bits in a.bits:
                t = time.time()
                row = analyse(w, bits, r, a.group, False, a.k_step)
                hi = bits + int(round(a.extra_bits))
                row["rtn_same_budget_rel_error"] = rtn_loss(w, hi, a.group) / np.linalg.norm(w) if hi <= 8 else None
                for key in ("curve_rel_loss", "curve_rel_quant_error", "surrogate", "curve_k"):
                    row.pop(key)
                out["rows"].append({**base, **row})
                e = row["rel_error"]
                print(f"L{layer:<2d} {proj:9s} {m}×{n} r={r:<3d} ρ_r={base['rho_r']:.3f} {bits}비트: "
                      f"RTN {e['rtn']:.4f} QER {e['qer_k0']:.4f} 분할 {e['split_best']:.4f}(k={row['k_best']}) "
                      f"대리 {e['split_surrogate']:.4f}(k*={row['k_surrogate']}) regret {100 * row['surrogate_regret']:.2f}% "
                      f"| {hi}비트 RTN {row['rtn_same_budget_rel_error']:.4f}  {time.time() - t:.0f}s", flush=True)
    save_json(f"t1_{a.tag}.json", out)
    table(out)


def table(out):
    """행렬 종류 × 비트로 평균한 표(README용)."""
    rows = out["rows"]
    print("\n| 행렬 | 비트 | r | ρ_r(W) | RTN | QER (k=0) | 분할 최적 | 분할 대리 k* | 대리 후회 | 같은 예산 RTN |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for proj in dict.fromkeys(r["proj"] for r in rows):
        for bits in dict.fromkeys(r["bits"] for r in rows):
            sel = [r for r in rows if r["proj"] == proj and r["bits"] == bits]
            if not sel:
                continue
            mean = lambda key: np.mean([r["rel_error"][key] for r in sel])  # noqa: E731
            ks = ",".join(str(r["k_best"]) for r in sel)
            kstars = ",".join(str(r["k_surrogate"]) for r in sel)
            print(f"| {proj} {sel[0]['shape'][0]}×{sel[0]['shape'][1]} | {bits} | {sel[0]['rank']} | "
                  f"{np.mean([r['rho_r'] for r in sel]):.3f} | {mean('rtn'):.4f} | {mean('qer_k0'):.4f} | "
                  f"{mean('split_best'):.4f} (k={ks}) | {mean('split_surrogate'):.4f} (k*={kstars}) | "
                  f"≤{100 * max(r['surrogate_regret'] for r in sel):.2f}% | "
                  f"{np.mean([r['rtn_same_budget_rel_error'] for r in sel]):.4f} |")


if __name__ == "__main__":
    main()
