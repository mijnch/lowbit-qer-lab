"""T2 보조 — 층 하나의 **출력** 오차 ‖X(W − Ŵ)ᵀ‖_F / ‖XWᵀ‖_F 를 실제 활성값 X로 잰다 (S ≠ I의 축소판).

    python scripts/t2_output_error.py --model models/SmolLM2-135M --layers 0 15 29 --windows 4

가중치 프로베니우스 오차(T1)와 perplexity(T2)의 순위가 다를 때, 어느 쪽이 층 출력 오차와 같은 순위인지
본다. X는 WikiText-2 테스트 분할 앞 --windows 창을 fp 모델에 흘려 각 선형층 입력에서 모은다.
결과: results/t2_output_error_<model>.json
"""
import argparse
import re
from pathlib import Path

import numpy as np

from _common import save_json, portable_path
from qer.apply import compress_matrix
from t2_perplexity import environment, load_model, load_windows, target_linears


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 15, 29])
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--extra-bits", type=float, default=1.0)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--methods", nargs="+", default=["rtn", "qer", "split"])
    a = ap.parse_args()

    import torch
    tok, model = load_model(a.model, "float32")
    windows, _ = load_windows(tok, "Salesforce/wikitext", "wikitext-2-raw-v1", "test", a.seq_len, a.windows)
    pattern = "|".join(rf"\.layers\.{i}\." for i in a.layers)
    linears = target_linears(model, pattern)
    inputs = {name: [] for name, _ in linears}
    hooks = [mod.register_forward_hook(lambda m, inp, out, name=name: inputs[name].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).numpy()))
             for name, mod in linears]
    with torch.no_grad():
        for w in windows:
            model(input_ids=w.unsqueeze(0))
    for h in hooks:
        h.remove()

    out = {"model": portable_path(a.model), "layers": a.layers, "windows": int(len(windows)), "seq_len": a.seq_len,
           "env": environment(), "rows": []}
    print("| 층 | 행렬 | 비트 | 방법 | 가중치 오차 | 출력 오차 | k |")
    for name, mod in linears:
        x = np.concatenate(inputs[name]).astype(np.float64)        # (토큰 수, in)
        w = mod.weight.detach().numpy().astype(np.float64)        # (out, in)
        y = x @ w.T
        ynorm = np.linalg.norm(y)
        for bits in a.bits:
            for method in a.methods:
                w_hat, info = compress_matrix(w, method, bits=bits, group=a.group, extra_bits=a.extra_bits)
                row = {"name": name, "bits": bits, "method": method, "k": info.get("k"), "rank": info.get("rank"),
                       "weight_rel_error": info["rel_error"],
                       "output_rel_error": float(np.linalg.norm(x @ (w - w_hat).T) / ynorm)}
                out["rows"].append(row)
                print(f"| {name} | {bits} | {method} | {row['weight_rel_error']:.4f} | {row['output_rel_error']:.4f} | {row['k']} |", flush=True)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(a.model).name if Path(a.model).is_dir() else a.model)
    save_json(f"t2_output_error_{tag}.json", out)
    summary(out)


def summary(out):
    """비트·방법별로 (출력 오차 / 가중치 오차)와, 층마다 출력 오차 순위에서 QER이 분할보다 나은 횟수."""
    rows = out["rows"]
    for bits in sorted({r["bits"] for r in rows}):
        for method in ("rtn", "qer", "split"):
            sel = [r for r in rows if r["bits"] == bits and r["method"] == method]
            if sel:
                print(f"{bits}비트 {method:5s} 가중치 오차 평균 {np.mean([r['weight_rel_error'] for r in sel]):.4f}  "
                      f"출력 오차 평균 {np.mean([r['output_rel_error'] for r in sel]):.4f}")
        by = {(r["name"], r["method"]): r["output_rel_error"] for r in rows if r["bits"] == bits}
        pairs = [(by[n, "qer"], by[n, "split"]) for n in sorted({r["name"] for r in rows})
                 if (n, "qer") in by and (n, "split") in by]
        if not pairs:            # --methods 에 qer·split 이 함께 없으면 비교하지 않는다
            continue
        lower = sum(1 for q, s in pairs if s < q)
        same = sum(1 for q, s in pairs if s == q)      # k* = 0 이면 분할 = QER
        print(f"{bits}비트: 출력 오차에서 분할 < QER 인 행렬 {lower}/{len(pairs)} "
              f"(같음 {same}, QER < 분할 {len(pairs) - lower - same})")


if __name__ == "__main__":
    main()
