"""T2 — 작은 LLM의 모든 디코더 선형층에 RTN / QER / 분할을 적용하고 WikiText-2 perplexity를 잰다.

    pip install -r requirements-llm.txt
    python scripts/t2_perplexity.py --model HuggingFaceTB/SmolLM2-135M --methods fp rtn qer split \
        --bits 3 4 --extra-bits 1.0 --seq-len 1024 --max-windows 40

같은 예산 비교: "b비트 + 랭크(extra-bits)" 대 "(b + extra)비트 RTN"을 같은 표에 둔다.
perplexity = exp(토큰당 평균 음의 로그우도), 테스트 분할을 이어 붙여 seq-len 길이의 겹치지 않는 창으로 잰다.
CPU에서는 --max-windows 로 창 수를 줄인다(그 경우 전체 분할이 아니라 앞부분만 잰 것이고, JSON에 기록된다).

※ 이 스크립트는 저장소를 만든 환경(모델 허브 접근 불가, torch 미설치)에서 실행하지 못했다.
  numpy 부분(qer.apply.compress_matrix)은 테스트했고, torch·transformers·datasets 연결 부분은
  실행 확인이 필요하다. README의 측정/미측정 표 참고.
"""
import argparse
import math
import re
import time

import numpy as np

from _common import save_json
from qer.apply import compress_matrix


def load_model(name, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype))
    model.eval()
    return tok, model


def load_windows(tok, dataset, config, split, seq_len, max_windows):
    import torch
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split)
    text = "\n\n".join(t for t in ds["text"] if t)
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = (len(ids) // seq_len) * seq_len
    windows = ids[:n].view(-1, seq_len)
    if max_windows is not None:
        windows = windows[:max_windows]
    return torch.as_tensor(windows), len(ids)


def perplexity(model, windows):
    import torch
    nll, count = 0.0, 0
    with torch.no_grad():
        for w in windows:
            w = w.unsqueeze(0)
            logits = model(input_ids=w).logits[:, :-1].float()
            target = w[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum")
            nll += float(loss)
            count += target.numel()
    return math.exp(nll / count), nll, count


def target_linears(model, pattern):
    import torch
    rx = re.compile(pattern)
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and rx.search(name)]


def apply(linears, originals, method, **kw):
    """선형층마다 compress_matrix. 되돌리기용 원본은 originals에 있다. → 층별 info 목록"""
    import torch
    infos = []
    for name, mod in linears:
        w = originals[name].numpy().astype(np.float64)
        w_hat, info = compress_matrix(w, method, **kw)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(w_hat).to(mod.weight.dtype))
        info["name"] = name
        infos.append(info)
    return infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--config", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=["fp", "rtn", "qer", "split"])
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--extra-bits", type=float, default=1.0)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--k", default="surrogate", help="split의 k: surrogate | best | 정수")
    ap.add_argument("--layers", default=r"\.layers\.\d+\.", help="압축할 nn.Linear 이름의 정규식(임베딩·lm_head 제외)")
    a = ap.parse_args()

    tok, model = load_model(a.model, a.dtype)
    windows, total_tokens = load_windows(tok, a.dataset, a.config, a.split, a.seq_len, a.max_windows)
    linears = target_linears(model, a.layers)
    originals = {name: mod.weight.detach().cpu().clone() for name, mod in linears}
    numel = sum(v.numel() for v in originals.values())
    print(f"{a.model}: 선형층 {len(linears)}개, 가중치 {numel:,}개, 창 {len(windows)}×{a.seq_len} (분할 전체 토큰 {total_tokens:,})")

    out = {"model": a.model, "dataset": f"{a.dataset}/{a.config}/{a.split}", "seq_len": a.seq_len,
           "windows": int(len(windows)), "tokens_in_split": int(total_tokens), "layers": len(linears),
           "group": a.group, "symmetric": a.symmetric, "extra_bits": a.extra_bits, "k": a.k, "rows": []}
    k = a.k if a.k in ("surrogate", "best") else int(a.k)
    configs = [("fp", None)] if "fp" in a.methods else []
    for bits in a.bits:
        for method in a.methods:
            if method == "fp":
                continue
            configs.append((method, bits))
        hi = bits + int(round(a.extra_bits))
        if "rtn" in a.methods and hi <= 8 and ("rtn", hi) not in configs:
            configs.append(("rtn", hi))                      # 같은 예산의 상대
    for method, bits in configs:
        t = time.time()
        if method == "fp":
            infos = apply(linears, originals, "fp")
        else:
            infos = apply(linears, originals, method, bits=bits, group=a.group, symmetric=a.symmetric,
                          extra_bits=a.extra_bits, k=k)
        ppl, nll, count = perplexity(model, windows)
        bpw = sum(i["bits_per_weight"] * np.prod(i["shape"]) for i in infos) / numel
        rel = float(np.sqrt(sum(i["rel_error"] ** 2 * np.prod(i["shape"]) for i in infos) / numel))
        row = {"method": method, "bits": bits, "perplexity": ppl, "nll_sum": nll, "tokens": count,
               "bits_per_weight_mean": float(bpw), "rel_error_rms": rel,
               "k_values": [i.get("k") for i in infos] if method == "split" else None,
               "seconds": round(time.time() - t, 1)}
        out["rows"].append(row)
        print(f"{method:6s} {str(bits):>4s}비트  ppl {ppl:8.3f}  {bpw:.2f} bits/weight  rel err {rel:.4f}  {row['seconds']}s")
        # 되돌리기
        apply(linears, originals, "fp")
    fp = next((r["perplexity"] for r in out["rows"] if r["method"] == "fp"), None)
    for bits in a.bits:
        r_ = {r["method"]: r["perplexity"] for r in out["rows"] if r["bits"] == bits}
        if fp is not None and "rtn" in r_ and "qer" in r_ and not (fp <= r_["qer"] <= r_["rtn"]):
            print(f"※ {bits}비트: 기대한 순서(FP ≤ QER ≤ RTN)가 아니다 — 확인 필요")
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", a.model)
    save_json(f"t2_{tag}.json", out)


if __name__ == "__main__":
    main()
