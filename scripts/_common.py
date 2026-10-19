"""스크립트 공통: 저장소 경로, 결과 JSON, 가중치 읽기(safetensors / 합성)."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"→ {path.relative_to(ROOT)}")


def load_weight(spec):
    """가중치 지정 → (이름, W(out × in) float64).

    'synthetic:<kind>:<m>x<n>:<seed>'            qer.synth.make 로 만든 합성 행렬
    '<file.safetensors>:<tensor name>'            HF 저장소에서 받아 둔 safetensors 파일의 텐서
                                                  (예: model.safetensors:model.layers.0.self_attn.q_proj.weight)
    """
    if spec.startswith("synthetic:"):
        from qer import synth
        _, kind, size, seed = spec.split(":")
        m, n = (int(v) for v in size.lower().split("x"))
        return spec, synth.make(m, n, kind, seed=int(seed))
    path, name = spec.rsplit(":", 1)
    w = read_safetensor(path, name)
    if w.ndim != 2:
        raise ValueError(f"{name}: 2차원이 아니다 {w.shape}")
    return f"{Path(path).name}:{name}", np.asarray(w, np.float64)


_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "BF16": None}


def safetensor_names(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return [k for k in header if k != "__metadata__"]


def read_safetensor(path, name):
    """safetensors 파일에서 텐서 하나를 torch 없이 읽는다(헤더 JSON + data_offsets).
    bfloat16은 numpy에 없으므로 상위 16비트를 float32로 옮겨 만든다."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        if name not in header:
            names = safetensor_names(path)
            raise KeyError(f"{name} 이 {path}에 없다. 있는 이름 {len(names)}개, 예: {names[:4]}")
        info = header[name]
        a, b = info["data_offsets"]
        fh.seek(8 + n + a)
        raw = fh.read(b - a)
    dtype = info["dtype"]
    if dtype not in _DTYPES:
        raise ValueError(f"{name}: 지원하지 않는 dtype {dtype}")
    if dtype == "BF16":
        u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
        arr = u.view(np.float32)
    else:
        arr = np.frombuffer(raw, _DTYPES[dtype])
    return arr.reshape(info["shape"])
