"""scripts/_common.read_safetensor — 헤더·오프셋으로 직접 읽은 값이 쓴 값과 같다(BF16 포함)."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import read_safetensor, safetensor_names  # noqa: E402


def _write(path, tensors):
    """safetensors 형식: <u64 헤더 길이><JSON 헤더><바이트>. tensors: {이름: (dtype 문자열, 배열, 바이트)}"""
    header, blobs, off = {}, [], 0
    for name, (dtype, arr, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_reads_f32_f16_bf16(tmp_path):
    rng = np.random.default_rng(0)
    f32 = rng.standard_normal((3, 5)).astype(np.float32)
    f16 = rng.standard_normal((2, 4)).astype(np.float16)
    bf_src = rng.standard_normal((4, 2)).astype(np.float32)
    bf_bits = (bf_src.view(np.uint32) >> 16).astype(np.uint16)       # 하위 16비트를 버린 bf16
    bf_expect = (bf_bits.astype(np.uint32) << 16).view(np.float32)
    p = tmp_path / "t.safetensors"
    _write(p, {"a": ("F32", f32, f32.tobytes()), "b": ("F16", f16, f16.tobytes()),
               "c": ("BF16", bf_src, bf_bits.tobytes())})
    assert safetensor_names(p) == ["a", "b", "c"]
    assert np.array_equal(read_safetensor(p, "a"), f32)
    assert np.array_equal(read_safetensor(p, "b"), f16)
    assert np.array_equal(read_safetensor(p, "c"), bf_expect)
    assert read_safetensor(p, "c").dtype == np.float32


def test_missing_name_lists_examples(tmp_path):
    p = tmp_path / "t.safetensors"
    a = np.zeros((1, 1), np.float32)
    _write(p, {"only": ("F32", a, a.tobytes())})
    try:
        read_safetensor(p, "nope")
    except KeyError as e:
        assert "only" in str(e)
    else:
        raise AssertionError("없는 이름을 받아들였다")
