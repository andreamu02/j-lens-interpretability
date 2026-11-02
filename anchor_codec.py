from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import struct
import sys
import zlib
from array import array
from pathlib import Path

STORAGE = "local-vocab-uint-float-zlib-base64-v1"
_MARKER = "__anchor_topk_slice_v1__"
_CODES = {"uint16": "H", "uint32": "I", "float32": "f", "float64": "d"}
_WIDTHS = {"uint16": 2, "uint32": 4, "float32": 4, "float64": 8}


def _encode(values, dtype):
    data = array(_CODES[dtype], values)
    if data.itemsize != _WIDTHS[dtype]:
        raise ValueError(f"Unsupported array width: {dtype}")
    if sys.byteorder != "little":
        data.byteswap()
    return base64.b64encode(zlib.compress(data.tobytes(), 9)).decode("ascii")


def _decode(data, dtype, count):
    if dtype not in _CODES or not isinstance(count, int) or count < 0:
        raise ValueError("Invalid packed array specification")
    raw = zlib.decompress(base64.b64decode(data, validate=True))
    if len(raw) != count * _WIDTHS[dtype]:
        raise ValueError("Packed array length mismatch")
    values = array(_CODES[dtype])
    if values.itemsize != _WIDTHS[dtype]:
        raise ValueError(f"Unsupported array width: {dtype}")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _exact_float32(value):
    try:
        restored = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return False
    if math.isnan(value):
        return math.isnan(restored)
    return restored == value and (
        value != 0 or math.copysign(1, restored) == math.copysign(1, value)
    )


def pack_readout(readout):
    vocab = {}
    vocab_ids = []
    vocab_tokens = []
    indices = []
    logp = []

    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return value
        if _MARKER in value:
            raise ValueError("Reserved packing key in input")
        topk = all(isinstance(value.get(k), list) for k in ("ids", "tokens", "logp"))
        if not topk:
            return {k: visit(v) for k, v in value.items()}
        ids, tokens, scores = (value[k] for k in ("ids", "tokens", "logp"))
        if not len(ids) == len(tokens) == len(scores):
            raise ValueError("Top-K array length mismatch")
        begin = len(indices)
        for token_id, token, score in zip(ids, tokens, scores, strict=False):
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError("Token ID must be an integer")
            if not isinstance(token, str) or not isinstance(score, float):
                raise ValueError("Top-K must contain token strings and float scores")
            key = (token_id, token)
            if key not in vocab:
                vocab[key] = len(vocab_ids)
                vocab_ids.append(token_id)
                vocab_tokens.append(token)
            indices.append(vocab[key])
            logp.append(score)
        result = {
            k: visit(v) for k, v in value.items() if k not in {"ids", "tokens", "logp"}
        }
        result[_MARKER] = [begin, len(ids)]
        return result

    tree = visit(readout)
    index_dtype = "uint16" if len(vocab_ids) <= 65536 else "uint32"
    logp_dtype = "float32" if all(_exact_float32(v) for v in logp) else "float64"
    return {
        "version": 1,
        "codec": "zlib+base64",
        "endian": "little",
        "vocab_ids": vocab_ids,
        "vocab_tokens": vocab_tokens,
        "count": len(indices),
        "index_dtype": index_dtype,
        "logp_dtype": logp_dtype,
        "indices": _encode(indices, index_dtype),
        "logp": _encode(logp, logp_dtype),
        "tree": tree,
    }


def unpack_readout(packed):
    if (
        packed.get("version") != 1
        or packed.get("codec") != "zlib+base64"
        or packed.get("endian") != "little"
    ):
        raise ValueError("Unsupported anchor packing format")
    if packed.get("index_dtype") not in {"uint16", "uint32"} or packed.get(
        "logp_dtype"
    ) not in {"float32", "float64"}:
        raise ValueError("Invalid packed array types")
    ids, tokens = packed["vocab_ids"], packed["vocab_tokens"]
    if len(ids) != len(tokens):
        raise ValueError("Vocabulary length mismatch")
    count = packed["count"]
    indices = _decode(packed["indices"], packed["index_dtype"], count)
    logp = _decode(packed["logp"], packed["logp_dtype"], count)
    if any(i >= len(ids) for i in indices):
        raise ValueError("Local vocabulary index out of bounds")
    cursor = 0

    def visit(value):
        nonlocal cursor
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: visit(v) for k, v in value.items() if k != _MARKER}
        if _MARKER in value:
            start, size = value[_MARKER]
            if (
                not isinstance(start, int)
                or not isinstance(size, int)
                or start != cursor
                or size < 0
                or start + size > count
            ):
                raise ValueError("Invalid Top-K slice")
            cursor += size
            local = indices[start : start + size]
            result.update(
                ids=[ids[i] for i in local],
                tokens=[tokens[i] for i in local],
                logp=list(logp[start : start + size]),
            )
        return result

    result = visit(packed["tree"])
    if cursor != count:
        raise ValueError("Unreferenced packed values")
    return result


def prompt_tail_alias(readout):
    assistant = readout.get("anchors", {}).get("assistant_start")
    if assistant is None:
        return None
    return {
        "aggregation": readout["aggregation"],
        "prompt_end_exclusive": assistant["anchor_end_exclusive"],
        "prefix_boundary_exact": assistant["boundary_exact"],
        "prefix_token_count_requested": assistant["prefix_token_count_requested"],
        "windows": readout["windows"],
        "top_k": readout["top_k"],
        "tail_positions": assistant["tail_positions"],
        "tail_token_ids": assistant["tail_token_ids"],
        "tail_tokens": assistant["tail_tokens"],
        "lens": assistant["lens"],
        "model": assistant["model"],
    }


def pack_record(record):
    out = dict(record)
    readout = out.get("anchor_readout")
    if readout is not None:
        if "anchor_readout_packed" in out:
            raise ValueError("Record contains both anchor formats")
        out.pop("anchor_readout")
        out["anchor_readout_packed"] = pack_readout(readout)
        alias = out.get("prompt_tail_readout")
        if alias is not None and alias == prompt_tail_alias(readout):
            out.pop("prompt_tail_readout")
            out["prompt_tail_readout_ref"] = "assistant_start"
    if out.get("prompt_tail_readout") is not None:
        if "prompt_tail_readout_packed" in out:
            raise ValueError("Record contains both prompt-tail formats")
        out["prompt_tail_readout_packed"] = pack_readout(out.pop("prompt_tail_readout"))
    return out


def unpack_record(record):
    out = dict(record)
    for plain, packed in (
        ("anchor_readout", "anchor_readout_packed"),
        ("prompt_tail_readout", "prompt_tail_readout_packed"),
    ):
        if packed in out:
            if plain in out:
                raise ValueError(f"Record contains both formats: {plain}")
            out[plain] = unpack_readout(out.pop(packed))
    if "prompt_tail_readout_ref" in out:
        if (
            out.pop("prompt_tail_readout_ref") != "assistant_start"
            or "prompt_tail_readout" in out
        ):
            raise ValueError("Invalid prompt-tail reference")
        if "anchor_readout" not in out:
            raise ValueError("Missing referenced anchor readout")
        alias = prompt_tail_alias(out["anchor_readout"])
        if alias is None:
            raise ValueError("Missing referenced assistant-start anchor")
        out["prompt_tail_readout"] = alias
    return out


def _open(path, mode):
    return (
        gzip.open(path, mode, encoding="utf-8")
        if str(path).endswith(".gz")
        else open(path, mode, encoding="utf-8")
    )


def iter_records(path):
    with _open(path, "rt") as source:
        for line in source:
            if line.strip():
                yield unpack_record(json.loads(line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.output.exists():
        parser.error("Output already exists")
    total = 0
    with _open(args.input, "rt") as source, _open(args.output, "xt") as target:
        for line in source:
            if not line.strip():
                continue
            original = json.loads(line)
            result = unpack_record(original) if args.decode else pack_record(original)
            if not args.decode and unpack_record(result) != original:
                raise ValueError(f"Round-trip mismatch at record {total + 1}")
            target.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            total += 1
    print(
        json.dumps(
            {
                "records": total,
                "input_bytes": args.input.stat().st_size,
                "output_bytes": args.output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
