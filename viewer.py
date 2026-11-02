from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import sys
import zlib
from array import array
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from anchor_codec import pack_readout, unpack_record

ROOT = Path("results")
BASE = Path(__file__).parent
VOCAB_ROOT = BASE / "vocab"
VOCAB_OVERRIDE = None
MODEL_IDS = {
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma3-12b": "google/gemma-3-12b-it",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}
INDEX = {}
TOK = ("tokens", "position_tokens")
LENS = ("lens_topk", "band_top_k")
MTOP = ("model_topk", "model_top_k")
PRED = ("pred", "pred_label")


def pick(data, names):
    return next((data[n] for n in names if n in data), None)


def open_binary(path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else path.open("rb")


def files():
    return sorted(
        p
        for p in ROOT.iterdir()
        if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))
    )


def signature():
    return "|".join(
        f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in files()
    )


def _case_header(line, index):
    text = line[:65536].decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    values = {}
    pos = 0
    try:
        pos = len(text) - len(text.lstrip())
        if text[pos] != "{":
            raise ValueError("Expected JSON object")
        pos += 1
        while True:
            while text[pos].isspace():
                pos += 1
            key, pos = decoder.raw_decode(text, pos)
            while text[pos].isspace():
                pos += 1
            if text[pos] != ":":
                raise ValueError("Expected colon")
            pos += 1
            while text[pos].isspace():
                pos += 1
            value, pos = decoder.raw_decode(text, pos)
            if key == "_meta":
                return None
            values[key] = value
            if "case_id" in values and "ground_truth" in values:
                return {
                    "i": index,
                    "id": values["case_id"],
                    "gt": values["ground_truth"],
                }
            while text[pos].isspace():
                pos += 1
            if text[pos] == "}":
                break
            if text[pos] != ",":
                raise ValueError("Expected comma")
            pos += 1
    except (ValueError, IndexError):
        pass
    row = json.loads(line)
    if "_meta" in row:
        return None
    return {
        "i": index,
        "id": row.get("case_id", str(index)),
        "gt": row.get("ground_truth"),
    }


def run_index(path):
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in INDEX:
        return INDEX[key]
    positions, cases = [], []
    with open_binary(path) as source:
        while True:
            pos = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            if line.strip():
                header = _case_header(line, len(positions))
                if header is not None:
                    positions.append(pos)
                    cases.append(header)
    for old in list(INDEX):
        if old[0] == str(path):
            INDEX.pop(old)
    INDEX[key] = (positions, cases)
    return INDEX[key]


def offsets(path):
    return run_index(path)[0]


@lru_cache(maxsize=8)
def _record_cached(path_str, index, mtime, size):
    path = Path(path_str)
    with open_binary(path) as source:
        source.seek(offsets(path)[index])
        return json.loads(source.readline())


def record(path, index):
    stat = path.stat()
    return _record_cached(str(path), index, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=32)
def _meta_cached(path_str, mtime, size):
    with open_binary(Path(path_str)) as source:
        for line in source:
            if line.strip():
                return json.loads(line).get("_meta", {})
    return {}


def meta(path):
    stat = path.stat()
    return _meta_cached(str(path), stat.st_mtime_ns, stat.st_size)


def run_tokenizer_id(run_meta):
    return run_meta.get("hf_id") or MODEL_IDS.get(run_meta.get("model"))


@lru_cache(maxsize=32)
def _vocab_cached(path_str, mtime, size):
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_vocab(path):
    stat = path.stat()
    return _vocab_cached(str(path), stat.st_mtime_ns, stat.st_size)


def select_vocab(run_meta):
    actual = run_tokenizer_id(run_meta)
    if not actual:
        return {}, None, "Modello/tokenizer non dichiarato nel JSONL"
    if VOCAB_OVERRIDE is not None:
        candidates = [VOCAB_OVERRIDE]
    else:
        model_name = next(
            (name for name, hf_id in MODEL_IDS.items() if hf_id == actual), None
        )
        preferred = (
            VOCAB_ROOT / model_name / "token_groups.json" if model_name else None
        )
        if preferred is not None and preferred.is_file():
            try:
                vocab = load_vocab(preferred)
            except (OSError, ValueError) as error:
                return {}, str(preferred), str(error)
            if vocab.get("meta", {}).get("tokenizer_id") != actual:
                return {}, str(preferred), f"Tokenizer incompatibile in {preferred}"
            return vocab, str(preferred), None
        candidates = sorted(VOCAB_ROOT.rglob("token_groups.json"))
    matches = []
    for path in candidates:
        try:
            vocab = load_vocab(path)
        except (OSError, ValueError) as error:
            if VOCAB_OVERRIDE is not None:
                return {}, str(path), str(error)
            continue
        if vocab.get("meta", {}).get("tokenizer_id") == actual:
            matches.append((vocab, path))
    if len(matches) == 1:
        vocab, path = matches[0]
        return vocab, str(path), None
    if not matches:
        return {}, None, f"Nessun vocabolario compatibile con {actual}"
    paths = ", ".join(str(path) for _, path in matches)
    return {}, None, f"Più vocabolari compatibili con {actual}: {paths}. Usa --vocab"


def vocab_status(run_meta):
    vocab, path, error = select_vocab(run_meta)
    expected = vocab.get("meta", {}).get("tokenizer_id")
    actual = run_tokenizer_id(run_meta)
    active = bool(expected and actual == expected)
    return {
        "active": active,
        "tokenizer_id": expected,
        "run_tokenizer_id": actual,
        "coverage": vocab.get("meta", {}).get("coverage"),
        "version": vocab.get("meta", {}).get("version"),
        "path": path,
        "error": error,
    }


def _unpack_uint_array(payload):
    if payload.get("codec") != "zlib+base64" or payload.get("dtype") not in {
        "uint16",
        "uint32",
    }:
        raise ValueError("Unsupported packed Top-K")
    data = array("H" if payload["dtype"] == "uint16" else "I")
    data.frombytes(zlib.decompress(base64.b64decode(payload["data"], validate=True)))
    if payload.get("endian", "little") != sys.byteorder:
        data.byteswap()
    if len(data) != math.prod(payload["shape"]):
        raise ValueError("Packed Top-K shape mismatch")
    return data


def _packed_cache(rec):
    if "_topk_unpacked" in rec:
        return rec["_topk_unpacked"]
    lp, mp = rec.get("lens_topk_packed"), rec.get("model_topk_packed")
    if not lp or not mp:
        return None
    result = {
        "lens": _unpack_uint_array(lp),
        "model": _unpack_uint_array(mp),
        "vocab": rec.get("topk_vocab", []),
        "vocab_ids": rec.get("topk_vocab_ids", []),
        "lens_shape": lp["shape"],
        "model_shape": mp["shape"],
        "layers": lp["layers"],
    }
    rec["_topk_unpacked"] = result
    return result


def packed_topk_at(rec, pos):
    cache = _packed_cache(rec)
    if cache is None:
        return None, None
    nl, npos, k = cache["lens_shape"]
    if pos < 0 or pos >= npos:
        raise ValueError("Position out of bounds")

    def tokens(data, start, length):
        indices = data[start : start + length]
        return [
            {
                "token": cache["vocab"][i],
                "id": cache["vocab_ids"][i] if cache["vocab_ids"] else None,
            }
            for i in indices
        ]

    layers = [
        {"layer": layer, "top": tokens(cache["lens"], (li * npos + pos) * k, k)}
        for li, layer in enumerate(cache["layers"])
    ]
    nmodel, mk = cache["model_shape"]
    model_top = tokens(cache["model"], pos * mk, mk) if pos < nmodel else None
    return layers, model_top


def position_readout(rec, pos, src):
    if pos < 0:
        raise ValueError("Negative position")
    if src == "output":
        layers, model_top = packed_topk_at(rec, pos)
        if layers is not None:
            return {
                "layers": sorted(layers, key=lambda x: int(x["layer"]), reverse=True),
                "model": model_top,
            }
        lens, model = pick(rec, LENS) or {}, pick(rec, MTOP) or []
    elif src == "input":
        lens, model = rec.get("input_lens_topk", {}), rec.get("input_model_topk", [])
    else:
        raise ValueError("Unknown token source")
    layers = [
        {"layer": layer, "top": lens[layer][pos]}
        for layer in sorted(lens, key=int, reverse=True)
        if pos < len(lens[layer])
    ]
    return {"layers": layers, "model": model[pos] if pos < len(model) else None}


def anchor_tree(rec):
    if "anchor_readout_packed" in rec:
        return rec["anchor_readout_packed"]["tree"]
    return rec.get("anchor_readout") or {}


def case_info(rec, run_meta):
    tree = anchor_tree(rec)
    return {
        "case_id": rec.get("case_id"),
        "cwe": rec.get("cwe"),
        "cwes": rec.get("cwes") or [rec.get("cwe")],
        "target_labels": rec.get("target_labels", {}),
        "ground_truth": rec.get("ground_truth"),
        "pred": pick(rec, PRED),
        "tokens": pick(rec, TOK) or [],
        "input_tokens": rec.get("input_tokens", []),
        "anchors": list(tree.get("anchors", {})),
        "windows": tree.get("windows", []),
        "vocab_status": vocab_status(run_meta),
    }


def anchor_readout(rec, name, window, limit):
    if "_anchor_decoded" not in rec:
        rec["_anchor_decoded"] = unpack_record(
            {k: v for k, v in rec.items() if not k.startswith("_")}
        ).get("anchor_readout", {})
    readout = rec["_anchor_decoded"]
    anchor = readout.get("anchors", {}).get(name)
    if anchor is None:
        raise ValueError("Anchor not present")

    def top(entry):
        ids, texts, scores = (
            entry.get("ids", []),
            entry.get("tokens", []),
            entry.get("logp", []),
        )
        return [
            {"token": text, "id": token_id, "logp": score, "rank": rank + 1}
            for rank, (token_id, text, score) in enumerate(
                zip(ids[:limit], texts[:limit], scores[:limit], strict=False)
            )
        ]

    layers = [
        {"layer": layer, "top": top(group[str(window)])}
        for layer, group in sorted(
            anchor.get("lens", {}).items(), key=lambda x: int(x[0]), reverse=True
        )
        if str(window) in group
    ]
    entry = anchor.get("model", {}).get(str(window))
    return {
        "layers": layers,
        "model": top(entry) if entry else None,
        "name": name,
        "window": window,
        "top_k": anchor.get("top_k", readout.get("top_k")),
        "aggregation": readout.get("aggregation"),
        "boundary_exact": anchor.get("boundary_exact"),
        "position": anchor.get("anchor_end_exclusive"),
    }


def _raw_binary_payload(payload):
    result = dict(payload)
    result["data"] = base64.b64encode(
        zlib.decompress(base64.b64decode(payload["data"]))
    ).decode("ascii")
    result["codec"] = "base64"
    return result


def _standalone_record(rec):
    result = {
        k: rec.get(k)
        for k in ("case_id", "cwe", "cwes", "target_labels", "ground_truth", "pair_id")
    }
    result.update(
        pred=pick(rec, PRED),
        tokens=pick(rec, TOK) or [],
        input_tokens=rec.get("input_tokens", []),
    )
    lp, mp = rec.get("lens_topk_packed"), rec.get("model_topk_packed")
    if lp and mp:
        result.update(
            topk_vocab=rec.get("topk_vocab", []),
            topk_vocab_ids=rec.get("topk_vocab_ids", []),
            lens_bin=_raw_binary_payload(lp),
            model_bin=_raw_binary_payload(mp),
        )
    else:
        result.update(lens_topk=pick(rec, LENS) or {}, model_topk=pick(rec, MTOP) or [])
    result.update(
        input_lens_topk=rec.get("input_lens_topk", {}),
        input_model_topk=rec.get("input_model_topk", []),
    )
    packed = rec.get("anchor_readout_packed")
    if packed is None and rec.get("anchor_readout"):
        packed = pack_readout(rec["anchor_readout"])
    if packed:
        packed = dict(packed)
        for key in ("indices", "logp"):
            packed[key] = base64.b64encode(
                zlib.decompress(base64.b64decode(packed[key]))
            ).decode("ascii")
        packed["codec"] = "base64"
        result["anchor_bin"] = packed
    return result


def render_page(data=None):
    html = (BASE / "viewer_ui.html").read_text()
    payload = (
        json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
            "<", "\\u003c"
        )
        if data is not None
        else "null"
    )
    return html.replace("__DATA__", payload)


def export_single_html(path, out):
    records = []
    with open_binary(path) as source:
        for line in source:
            if line.strip():
                rec = json.loads(line)
                if "_meta" not in rec:
                    records.append(_standalone_record(rec))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_page(
            {
                "name": path.name,
                "meta": meta(path),
                "records": records,
                "vocab": select_vocab(meta(path))[0],
                "vocab_status": vocab_status(meta(path)),
            }
        )
    )
    print(
        f"HTML standalone: {out} ({out.stat().st_size / 1048576:.1f} MiB, {len(records)} casi)"
    )


def run_path(name):
    path = ROOT / name
    if path.parent.resolve() != ROOT.resolve() or path not in files():
        raise ValueError("Unknown run")
    return path


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/":
                data = render_page().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if url.path == "/api/vocab":
                path = run_path(q["f"][0])
                self.send_json(select_vocab(meta(path))[0])
                return
            if url.path == "/api/version":
                self.send_json({"sig": signature()})
                return
            if url.path == "/api/files":
                self.send_json([{"name": p.name} for p in files()])
                return
            path = run_path(q["f"][0])
            if url.path == "/api/cases":
                self.send_json(run_index(path)[1])
                return
            rec = record(path, int(q["i"][0]))
            if url.path == "/api/case":
                self.send_json(case_info(rec, meta(path)))
            elif url.path == "/api/pos":
                self.send_json(
                    position_readout(rec, int(q["p"][0]), q.get("src", ["output"])[0])
                )
            elif url.path == "/api/anchor":
                limit = int(q.get("k", ["50"])[0])
                if limit not in (50, 200, 500):
                    raise ValueError("Invalid Top-K limit")
                self.send_json(anchor_readout(rec, q["a"][0], int(q["w"][0]), limit))
            else:
                self.send_error(404)
        except Exception as error:
            self.send_json({"error": str(error)}, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--vocab", type=Path, help="Override: compiled vocabulary file or folder"
    )
    parser.add_argument("--vocab-root", type=Path, default=BASE / "vocab")
    parser.add_argument("--export-html", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    global ROOT, VOCAB_ROOT, VOCAB_OVERRIDE
    ROOT = Path(args.dir).expanduser()
    VOCAB_ROOT = args.vocab_root.expanduser().resolve()
    if args.vocab:
        VOCAB_OVERRIDE = args.vocab.expanduser().resolve()
        if VOCAB_OVERRIDE.is_dir():
            VOCAB_OVERRIDE /= "token_groups.json"
        load_vocab(VOCAB_OVERRIDE)
    if args.export_html:
        path = args.export_html.expanduser()
        export_single_html(
            path, args.out.expanduser() if args.out else path.with_suffix(".html")
        )
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"{len(files())} file in {ROOT}/ -> http://127.0.0.1:{args.port}", flush=True)
    print(
        f"ssh -L {args.port}:localhost:{args.port} {os.environ.get('USER', 'utente')}@<host>",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
