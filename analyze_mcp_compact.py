#!/usr/bin/env python3
"""CPU-only compact MCP analysis: exact saved masses/SG, uncertainty and discordance.
Writes exactly one self-contained HTML (or a ZIP containing that HTML).
No model loading, token-level database, heatmaps, or per-case plot explosion.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import io
import itertools
import json
import math
import os
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

# Respect explicit caller limits; bound BLAS defaults before importing NumPy.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "5.0-interactive-series-autoscale"
GROUPS = ("safety", "vuln", "expected_cwe")
G_LABEL = ("Safe", "Vuln", "CWE attesa")
COLORS = {
    "safe": "#18864b",
    "vulnerable": "#c63735",
    "TP": "#16803c",
    "TN": "#2563eb",
    "FP": "#e66b16",
    "FN": "#7c3aed",
}
METRICS = (
    "mass_safe",
    "mass_vuln",
    "mass_cwe",
    "sg_safe",
    "sg_vuln",
    "sg_cwe",
    "sgz_safe",
    "sgz_vuln",
    "sgz_cwe",
    "mass_generic",
    "mass_outside",
    "d_mass",
    "d_sg",
    "discord_mass",
    "discord_sg",
    "ab_entropy_bits",
    "chosen_conditional",
    "pair_probability",
    "chosen_margin",
    "top1_probability",
    "top1_outside",
    "sg_generic",
    "sgz_generic",
    "mass_union",
    "sg_union",
    "sgz_union",
    "mass_other_cwe_mean",
    "mass_other_cwe_max",
    "sg_other_cwe_mean",
    "sg_other_cwe_max",
    "d_cwe_mass",
    "d_cwe_sg",
    "cwe_rank_mass",
    "cwe_rank_sg",
)
M = {k: j for j, k in enumerate(METRICS)}


def anchor_sort_key(name):
    return ({"code_end": 0, "mcp_record": 1, "mcp_pre_verdict": 2}.get(name, 3), name)


def plot_context(s):
    model = s.run.get("meta", {}).get("model", s.run.get("meta", {}).get("hf_id", ""))
    variant = s.run.get("variant", "")
    return " · ".join(str(x) for x in (model, variant.upper(), s.anchor) if x)


def norm(x):
    return {
        "safe": "safe",
        "patched": "safe",
        "vuln": "vulnerable",
        "vulnerable": "vulnerable",
        "unsafe": "vulnerable",
    }.get(str(x).lower(), "unknown")


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                try:
                    yield json.loads(text)
                except ValueError as ex:
                    raise ValueError(f"{path}:{line}: invalid JSON") from ex


def finite(x):
    return float(x) if x is not None and math.isfinite(float(x)) else np.nan


def mean(a, axis=None):
    a = np.asarray(a, dtype=float)
    count = np.isfinite(a).sum(axis=axis)
    total = np.where(np.isfinite(a), a, 0).sum(axis=axis)
    return np.divide(
        total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0
    )


def signature(groups):
    g = {k: sorted(set(v)) for k, v in groups.items()}
    if "union" in g or "outside" in g:
        raise ValueError("Compiled groups must not contain derived union/outside")
    g["union"] = sorted({t for ids in g.values() for t in ids})
    return hashlib.sha256(
        json.dumps(g, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_data(x):
    if x.get("codec") != "zlib+base64" or x.get("dtype") not in ("<f4", "<f8"):
        raise ValueError("Unsupported full-scan statistics array")
    a = np.frombuffer(
        zlib.decompress(base64.b64decode(x["data"], validate=True)), dtype=x["dtype"]
    )
    if a.size != math.prod(x["shape"]):
        raise ValueError("Full-scan array length mismatch")
    return a.reshape(x["shape"])


def fullscan_anchors(rec, requested):
    """Adapt v7 full-scan matrices without decoding or expanding top-k tokens."""
    fr = rec["full_readout"]
    if fr.get("version") != 1:
        raise ValueError("Unsupported full-scan version")
    names = fr["group_names"]
    positions = rec["position_map"]["anchors"]
    selected = {
        k: v for k, v in positions.items() if requested is None or k in requested
    }
    out = {
        k: {
            "boundary_exact": p.get("boundary_exact", False),
            "vocab_scores": {"profile": "full", "lens": {}, "model": {}},
            "choice_scores": {"lens": {}, "model": {}},
        }
        for k, p in selected.items()
    }
    streams = [("lens", str(l), x) for l, x in fr["lens"].items()] + [
        ("model", None, fr["model"])
    ]
    for stream, layer, data in streams:
        masses, means, glob, ab = (
            array_data(data[k])
            for k in ("group_mass", "group_mean", "global", "choice_logits")
        )
        for name, p in selected.items():
            pos = p.get("position")
            if pos is None:
                continue
            if not 0 <= pos < len(masses):
                raise ValueError("Full-scan anchor position out of range")
            groups = {}
            for j, g in enumerate(names):
                sg = float(means[pos, j] - glob[pos, 0])
                groups[g] = {
                    "probability_mass": float(masses[pos, j]),
                    "sg": sg,
                    "sg_z": sg / float(glob[pos, 1]) if glob[pos, 1] > 0 else None,
                }
            if "union" in groups:
                groups["outside"] = {
                    "probability_mass": 1 - groups["union"]["probability_mass"],
                    "sg": None,
                    "sg_z": None,
                }
            margin = float(ab[pos, 0] - ab[pos, 1])
            qa = float(np.exp(-np.logaddexp(0, -margin)))
            pa, pb = np.exp(ab[pos] - glob[pos, 2]).tolist()
            entropy = -sum(q * math.log(q) for q in (qa, 1 - qa) if q > 0)
            choice = {
                "conditional_probability": {"A": qa, "B": 1 - qa},
                "probability": {"A": pa, "B": pb},
                "pair_probability": pa + pb,
                "conditional_entropy_nats": entropy,
                "logit_margin_A_minus_B": margin,
            }
            bucket = out[name]
            if stream == "model":
                bucket["vocab_scores"]["model"]["1"] = {"groups": groups}
                bucket["choice_scores"]["model"]["1"] = choice
            else:
                bucket["vocab_scores"]["lens"][layer] = {"1": {"groups": groups}}
                bucket["choice_scores"]["lens"][layer] = {"1": choice}
    return out


def anchor_tree(rec, requested):
    if rec.get("anchor_readout"):
        return rec["anchor_readout"].get("anchors", {})
    packed = rec.get("anchor_readout_packed")
    if packed:
        if packed.get("version") != 1 or packed.get("codec") != "zlib+base64":
            raise ValueError("Unsupported packed anchor format")
        # Exact statistics are ordinary fields in tree; top-k binary blobs are unnecessary.
        return packed["tree"].get("anchors", {})
    if rec.get("full_readout"):
        return fullscan_anchors(rec, requested)
    return {}


def top1_metrics(rec, anchor, layer, union, cache):
    """One diagnostic token per anchor; never export a token-level database."""
    rows = (
        anchor.get("model", {})
        if layer == -1
        else anchor.get("lens", {}).get(str(layer), {})
    )
    entry = rows.get("1", {})
    ids, logp = entry.get("ids"), entry.get("logp")
    if ids and logp:
        token_id, lp = int(ids[0]), float(logp[0])
    elif "__anchor_topk_slice_v1__" in entry:
        packed = rec["anchor_readout_packed"]
        start, size = entry["__anchor_topk_slice_v1__"]
        if size == 0:
            return np.nan, np.nan
        if not cache:
            for field, dtype_key in (
                ("indices", "index_dtype"),
                ("logp", "logp_dtype"),
            ):
                dt = {
                    "uint16": "<u2",
                    "uint32": "<u4",
                    "float32": "<f4",
                    "float64": "<f8",
                }.get(packed[dtype_key])
                if dt is None or packed.get("endian") != "little":
                    raise ValueError("Unsupported top-k array format")
                arr = np.frombuffer(
                    zlib.decompress(base64.b64decode(packed[field], validate=True)),
                    dtype=dt,
                )
                if arr.size != packed["count"]:
                    raise ValueError("Top-k packed array size mismatch")
                cache[field] = arr
        if (
            not isinstance(start, int)
            or not isinstance(size, int)
            or start < 0
            or start + size > packed["count"]
        ):
            raise ValueError("Invalid top-k slice")
        local = int(cache["indices"][start])
        if not 0 <= local < len(packed["vocab_ids"]):
            raise ValueError("Top-k vocabulary index out of range")
        token_id, lp = int(packed["vocab_ids"][local]), float(cache["logp"][start])
    else:
        return np.nan, np.nan
    if not math.isfinite(lp) or lp > 1e-5:
        raise ValueError("Invalid top-1 log probability")
    return math.exp(lp), float(token_id not in union)


def values(entry, choice, case, eps):
    groups = entry.get("groups", {})
    expected = case["cwe"]
    if any(g not in groups for g in ("safety", "vuln", expected)):
        raise ValueError(
            f"Missing exact safe/vuln/CWE statistics for {case['case_id']}"
        )
    result = np.full(len(METRICS), np.nan)
    for j, g in enumerate(("safety", "vuln", expected)):
        x = groups[g]
        for offset, field in ((0, "probability_mass"), (3, "sg"), (6, "sg_z")):
            result[offset + j] = finite(x.get(field))
    if not np.isfinite(result[:6]).all():
        raise ValueError(
            "Exact mass and SG must be saved and finite; top-k cannot reconstruct them"
        )
    if np.any(result[:3] < -1e-6) or np.any(result[:3] > 1 + 1e-5):
        raise ValueError("Exact probability mass outside [0,1]")
    for field, group in (("mass_generic", "generic"), ("mass_outside", "outside")):
        result[M[field]] = finite(groups.get(group, {}).get("probability_mass"))
    if not np.isfinite(result[M["mass_outside"]]) and "union" in groups:
        result[M["mass_outside"]] = 1 - finite(groups["union"].get("probability_mass"))
    result[M["d_mass"]] = result[1] - result[0]
    result[M["d_sg"]] = result[4] - result[3]
    result[M["sg_generic"]] = finite(groups.get("generic", {}).get("sg"))
    result[M["sgz_generic"]] = finite(groups.get("generic", {}).get("sg_z"))
    union = groups.get("union", {})
    for metric, field in (
        ("mass_union", "probability_mass"),
        ("sg_union", "sg"),
        ("sgz_union", "sg_z"),
    ):
        result[M[metric]] = finite(union.get(field))
    for field, stem, expected_value in (
        ("probability_mass", "mass", result[2]),
        ("sg", "sg", result[5]),
    ):
        others = np.array(
            [
                finite(v.get(field))
                for g, v in groups.items()
                if g.startswith("CWE-") and g != expected
            ]
        )
        others = others[np.isfinite(others)]
        if len(others):
            result[M[f"{stem}_other_cwe_mean"]] = float(others.mean())
            result[M[f"{stem}_other_cwe_max"]] = float(others.max())
            result[M[f"d_cwe_{stem}"]] = expected_value - float(others.max())
            result[M[f"cwe_rank_{stem}"]] = (
                1
                + float((others > expected_value + eps).sum())
                + 0.5 * float((np.abs(others - expected_value) <= eps).sum())
            )
    pred = case["pred"]
    if pred in ("safe", "vulnerable"):
        for field, contrast in (("discord_mass", "d_mass"), ("discord_sg", "d_sg")):
            d = result[M[contrast]]
            if abs(d) > eps:
                result[M[field]] = float((d > 0) != (pred == "vulnerable"))
    if choice:
        qa = finite(choice.get("conditional_probability", {}).get("A"))
        qb = finite(choice.get("conditional_probability", {}).get("B"))
        if (
            not np.isfinite([qa, qb]).all()
            or min(qa, qb) < 0
            or max(qa, qb) > 1 + 1e-6
            or abs(qa + qb - 1) > 1e-5
        ):
            raise ValueError("Invalid conditional A/B probabilities")
        entropy = -sum(q * math.log(q) for q in (qa, qb) if q > 0)
        saved = finite(choice.get("conditional_entropy_nats"))
        if not np.isfinite(saved) or abs(saved - entropy) > 3e-5:
            raise ValueError("Inconsistent window-1 A/B entropy")
        result[M["ab_entropy_bits"]] = saved / math.log(2)
        pair = finite(choice.get("pair_probability"))
        if not np.isfinite(pair) or not 0 <= pair <= 1 + 1e-5:
            raise ValueError("Invalid absolute probability of A+B")
        result[M["pair_probability"]] = pair
        if pred in ("safe", "vulnerable"):
            result[M["chosen_conditional"]] = qa if pred == "vulnerable" else qb
            margin = finite(choice.get("logit_margin_A_minus_B"))
            result[M["chosen_margin"]] = margin if pred == "vulnerable" else -margin
    return result


def load_inputs(paths, vocab_path, requested, eps):
    vf = vocab_path / "token_groups.json" if vocab_path.is_dir() else vocab_path
    vocab = json.loads(vf.read_text(encoding="utf-8"))
    expected_model = vocab["meta"]["tokenizer_id"]
    group_sig = signature(vocab["groups"])
    union_ids = {t for ids in vocab["groups"].values() for t in ids}
    catalog = vf.parent / "concepts.json"
    catalog_sha = (
        hashlib.sha256(catalog.read_bytes()).hexdigest() if catalog.exists() else None
    )
    runs, seen_runs, seen_content = [], set(), set()
    for path in paths:
        run = path.name.removesuffix(".gz").removesuffix(".jsonl")
        if run in seen_runs:
            raise ValueError(f"Duplicate run name: {run}")
        seen_runs.add(run)
        meta, cases, data, audit = None, {}, {}, Counter()
        other_values = defaultdict(list)
        digest = hashlib.sha256()
        for rec in records(path):
            if "_meta" in rec:
                if meta is not None or cases:
                    raise ValueError("Metadata must occur once before cases")
                meta = rec["_meta"]
                if meta.get("hf_id") != expected_model:
                    raise ValueError(f"Model/vocabulary mismatch in {path}")
                if len(meta.get("pilot", {}).get("frozen", {}).get("markers", [])) > 1:
                    raise ValueError("Do not pool multiple marker conditions")
                sg = meta.get("vocab_scores", {})
                if (
                    sg.get("version") != "sg-full-vocab-v1"
                    or sg.get("profile", "full") != "full"
                    or sg.get("group_signature") != group_sig
                ):
                    raise ValueError(
                        f"Saved SG groups do not match --vocab in {path}; no recomputation from top-k"
                    )
                if (
                    catalog_sha
                    and sg.get("catalog_sha256")
                    and catalog_sha != sg["catalog_sha256"]
                ):
                    raise ValueError(
                        "Catalog differs from the one used during extraction"
                    )
                continue
            if meta is None:
                raise ValueError("Missing _meta header")
            if rec.get("pilot_marker") is not None:
                raise ValueError(
                    "Marker pilots need separate analyses; do not pool marker repetitions"
                )
            cid = str(rec["case_id"])
            if cid in cases:
                raise ValueError(f"Duplicate case_id: {run}/{cid}")
            cwes = set(rec.get("cwes") or [rec.get("cwe")]) - {None}
            if len(cwes) != 1:
                raise ValueError(
                    "Compact expected-CWE analysis requires one CWE per case; overlapping CWE masses must not be summed"
                )
            gt, pred = (
                norm(rec.get("ground_truth")),
                norm(rec.get("pred", rec.get("pred_label"))),
            )
            if gt == "unknown":
                raise ValueError(f"Unknown ground truth: {cid}")
            outcome = {
                ("vulnerable", "vulnerable"): "TP",
                ("safe", "safe"): "TN",
                ("safe", "vulnerable"): "FP",
                ("vulnerable", "safe"): "FN",
            }.get((gt, pred), "NC")
            case = {
                "case_id": cid,
                "cluster": str(rec.get("quartet_id") or rec.get("pair_id") or cid),
                "cwe": next(iter(cwes)),
                "gt": gt,
                "pred": pred,
                "outcome": outcome,
                "truncated": bool(rec.get("truncated")),
                "carrier_ok": rec.get("carrier_ok"),
            }
            cases[cid] = case
            digest.update(
                json.dumps(
                    [
                        cid,
                        case["gt"],
                        case["pred"],
                        rec.get("source_code_sha256"),
                        rec.get("response"),
                    ],
                    sort_keys=True,
                ).encode()
            )
            tree = anchor_tree(rec, requested)
            top1_cache = {}
            for anchor, a in tree.items():
                if requested is not None and anchor not in requested:
                    continue
                if not a.get("boundary_exact"):
                    audit[f"{anchor}: inexact boundary"] += 1
                    continue
                scores = a.get("vocab_scores")
                if not scores:
                    audit[f"{anchor}: missing exact statistics"] += 1
                    continue
                if scores.get("profile", "full") != "full":
                    raise ValueError("Only the full vocabulary profile is supported")
                entries = [(int(l), ws) for l, ws in scores.get("lens", {}).items()] + [
                    (-1, scores.get("model", {}))
                ]
                for layer, ws in entries:
                    e = ws.get("1")
                    if e is None:
                        continue
                    if (
                        e.get("used_positions", 1) != 1
                        or e.get("aggregation", "mean_of_per_position_statistics")
                        != "mean_of_per_position_statistics"
                    ):
                        raise ValueError(
                            "Only exact single-position window 1 is supported"
                        )
                    choices = a.get("choice_scores", {})
                    ch = (
                        choices.get("model", {}).get("1")
                        if layer == -1
                        else choices.get("lens", {}).get(str(layer), {}).get("1")
                    )
                    key = (cid, anchor, layer)
                    if key in data:
                        raise ValueError("Duplicate anchor/layer row")
                    row = values(e, ch, case, eps)
                    row[M["top1_probability"]], row[M["top1_outside"]] = top1_metrics(
                        rec, a, layer, union_ids, top1_cache
                    )
                    data[key] = row
                    if layer >= 0:
                        for group, score in e.get("groups", {}).items():
                            if group.startswith("CWE-") and group != case["cwe"]:
                                other_values[(cid, anchor, group)].append(
                                    (
                                        finite(score.get("probability_mass")),
                                        finite(score.get("sg")),
                                    )
                                )
        if not cases or not data:
            raise ValueError(f"No usable exact anchor statistics in {path}")
        fingerprint = (
            meta.get("hf_id"),
            repr(meta.get("band")),
            str(meta.get("dataset", "")),
            digest.hexdigest(),
        )
        if fingerprint in seen_content:
            raise ValueError(
                "Repeated case/prediction/code content and band: duplicate inputs"
            )
        seen_content.add(fingerprint)
        anchors = sorted(
            {a for _, a, layer in data if layer >= 0},
            key=anchor_sort_key,
        )
        other_cwes = defaultdict(dict)
        for (cid, anchor, group), observations in other_values.items():
            other_cwes[(cid, anchor)][group] = mean(observations, axis=0).tolist()
        runs.append(
            {
                "name": run,
                "path": str(path),
                "meta": meta,
                "cases": cases,
                "data": data,
                "anchors": anchors,
                "audit": dict(audit),
                "other_cwes": dict(other_cwes),
                "generic_equals_union": set(vocab["groups"].get("generic", []))
                == union_ids,
                "variant": meta.get(
                    "variant",
                    "extended"
                    if "extended" in str(meta.get("dataset", path)).lower()
                    else "base",
                ),
            }
        )
        print(
            f"Read {path.name}: {len(cases)} cases, {len(anchors)} anchors, {len(data)} position/layer rows",
            flush=True,
        )
    return runs


class Summary:
    """Pair/quartet bootstrap; layers never become independent observations."""

    def __init__(self, run, anchor, reps, seed):
        self.run, self.anchor = run, anchor
        self.layers = self.layers_for(run, anchor)
        self.ids = sorted(
            {cid for cid, a, layer in run["data"] if a == anchor and layer >= 0}
        )
        self.cases = [run["cases"][cid] for cid in self.ids]
        self.x = np.array(
            [
                [
                    run["data"].get((cid, anchor, l), np.full(len(METRICS), np.nan))
                    for l in self.layers
                ]
                for cid in self.ids
            ]
        )
        clusters = sorted({x["cluster"] for x in self.cases})
        self.cluster_index = np.array(
            [clusters.index(x["cluster"]) for x in self.cases]
        )
        self.nclusters = len(clusters)
        self.weights = np.zeros((reps, len(clusters)))
        if reps:
            draw = np.random.default_rng(seed).integers(
                len(clusters), size=(reps, len(clusters))
            )
            for j in range(len(clusters)):
                self.weights[:, j] = (draw == j).sum(1)
        self.cache = {}

    @staticmethod
    def layers_for(run, anchor):
        return sorted({l for _, a, l in run["data"] if a == anchor and l >= 0})

    def select(self, field=None, value=None):
        return np.array([field is None or x[field] == value for x in self.cases])

    def estimate(self, field=None, value=None, collapse=False):
        key = (field, value, collapse)
        if key in self.cache:
            return self.cache[key]
        mask = self.select(field, value)
        x = mean(self.x, axis=1)[:, None, :] if collapse else self.x
        y = x[mask]
        shape = x.shape[1:]
        if not len(y):
            result = (
                np.full(shape, np.nan),
                np.full(shape, np.nan),
                np.full(shape, np.nan),
                0,
            )
        else:
            center = mean(y, axis=0)
            low, high = np.full(shape, np.nan), np.full(shape, np.nan)
            if len(self.weights) and len(set(self.cluster_index[mask])) >= 2:
                sums = np.zeros((self.nclusters, *shape))
                counts = np.zeros_like(sums)
                for row, cl in zip(y, self.cluster_index[mask], strict=True):
                    sums[cl] += np.where(np.isfinite(row), row, 0)
                    counts[cl] += np.isfinite(row)
                numerator = self.weights @ sums.reshape(self.nclusters, -1)
                denominator = self.weights @ counts.reshape(self.nclusters, -1)
                boot = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan),
                    where=denominator > 0,
                )
                for j in range(boot.shape[1]):
                    finite_boot = boot[np.isfinite(boot[:, j]), j]
                    if len(finite_boot) >= max(20, len(boot) // 2):
                        low.flat[j], high.flat[j] = np.quantile(
                            finite_boot, [0.025, 0.975]
                        )
            result = center, low, high, len(y)
        self.cache[key] = result
        return result


def fmt(x, percent=False):
    if not np.isfinite(x):
        return "—"
    return f"{x * (100 if percent else 1):.3f}" + ("%" if percent else "")


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join("<th>" + html.escape(str(x)) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _chart_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _chart_points(x, y):
    return [[_chart_number(a), _chart_number(b)] for a, b in zip(x, y, strict=True)]


def _chart_key(label):
    import re

    return re.sub(r"\s*(?:\(n=\d+\)|n=\d+)\s*$", "", label).strip()


def _interactive_chart_data(fig):
    """Export plotted values and intervals, never infer numbers from SVG coordinates."""
    from matplotlib.collections import PathCollection, PolyCollection
    from matplotlib.colors import to_hex
    from matplotlib.container import BarContainer

    width, height = fig.get_size_inches() * 100
    panels = []
    for ax in fig.axes:
        pos = ax.get_position()
        series, refs = [], []
        for line in ax.lines:
            label = str(line.get_label())
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            if label.startswith("_"):
                if len(x) == 2 and len(y) == 2:
                    if np.allclose(x, [0, 1]) and y[0] == y[1]:
                        refs.append({"axis": "y", "value": _chart_number(y[0])})
                    elif np.allclose(y, [0, 1]) and x[0] == x[1]:
                        refs.append({"axis": "x", "value": _chart_number(x[0])})
                continue
            color = to_hex(line.get_color())
            bands = []
            for collection in ax.collections:
                if not isinstance(collection, PolyCollection):
                    continue
                fc = collection.get_facecolor()
                if len(fc) and to_hex(fc[0]) == color:
                    bands.extend(
                        [
                            _chart_points(path.vertices[:, 0], path.vertices[:, 1])
                            for path in collection.get_paths()
                        ]
                    )
            series.append(
                {
                    "kind": "line",
                    "label": label,
                    "key": _chart_key(label),
                    "color": color,
                    "style": line.get_linestyle(),
                    "points": _chart_points(x, y),
                    "bands": bands,
                    "visible": not str(line.get_gid() or "").startswith("other-cwe-"),
                }
            )
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            label = str(collection.get_label())
            if label.startswith("_"):
                continue
            offsets = np.ma.filled(collection.get_offsets(), np.nan)
            fc = collection.get_facecolor()
            series.append(
                {
                    "kind": "scatter",
                    "label": label,
                    "key": _chart_key(label),
                    "color": to_hex(fc[0]) if len(fc) else "#334155",
                    "points": _chart_points(offsets[:, 0], offsets[:, 1]),
                    "visible": True,
                }
            )
        for container in ax.containers:
            if not isinstance(container, BarContainer):
                continue
            patches = container.patches
            label = str(container.get_label())
            if label.startswith("_"):
                label = next(
                    (
                        str(p.get_label())
                        for p in patches
                        if not str(p.get_label()).startswith("_")
                    ),
                    label,
                )
            if label.startswith("_"):
                continue
            series.append(
                {
                    "kind": "bars",
                    "label": label,
                    "key": _chart_key(label),
                    "color": to_hex(patches[0].get_facecolor())
                    if patches
                    else "#334155",
                    "rects": [
                        [
                            _chart_number(p.get_x()),
                            _chart_number(p.get_y()),
                            _chart_number(p.get_width()),
                            _chart_number(p.get_height()),
                        ]
                        for p in patches
                    ],
                    "visible": True,
                }
            )
        panels.append(
            {
                "title": ax.get_title(),
                "titleColor": to_hex(ax.title.get_color()),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "rect": [
                    pos.x0 * width,
                    (1 - pos.y1) * height,
                    pos.width * width,
                    pos.height * height,
                ],
                "xlim": list(map(float, ax.get_xlim())),
                "ylim": list(map(float, ax.get_ylim())),
                "background": to_hex(ax.get_facecolor()),
                "series": series,
                "refs": refs,
                "notes": [t.get_text() for t in ax.texts],
            }
        )
    shared = []
    used = set()
    for i, ax in enumerate(fig.axes):
        if i in used:
            continue
        group = [
            j
            for j, other in enumerate(fig.axes)
            if ax.get_shared_y_axes().joined(ax, other)
        ]
        if len(group) > 1:
            shared.append(group)
            used.update(group)
    return {
        "width": float(width),
        "height": float(height),
        "title": fig._suptitle.get_text() if fig._suptitle else "",
        "panels": panels,
        "sharedY": shared,
    }


def figure_svg(fig):
    # Keep the original Matplotlib SVG as an offline fallback without JavaScript.
    data = _interactive_chart_data(fig)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    value = buf.getvalue()
    svg = value[value.index("<svg") :]
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    return (
        '<div class="interactive-chart"><script type="application/json" class="chart-spec">'
        + payload
        + '</script><div class="chart-controls"></div><div class="chart-canvas">'
        + svg
        + "</div></div>"
    )


def curve(ax, summary, metric, field, values, scale=1):
    is_sg = metric.startswith("sg") or metric in ("d_sg", "discord_sg")
    gt_palette = (
        {"safe": "#0277bd", "vulnerable": "#c27b00"}
        if is_sg
        else {"safe": "#18864b", "vulnerable": "#c63735"}
    )
    palette = gt_palette if field == "gt" else (COHORT_SG_COLORS if is_sg else COLORS)
    for category in values:
        center, lo, hi, n = summary.estimate(field, category)
        j = M[metric]
        if not n or not np.isfinite(center[:, j]).any():
            continue
        color = palette[category]
        ax.plot(
            summary.layers,
            center[:, j] * scale,
            color=color,
            ls="--" if is_sg else "-",
            label=f"{category} (n={n})",
            linewidth=1.6,
        )
        ax.fill_between(
            summary.layers, lo[:, j] * scale, hi[:, j] * scale, color=color, alpha=0.12
        )
    ax.set_facecolor("#f3f7fc" if is_sg else "#fffdf7")
    ax.grid(alpha=0.2)
    ax.set_xlabel("Layer")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)


def gt_plot(s):
    return plot_grid(s, "gt")


COHORT_SG_COLORS = {"TP": "#65a30d", "TN": "#0e7490", "FP": "#be3b21", "FN": "#a21caf"}


def plot_grid(s, cohort):
    """Keep the same metrics across GT and outcome panels, with explicit visual families."""
    specs = [
        ("Safe", "mass_safe", "sg_safe"),
        ("Vuln", "mass_vuln", "sg_vuln"),
        ("CWE attesa", "mass_cwe", "sg_cwe"),
        ("Union", "mass_union", "sg_union"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5))
    categories = ("safe", "vulnerable") if cohort == "gt" else ("TP", "TN", "FP", "FN")
    for col, (label, mass, sg) in enumerate(specs):
        for row, metric in enumerate((mass, sg)):
            ax = axs[row, col]
            curve(ax, s, metric, cohort, categories, 100 if row == 0 else 1)
            ax.set_title(
                label + (" · massa esatta (%)" if row == 0 else " · SG (logit)"),
                fontsize=10,
            )
    title = (
        "Ground truth safe/vuln"
        if cohort == "gt"
        else "Esiti del verdetto: TP / TN / FP / FN"
    )
    fig.suptitle(title + " · " + plot_context(s), fontsize=11)
    fig.tight_layout()
    return figure_svg(fig)


def outcome_group_plot(s):
    """Transpose the existing outcome view: one class per column, groups overlaid."""
    specs = [
        ("Safe", "mass_safe", "sg_safe", "#18864b", False),
        ("Vuln", "mass_vuln", "sg_vuln", "#c63735", False),
        ("CWE attesa", "mass_cwe", "sg_cwe", "#1565c0", False),
        ("Union", "mass_union", "sg_union", "#6951a2", False),
        (
            "Media altre CWE",
            "mass_other_cwe_mean",
            "sg_other_cwe_mean",
            "#bd7a15",
            True,
        ),
        (
            "Massimo altre CWE",
            "mass_other_cwe_max",
            "sg_other_cwe_max",
            "#8b4513",
            True,
        ),
    ]
    uid = hashlib.sha256((s.run["name"] + "|" + s.anchor).encode()).hexdigest()[:12]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharey="row")
    for col, outcome in enumerate(("TP", "TN", "FP", "FN")):
        center, low, high, n = s.estimate("outcome", outcome)
        for row in range(2):
            ax = axs[row, col]
            ax.set_facecolor("#fffdf7" if row == 0 else "#f3f7fc")
            ax.set_title(f"{outcome} · n={n}", color=COLORS[outcome], fontweight="bold")
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("Massa esatta (%)" if row == 0 else "SG (logit)")
            if not n:
                ax.text(
                    0.5,
                    0.5,
                    "Nessun esempio",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#6b7280",
                )
                continue
            for k, (label, mass, sg, color, optional) in enumerate(specs):
                metric, scale = (mass, 100) if row == 0 else (sg, 1)
                j = M[metric]
                if not np.isfinite(center[:, j]).any():
                    continue
                style = (
                    (":" if k == 4 else "-.")
                    if optional
                    else ("-" if row == 0 else "--")
                )
                (line,) = ax.plot(
                    s.layers,
                    center[:, j] * scale,
                    color=color,
                    ls=style,
                    label=label,
                    linewidth=1.5,
                )
                band = ax.fill_between(
                    s.layers,
                    low[:, j] * scale,
                    high[:, j] * scale,
                    color=color,
                    alpha=0.075,
                )
                if optional:
                    line.set_gid(f"other-cwe-{uid}-{col}-{row}-{k}-line")
                    band.set_gid(f"other-cwe-{uid}-{col}-{row}-{k}-band")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(fontsize=6, loc="best")
                for k, label in enumerate(labels):
                    if label in ("Media altre CWE", "Massimo altre CWE"):
                        legend.get_texts()[k].set_gid(
                            f"other-cwe-{uid}-{col}-{row}-{k}-label"
                        )
                        legend.get_lines()[k].set_gid(
                            f"other-cwe-{uid}-{col}-{row}-{k}-key"
                        )
    fig.suptitle("Gruppi nella stessa classe · " + plot_context(s), fontsize=11)
    fig.tight_layout()
    return (
        '<div class="outcome-groups">'
        + figure_svg(fig)
        + '<p class="caption">Ogni colonna contiene una sola classe del verdetto. '
        "I colori identificano i gruppi: safe verde, vuln rosso, CWE attesa blu, union viola. "
        "Massa continua, SG tratteggiato; scale automatiche sulle serie visibili; scala comune per riga attivabile nei controlli. "
        "Bande IC95 bootstrap per coppia/quartetto. Le altre CWE sono riassunte per esempio e layer, "
        "poi mediate: non vengono sommate. Massimo massa e massimo SG possono riferirsi a CWE diverse. "
        "Gruppi sovrapposti e di dimensioni diverse: le curve non sono componenti da sommare. "
        "Union e generic coincidono solo quando verificato dal vocabolario compilato.</p></div>"
    )


def outcome_cards(s):
    labels = {
        "TP": "GT vulnerable · pred vulnerable",
        "TN": "GT safe · pred safe",
        "FP": "GT safe · pred vulnerable",
        "FN": "GT vulnerable · pred safe",
    }
    return (
        '<div class="outcome-cards">'
        + "".join(
            '<div style="border-top:5px solid '
            + COLORS[k]
            + '"><b>'
            + k
            + " · n="
            + str(int(s.select("outcome", k).sum()))
            + "</b><br><small>"
            + labels[k]
            + "</small></div>"
            for k in ("TP", "TN", "FP", "FN")
        )
        + "</div>"
    )


def cwe_competition_plot(s):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5))
    for col, gt in enumerate(("safe", "vulnerable")):
        center, lo, hi, n = s.estimate("gt", gt)
        for row, specs in enumerate(
            (
                (
                    ("mass_cwe", "Attesa", "#1565c0", "-"),
                    ("mass_other_cwe_mean", "Media altre", "#708090", "--"),
                    ("mass_other_cwe_max", "Massimo altre", "#e87917", ":"),
                ),
                (
                    ("sg_cwe", "Attesa", "#7b2cbf", "--"),
                    ("sg_other_cwe_mean", "Media altre", "#708090", ":"),
                    ("sg_other_cwe_max", "Massimo altre", "#a0522d", "-."),
                ),
            )
        ):
            ax = axs[row, col]
            scale = 100 if row == 0 else 1
            for metric, label, color, style in specs:
                j = M[metric]
                ax.plot(
                    s.layers,
                    center[:, j] * scale,
                    label=label,
                    color=color,
                    ls=style,
                    lw=1.7,
                )
                ax.fill_between(
                    s.layers,
                    lo[:, j] * scale,
                    hi[:, j] * scale,
                    color=color,
                    alpha=0.10,
                )
            ax.set_title(
                f"GT {gt} (n={n}) · "
                + ("massa esatta (%)" if row == 0 else "SG (logit)"),
                fontsize=10,
            )
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            ax.set_facecolor("#f6f8fd" if row else "#fffdf7")
    fig.suptitle("CWE attesa contro le altre CWE · " + plot_context(s), fontsize=11)
    fig.tight_layout()
    return figure_svg(fig)


def cwe_competition_table(s):
    rows = []
    for field, cohorts in (
        ("gt", ("safe", "vulnerable")),
        ("outcome", ("TP", "TN", "FP", "FN")),
    ):
        for cohort in cohorts:
            ce, lo, hi, n = s.estimate(field, cohort, collapse=True)
            if not n:
                continue
            rows.append(
                [
                    cohort,
                    n,
                    fmt(ce[0, M["mass_cwe"]], True),
                    fmt(ce[0, M["mass_other_cwe_mean"]], True),
                    fmt(ce[0, M["mass_other_cwe_max"]], True),
                    f"{fmt(ce[0, M['d_cwe_mass']] * 100)} [{fmt(lo[0, M['d_cwe_mass']] * 100)}, {fmt(hi[0, M['d_cwe_mass']] * 100)}]",
                    f"{fmt(ce[0, M['d_cwe_sg']])} [{fmt(lo[0, M['d_cwe_sg']])}, {fmt(hi[0, M['d_cwe_sg']])}]",
                    fmt(ce[0, M["cwe_rank_mass"]]),
                    fmt(ce[0, M["cwe_rank_sg"]]),
                ]
            )
    return table(
        [
            "Classe",
            "n",
            "Massa attesa",
            "Media altre",
            "Massimo altre",
            "Attesa−max altre (pp) [IC95]",
            "SG attesa−max altre [IC95]",
            "Rank massa (1=migliore)",
            "Rank SG",
        ],
        rows,
    )


def cwe_alternatives_table(s):
    rows = []
    for target in sorted({x["cwe"] for x in s.cases}):
        ids = [x["case_id"] for x in s.cases if x["cwe"] == target]
        values_by_group = defaultdict(list)
        for cid in ids:
            for group, values in (
                s.run.get("other_cwes", {}).get((cid, s.anchor), {}).items()
            ):
                values_by_group[group].append(values)
        averages = {group: mean(v, axis=0) for group, v in values_by_group.items()}
        top = sorted(
            averages,
            key=lambda g: (
                -averages[g][0] if np.isfinite(averages[g][0]) else float("inf")
            ),
        )[:3]
        for group in top:
            mass, sg = averages[group]
            rows.append([target, len(ids), group, fmt(mass, True), fmt(sg)])
    return table(
        ["CWE attesa", "n esempi", "Altra CWE", "Massa media", "SG medio"], rows
    )


def error_plot(s):
    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    for ax, metric, title, scale in zip(
        axs.flat,
        ("d_mass", "d_sg", "discord_mass", "discord_sg"),
        (
            "Massa vuln − safe (punti percentuali)",
            "SG vuln − safe (logit)",
            "Discordanza massa / verdetto (%)",
            "Discordanza SG / verdetto (%)",
        ),
        (100, 1, 100, 100),
        strict=True,
    ):
        curve(ax, s, metric, "outcome", ("TP", "TN", "FP", "FN"), scale)
        ax.set_title(title, fontsize=9)
        if metric.startswith("d_"):
            ax.axhline(0, color="gray", linewidth=0.7)
        else:
            ax.set_ylim(-3, 103)
    absent = [
        f"{c}=0" for c in ("TP", "TN", "FP", "FN") if not s.select("outcome", c).sum()
    ]
    fig.suptitle(
        "Confronti fra esiti · "
        + plot_context(s)
        + (" · " + ", ".join(absent) if absent else ""),
        fontsize=10,
    )
    fig.tight_layout()
    return figure_svg(fig)


def uncertainty_plot(s):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, metric, title, scale in zip(
        axs,
        ("ab_entropy_bits", "chosen_conditional", "pair_probability"),
        (
            "Entropia binaria A/B (bit)",
            "p(verdetto emesso | A o B)",
            "Probabilità assoluta p(A)+p(B) (%)",
        ),
        (1, 1, 100),
        strict=True,
    ):
        curve(ax, s, metric, "outcome", ("TP", "TN", "FP", "FN"), scale)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Incertezza A/B · " + plot_context(s), fontsize=10)
    fig.tight_layout()
    return figure_svg(fig)


def align_positions(run, summaries, a, b):
    """Pair identical cases and identical layers, never pool unrelated observations."""
    x, y = summaries[(run["name"], a)], summaries[(run["name"], b)]
    ids = sorted(set(x.ids) & set(y.ids))
    layers = sorted(set(x.layers) & set(y.layers))
    if not ids or not layers:
        return ids, layers, None, None
    xx = x.x[np.ix_([x.ids.index(c) for c in ids], [x.layers.index(l) for l in layers])]
    yy = y.x[np.ix_([y.ids.index(c) for c in ids], [y.layers.index(l) for l in layers])]
    valid = np.isfinite(xx) & np.isfinite(yy)
    return ids, layers, np.where(valid, xx, np.nan), np.where(valid, yy, np.nan)


def position_comparison_plot(run, summaries, gt):
    specs = [
        ("Safe", "mass_safe", "sg_safe"),
        ("Vuln", "mass_vuln", "sg_vuln"),
        ("CWE attesa", "mass_cwe", "sg_cwe"),
        ("Union", "mass_union", "sg_union"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5))
    # Each position has one stable color here; GT and outcome plots use other palettes.
    colors = plt.get_cmap("tab10")
    for i, anchor in enumerate(run["anchors"]):
        summary = summaries[(run["name"], anchor)]
        center, lo, hi, n = summary.estimate("gt", gt)
        if not n:
            continue
        color = colors(i % 10)
        for col, (_label, mass, sg) in enumerate(specs):
            for row, metric in enumerate((mass, sg)):
                ax, j, scale = axs[row, col], M[metric], 100 if row == 0 else 1
                ax.plot(
                    summary.layers,
                    center[:, j] * scale,
                    color=color,
                    ls="-" if row == 0 else "--",
                    label=f"{anchor} (n={n})",
                )
                ax.fill_between(
                    summary.layers,
                    lo[:, j] * scale,
                    hi[:, j] * scale,
                    color=color,
                    alpha=0.08,
                )
    for col, (label, _, _) in enumerate(specs):
        for row in range(2):
            ax = axs[row, col]
            ax.set_title(label + (" · massa (%)" if row == 0 else " · SG (logit)"))
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.2)
            ax.set_facecolor("#f7faf8" if row == 0 else "#faf6fc")
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6)
    model = run["meta"].get("model", run["meta"].get("hf_id", ""))
    fig.suptitle(
        f"Posizioni a confronto · {model} · {run['variant'].upper()} · GT {gt}"
    )
    fig.tight_layout()
    return figure_svg(fig)


def position_scatter(run, summaries):
    anchors = run["anchors"]
    if len(anchors) < 2:
        return ""
    origin = "code_end" if "code_end" in anchors else anchors[0]
    targets = [a for a in anchors if a != origin]
    fig, axs = plt.subplots(
        1, len(targets), figsize=(max(6, 4 * len(targets)), 4), squeeze=False
    )
    for ax, target in zip(axs[0], targets, strict=True):
        ids, layers, x, y = align_positions(run, summaries, origin, target)
        if x is None:
            continue
        xx, yy = mean(x[:, :, M["d_sg"]], axis=1), mean(y[:, :, M["d_sg"]], axis=1)
        for outcome in ("TP", "TN", "FP", "FN"):
            mask = (
                np.array([run["cases"][c]["outcome"] == outcome for c in ids])
                & np.isfinite(xx)
                & np.isfinite(yy)
            )
            ax.scatter(
                xx[mask],
                yy[mask],
                color=COLORS[outcome],
                label=f"{outcome} (n={mask.sum()})",
                alpha=0.65,
                s=20,
            )
        ax.axhline(0, color="gray", lw=0.7)
        ax.axvline(0, color="gray", lw=0.7)
        ax.set_xlabel(origin + " · SG vuln−safe")
        ax.set_ylabel(target + " · SG vuln−safe")
        ax.set_title(f"Stessi esempi · {len(layers)} layer comuni")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.15)
    fig.suptitle("Cambi di segno fra posizioni · " + run["name"], fontsize=9)
    fig.tight_layout()
    return figure_svg(fig)


def cross_position_section(runs, summaries, reps, seed, eps):
    parts = [
        "<h2>Incrocio fra posizioni sugli stessi esempi</h2>",
        "<p>Le curve sovrapposte descrivono tutti gli anchor salvati. I confronti appaiati usano soltanto gli stessi case_id e i layer comuni: delta = seconda posizione − prima posizione. Nessuna ricerca del layer migliore. IC95 bootstrap per coppia/quartetto, esplorativi e non corretti per confronti multipli.</p>",
    ]
    metrics = (
        "mass_safe",
        "mass_vuln",
        "mass_cwe",
        "mass_union",
        "sg_safe",
        "sg_vuln",
        "sg_cwe",
        "sg_union",
        "d_mass",
        "d_sg",
        "d_cwe_mass",
        "d_cwe_sg",
    )
    strata = (
        [(None, None, "Tutti")]
        + [("gt", g, "GT " + g) for g in ("safe", "vulnerable")]
        + [("outcome", o, o) for o in ("TP", "TN", "FP", "FN")]
    )
    for run in runs:
        parts += [
            "<details><summary>"
            + html.escape(run["name"])
            + " · confronto posizioni</summary>"
        ]
        for gt in ("safe", "vulnerable"):
            parts.append(position_comparison_plot(run, summaries, gt))
        parts += [
            "<p>I colori identificano le posizioni; linea continua = massa, tratteggiata = SG. Le numerosità sono per anchor; la tabella successiva usa esclusivamente osservazioni appaiate.</p>",
            position_scatter(run, summaries),
            "<p>Ogni punto è un esempio, mediato sui layer comuni. Quadranti opposti indicano un cambio di segno del contrasto SG fra le due posizioni; non equivalgono a un cambiamento del verdetto generato. FP/FN sono definiti dal verdetto finale rispetto alla GT.</p>",
        ]
        delta_rows, concordance_rows = [], []
        for a, b in itertools.combinations(run["anchors"], 2):
            ids, layers, x, y = align_positions(run, summaries, a, b)
            if x is None:
                continue
            delta = y - x
            # Preserve the layer trajectory so CIs and per-layer checks share the same units.
            fake = dict(
                run,
                data={
                    (cid, "difference", l): delta[i, j]
                    for i, cid in enumerate(ids)
                    for j, l in enumerate(layers)
                },
            )
            summary = Summary(fake, "difference", reps, seed)
            for field, value, label in strata:
                center, lo, hi, n = summary.estimate(field, value, collapse=True)
                if not n:
                    continue
                for metric in metrics:
                    j = M[metric]
                    scale = (
                        100
                        if metric.startswith("mass_")
                        or metric in ("d_mass", "d_cwe_mass")
                        else 1
                    )
                    if not np.isfinite(center[0, j]):
                        continue
                    effective = sum(
                        (field is None or run["cases"][c][field] == value)
                        and np.isfinite(delta[i, :, j]).any()
                        for i, c in enumerate(ids)
                    )
                    delta_rows.append(
                        [
                            a + " → " + b,
                            label,
                            effective,
                            len(layers),
                            metric,
                            fmt(center[0, j] * scale),
                            f"[{fmt(lo[0, j] * scale)}, {fmt(hi[0, j] * scale)}]",
                            "pp" if scale == 100 else "logit",
                        ]
                    )
                selected = np.array(
                    [field is None or run["cases"][c][field] == value for c in ids]
                )
                for metric in ("d_mass", "d_sg"):
                    j = M[metric]
                    # Four exhaustive categories on finite, non-tied example/layer cells.
                    pred = np.array([run["cases"][c]["pred"] for c in ids])
                    direction = np.where(
                        pred == "vulnerable",
                        1.0,
                        np.where(pred == "safe", -1.0, np.nan),
                    )[:, None]
                    vx, vy = x[:, :, j] * direction, y[:, :, j] * direction
                    valid = (
                        np.isfinite(vx)
                        & np.isfinite(vy)
                        & (np.abs(vx) > eps)
                        & (np.abs(vy) > eps)
                    )
                    eligible = selected & valid.any(axis=1)
                    categories = (
                        (vx > eps) & (vy > eps),
                        (vx < -eps) & (vy < -eps),
                        (vx < -eps) & (vy > eps),
                        (vx > eps) & (vy < -eps),
                    )
                    rates = []
                    for cat in categories:
                        fractions = np.divide(
                            (cat & valid).sum(1),
                            valid.sum(1),
                            out=np.full(len(ids), np.nan),
                            where=valid.sum(1) > 0,
                        )
                        rates.append(fmt(mean(fractions[eligible]), True))
                    nfinite = (
                        selected[:, None] & np.isfinite(vx) & np.isfinite(vy)
                    ).sum()
                    nties = (
                        selected[:, None]
                        & np.isfinite(vx)
                        & np.isfinite(vy)
                        & ((np.abs(vx) <= eps) | (np.abs(vy) <= eps))
                    ).sum()
                    concordance_rows.append(
                        [
                            a + " → " + b,
                            label,
                            metric,
                            int(eligible.sum()),
                            *rates,
                            int(nties),
                            int(nfinite),
                        ]
                    )
        parts += [
            "<details><summary>Differenze appaiate · GT e TP/TN/FP/FN</summary>",
            table(
                [
                    "Posizioni",
                    "Classe",
                    "n casi validi",
                    "Layer comuni",
                    "Metrica",
                    "Delta medio",
                    "IC95",
                    "Unità",
                ],
                delta_rows,
            ),
            "</details>",
            "<details><summary>Accordo con il verdetto: persistenza e cambi fra posizioni</summary><p>Percentuali medie per esempio sui layer comuni, usando il segno del contrasto vuln−safe. Entrambe concordi/discordi si riferisce al verdetto emesso, non alla GT. Le due transizioni distinguono quale posizione contraddice il verdetto. Pareggi e verdetti NC esclusi dalle percentuali; colonne finali riportano celle esempio×layer prima dell’esclusione dei pareggi. Sono misure descrittive, non prove di convinzioni interne o causalità.</p>",
            table(
                [
                    "Posizioni",
                    "Classe",
                    "Contrasto",
                    "n casi",
                    "Entrambe concordi",
                    "Entrambe discordi",
                    "Discorde → concorde",
                    "Concorde → discorde",
                    "Celle pari",
                    "Celle finite",
                ],
                concordance_rows,
            ),
            "</details>",
        ]
        errors = []
        for cid, case in sorted(run["cases"].items()):
            if case["outcome"] not in ("FP", "FN"):
                continue
            row = [cid, case["outcome"], case["cwe"]]
            for anchor in run["anchors"]:
                summary = summaries[(run["name"], anchor)]
                if cid not in summary.ids:
                    row += ["—"] * 4
                    continue
                v = mean(summary.x[summary.ids.index(cid)], axis=0)
                row += [
                    fmt(v[M["d_mass"]] * 100),
                    fmt(v[M["d_sg"]]),
                    fmt(v[M["discord_sg"]], True),
                    fmt(v[M["d_cwe_mass"]] * 100),
                ]
            errors.append(row)
        headers = ["Caso", "Esito", "CWE"]
        for a in run["anchors"]:
            headers += [
                a + " · Δmassa pp",
                a + " · ΔSG",
                a + " · layer discordi SG",
                a + " · CWE−max altre pp",
            ]
        parts += [
            "<details><summary>FP/FN · tutte le posizioni affiancate per caso</summary><p>Media sui layer disponibili per ciascun anchor; differenze di copertura possono influire. Per confronti rigorosamente appaiati usare le tabelle sopra. Il contrasto Δ è sempre vuln−safe, il margine CWE è attesa−massima altra CWE.</p>",
            table(headers, errors),
            "</details></details>",
        ]
    return "\n".join(parts)


def paired_comparisons(runs, summaries, reps, seed):
    rows, notices = [], []
    protocol_keys = (
        "hf_id",
        "band",
        "condition",
        "carrier",
        "carrier_task",
        "chat_template",
        "lens_kind",
        "gpt_oss_direct_final",
        "readout_tokenization",
    )
    for a, b in itertools.combinations(runs, 2):
        if a["variant"] == b["variant"]:
            continue
        base, ext = (a, b) if a["variant"] == "base" else (b, a)
        incompatible = [
            k for k in protocol_keys if base["meta"].get(k) != ext["meta"].get(k)
        ]
        ck1, ck2 = (
            base["meta"].get("lens_checkpoint", {}),
            ext["meta"].get("lens_checkpoint", {}),
        )
        if ck1.get("sha256") != ck2.get("sha256"):
            incompatible.append("lens sha256")
        if incompatible:
            notices.append(
                "Confronto appaiato omesso: protocolli diversi fra "
                + base["name"]
                + " e "
                + ext["name"]
                + " ("
                + ", ".join(incompatible)
                + ")"
            )
            continue
        ids = sorted(base["cases"].keys() & ext["cases"].keys())
        if not ids:
            continue
        if any(
            any(
                base["cases"][c][k] != ext["cases"][c][k]
                for k in ("gt", "cwe", "cluster")
            )
            for c in ids
        ):
            raise ValueError("Inconsistent paired base/extended labels/CWE/cluster")
        for anchor in sorted(set(base["anchors"]) & set(ext["anchors"])):
            x, y = summaries[(base["name"], anchor)], summaries[(ext["name"], anchor)]
            common = sorted(set(x.ids) & set(y.ids))
            if x.layers != y.layers:
                notices.append(
                    "Layer non coincidenti: confronto appaiato omesso per " + anchor
                )
                continue
            diff = mean(y.x[[y.ids.index(c) for c in common]], axis=1) - mean(
                x.x[[x.ids.index(c) for c in common]], axis=1
            )
            fake = dict(
                base,
                data={(c, anchor, 0): row for c, row in zip(common, diff, strict=True)},
            )
            summary = Summary(fake, anchor, reps, seed)
            for gt in ("safe", "vulnerable"):
                center, lo, hi, n = summary.estimate("gt", gt, collapse=True)
                for metric in (
                    "mass_safe",
                    "mass_vuln",
                    "mass_cwe",
                    "sg_safe",
                    "sg_vuln",
                    "sg_cwe",
                    "mass_union",
                    "sg_union",
                    "d_cwe_mass",
                    "d_cwe_sg",
                ):
                    j = M[metric]
                    scale = (
                        100
                        if metric.startswith("mass_") or metric == "d_cwe_mass"
                        else 1
                    )
                    rows.append(
                        [
                            anchor,
                            gt,
                            n,
                            metric,
                            fmt(center[0, j] * scale),
                            f"[{fmt(lo[0, j] * scale)}, {fmt(hi[0, j] * scale)}]",
                            ext["name"] + " − " + base["name"],
                        ]
                    )
    return rows, notices


def build_report(runs, requested, reps, seed, eps):
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 8})
    all_anchors = sorted(
        {a for r in runs for a in r["anchors"]},
        key=anchor_sort_key,
    )
    if requested:
        missing = requested - set(all_anchors)
        if missing:
            raise ValueError(
                "Requested anchors absent from every input: "
                + ", ".join(sorted(missing))
            )
    summaries = {
        (r["name"], a): Summary(r, a, reps, seed) for r in runs for a in r["anchors"]
    }
    audit_rows, counts_rows = [], []
    for r in runs:
        counts = Counter(c["outcome"] for c in r["cases"].values())
        n = len(r["cases"])
        counts_rows.append(
            [
                r["name"],
                r["variant"],
                n,
                *[counts[c] for c in ("TP", "TN", "FP", "FN", "NC")],
                fmt((counts["TP"] + counts["TN"]) / n, True),
                sum(c["truncated"] for c in r["cases"].values()),
            ]
        )
        for a in all_anchors:
            available = sum(
                any((cid, a, l) in r["data"] for l in Summary.layers_for(r, a))
                for cid in r["cases"]
            )
            audit_rows.append([r["name"], a, available, n - available])
    parts = [
        '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MCP · analisi compatta</title><style>body{font:15px system-ui;max-width:1250px;margin:30px auto;padding:0 18px;color:#1a2b40}h1,h2,h3{color:#173f65}svg{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;margin:16px 0}th,td{padding:7px;border-bottom:1px solid #dbe1e8;text-align:right}th:first-child,td:first-child{text-align:left}summary{cursor:pointer;padding:10px;background:#f0f4f8}details{margin:15px 0}.note{background:#edf4fb;padding:15px;border-left:4px solid #3476ac}.toolbar{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #ddd;z-index:2}select,button{font:inherit;padding:7px}.anchor-panel{display:none}.anchor-panel.active{display:block}.caption{color:#536273;font-size:13px}.outcome-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:15px 0}.outcome-cards>div{padding:12px;background:#f6f7fa;border-radius:5px}.outcome-cards b{font-size:19px}.outcome-cards small{font-size:12px}h3{margin-top:30px}@media(max-width:700px){.outcome-cards{grid-template-columns:repeat(2,1fr)}}.interactive-chart{margin:18px 0}.chart-controls{padding:10px 12px;background:#f4f7fa;border:1px solid #dce3eb;border-radius:6px}.chart-top-controls,.chart-line-controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:8px}.chart-top-controls label,.chart-choice{display:inline-flex;align-items:center;gap:5px}.chart-choice{background:white;border:1px solid #d5dce5;border-radius:4px;padding:4px 7px;font-size:13px}.chart-swatch{width:17px;height:3px;display:inline-block}.chart-controls small{color:#566579}.chart-controls button{font-size:12px;padding:4px 8px;margin:0 3px}.chart-controls select{max-width:360px;font-size:13px}.chart-canvas{overflow:auto}.chart-canvas svg{display:block;min-width:650px}</style>',
        "<h1>Massa, SG, incertezza e discordanze</h1>",
        "<p>Analisi CPU su statistiche già estratte. Un solo documento offline; nessun caricamento dei modelli, nessuna heatmap o matrice di confusione grafica.</p>",
        "<h2>Qualità e verdetti</h2>",
        table(
            [
                "Run",
                "Dataset",
                "n",
                "TP",
                "TN",
                "FP",
                "FN",
                "NC",
                "Accuracy su tutti",
                "Troncati",
            ],
            counts_rows,
        ),
        "<p>NC indica verdetti mancanti/non validi e rimane nel denominatore dell’accuracy. I grafici riportano le numerosità effettivamente disponibili; n può cambiare se mancano anchor o layer. I gruppi con n=0 non hanno una curva.</p>",
        "<details><summary>Copertura degli anchor e avvisi</summary>",
        table(
            ["Run", "Anchor", "Esempi con lens esatta", "Mancanti/non esatti"],
            audit_rows,
        ),
    ]
    for r in runs:
        if r["audit"]:
            parts.append(
                "<p>"
                + html.escape(
                    r["name"] + ": " + json.dumps(r["audit"], ensure_ascii=False)
                )
                + "</p>"
            )
    parts += [
        "</details>",
        '<div class="note"><b>Due tipi distinti di incertezza.</b> Le bande dei grafici sono IC95% bootstrap delle coppie/quartetti: descrivono l’incertezza della media sul dataset. L’entropia A/B descrive invece quanto la lettura distribuisce sostegno fra le due lettere. Non è probabilità di correttezza.</div>',
        "<p>Massa esatta: somma delle probabilità dei token del gruppo, normalizzate sull’intero vocabolario. SG: logit medio del gruppo meno logit medio globale. SG_z: SG diviso per la deviazione standard globale della singola lettura. Tutte le analisi usano il profilo full e finestra 1; nessuna massa completa viene ricostruita dai top-k.</p>",
        "<p>Si mostrano tutti i layer disponibili. Bootstrap pointwise per pair_id/quartet_id (case_id solo se assenti), mai sui layer come campioni indipendenti. Posizioni/layer scelti dopo aver visto questi dati restano esplorativi: gli intervalli non correggono la selezione multipla. Nessun AUROC o soglia ottimizzata è introdotto automaticamente. Il riepilogo include anche probabilità del top-1 salvato e quota di letture con top-1 fuori dai vocabolari, quando disponibili, senza esportare una tabella per ogni token. Se il ranking J++ è filtrato, top-1 indica il primo token di quel ranking.</p>",
        '<div class="toolbar"><label for="anchor">Posizione: </label><select id="anchor" onchange="showAnchor(this.value)">'
        + "".join(
            f'<option value="p{j}">{html.escape(a)}</option>'
            for j, a in enumerate(all_anchors)
        )
        + '</select> <button onclick="downloadCases()">Scarica dati per esempio (CSV)</button></div>',
    ]
    same_groups = all(r.get("generic_equals_union", False) for r in runs)
    parts.append(
        '<p class="note"><b>Union e generic.</b> '
        + (
            "Gli insiemi di token coincidono nel vocabolario compilato selezionato: qui Union = generic. Mostro una sola curva Union, senza duplicarla."
            if same_groups
            else "Gli insiemi di token NON coincidono nel vocabolario compilato: Union è la loro unione deduplicata; generic resta separato nel riepilogo e nel CSV."
        )
        + "</p>"
    )
    parts.append(
        "<p><b>Guida ai grafici:</b> i pannelli GT usano verde/rosso per la massa e azzurro/ocra per SG; i pannelli TP/TN/FP/FN hanno palette dedicate, conteggi e sfondi propri. Il confronto CWE usa colori per attesa/media/massimo altre, con legende esplicite. SG è tratteggiato anche quando i colori non sono distinguibili.</p>"
    )
    exported, error_rows, cwe_rows = [], [], []
    for index, anchor in enumerate(all_anchors):
        parts.append(
            f'<section id="p{index}" class="anchor-panel'
            + (" active" if index == 0 else "")
            + '"><h2>'
            + html.escape(anchor)
            + "</h2>"
        )
        for run in runs:
            s = summaries.get((run["name"], anchor))
            if s is None or not s.layers or not s.ids:
                continue
            parts += [
                "<h3>" + html.escape(run["name"]) + "</h3>",
                gt_plot(s),
                '<p class="caption">GT safe/vuln: massa esatta e SG. Le classi sono definite dalla ground truth. La CWE attesa è quella del caso anche per le patch: riconoscere il meccanismo non significa riconoscere una vulnerabilità.</p>',
                "<h3>Classi del verdetto: TP, TN, FP, FN</h3>",
                outcome_cards(s),
                "<h3>Una classe alla volta: gruppi safe, vuln, CWE e union sovrapposti</h3>",
                outcome_group_plot(s),
                "<details><summary>Confronto complementare: TP/TN/FP/FN nello stesso gruppo</summary>",
                plot_grid(s, "outcome"),
                "</details>",
                '<p class="caption">Confronto diretto delle quattro classi per safe, vuln, CWE attesa e union. Ogni legenda riporta n. Massa: linee continue e fondo caldo; SG: tratteggio, toni diversi e fondo azzurro.</p>',
                "<h3>CWE attesa rispetto alle altre CWE</h3>",
                cwe_competition_plot(s),
                "<p>Per ogni esempio e layer, confronto il gruppo atteso con la media degli altri gruppi CWE e con il massimo degli altri; poi medio fra esempi. Il massimo può riferirsi a CWE diverse da esempio a esempio. I rank usano pari merito medi e sono calcolati prima di mediare layer/esempi. Non sommo le masse delle altre CWE, perché i gruppi possono sovrapporsi. I gruppi possono avere dimensioni diverse: massa e SG offrono confronti complementari, non una classificazione validata.</p>",
                "<details><summary>Numeri CWE attesa / altre, anche per TP/TN/FP/FN</summary>",
                cwe_competition_table(s),
                "</details>",
                "<details><summary>Quali altre CWE emergono? Tre alternative per CWE attesa</summary><p>Classifica descrittiva per massa media: prima media dei layer per esempio, poi degli esempi con la stessa CWE attesa. Non è una matrice di errori del modello e non somma gruppi sovrapposti. Eventuali pari merito non dimostrano preferenza per una CWE.</p>",
                cwe_alternatives_table(s),
                "</details>",
                "<details><summary>Contrasti e tassi di discordanza per layer</summary>",
                error_plot(s),
                "<p>Contrasti vuln−safe: sopra zero prevale il gruppo vuln. Discordanza: il segno del contrasto è opposto al verdetto emesso. I pareggi entro "
                + str(eps)
                + " sono esclusi dal tasso e non contati come concordanze. Gruppi di dimensioni diverse, sovrapposizioni, polisemia e negazioni limitano l’interpretazione: non è prova di una credenza interna opposta.</p></details>",
            ]
            coverage = []
            for gt in ("safe", "vulnerable"):
                center, lo, hi, n = s.estimate("gt", gt, collapse=True)
                for j, label in enumerate(G_LABEL):
                    coverage.append(
                        [
                            gt,
                            n,
                            label,
                            fmt(center[0, j], True),
                            f"[{fmt(lo[0, j], True)}, {fmt(hi[0, j], True)}]",
                            fmt(center[0, j + 3]),
                            fmt(center[0, j + 6]),
                        ]
                    )
                coverage.append(
                    [
                        gt,
                        n,
                        "Union",
                        fmt(center[0, M["mass_union"]], True),
                        f"[{fmt(lo[0, M['mass_union']], True)}, {fmt(hi[0, M['mass_union']], True)}]",
                        fmt(center[0, M["sg_union"]]),
                        fmt(center[0, M["sgz_union"]]),
                    ]
                )
                coverage.append(
                    [
                        gt,
                        n,
                        "Generico / fuori union",
                        fmt(center[0, M["mass_generic"]], True),
                        "Fuori: " + fmt(center[0, M["mass_outside"]], True),
                        fmt(center[0, M["sg_generic"]]),
                        fmt(center[0, M["sgz_generic"]]),
                    ]
                )
            for gt in ("safe", "vulnerable"):
                center, _, _, n = s.estimate("gt", gt, collapse=True)
                coverage.append(
                    [
                        gt,
                        n,
                        "Top-1 (se disponibile)",
                        fmt(center[0, M["top1_probability"]], True),
                        "Top-1 fuori vocabolari: "
                        + fmt(center[0, M["top1_outside"]], True),
                        "—",
                        "—",
                    ]
                )
            parts += [
                "<details><summary>Numeri riassuntivi: prima media dei layer per esempio, poi fra esempi</summary>",
                table(
                    ["GT", "n", "Gruppo", "Massa", "IC95 massa", "SG", "SG_z"], coverage
                ),
                "</details>",
            ]
            if anchor in ("mcp_pre_verdict", "pre_verdict", "pre_decision"):
                parts += [
                    "<h3>Incertezza vicino alla decisione</h3>",
                    uncertainty_plot(s),
                    "<p>Entropia binaria in bit, probabilità condizionata del verdetto emesso e massa assoluta A+B. Una probabilità condizionata elevata con A+B molto piccolo va interpretata con cautela. L’entropia dell’intero vocabolario non è ricostruibile esattamente da massa, SG e top-k; non viene inventata.</p>",
                ]
            for n, cid in enumerate(s.ids):
                c = run["cases"][cid]
                avg = mean(s.x[n], axis=0)
                final = run["data"].get(
                    (cid, anchor, -1), np.full(len(METRICS), np.nan)
                )
                row = {
                    "run": run["name"],
                    "anchor": anchor,
                    **c,
                    "n_layers": int(np.isfinite(s.x[n, :, 0]).sum()),
                }
                row.update(
                    {
                        k: float(v) if np.isfinite(v) else None
                        for k, v in zip(METRICS, avg, strict=True)
                    }
                )
                row.update(
                    {
                        "model_" + k: float(final[M[k]])
                        if np.isfinite(final[M[k]])
                        else None
                        for k in (
                            "ab_entropy_bits",
                            "chosen_conditional",
                            "pair_probability",
                            "chosen_margin",
                        )
                    }
                )
                exported.append(row)
                if c["outcome"] in ("FP", "FN"):
                    error_rows.append(
                        [
                            run["name"],
                            anchor,
                            cid,
                            c["outcome"],
                            *[
                                fmt(
                                    avg[M[k]],
                                    k.startswith(("mass_", "discord_")),
                                )
                                for k in (
                                    "mass_safe",
                                    "mass_vuln",
                                    "mass_cwe",
                                    "d_sg",
                                    "discord_sg",
                                )
                            ],
                            fmt(final[M["chosen_conditional"]])
                            if anchor
                            in ("mcp_pre_verdict", "pre_verdict", "pre_decision")
                            else "—",
                        ]
                    )
            for cwe in sorted({x["cwe"] for x in s.cases}):
                selected = s.select("cwe", cwe)
                n = int(selected.sum())
                cen = mean(mean(s.x[selected], axis=1), axis=0)[None, :]
                cwe_rows.append(
                    [
                        run["name"],
                        anchor,
                        cwe,
                        n,
                        *[fmt(cen[0, j], True) for j in range(3)],
                        fmt(cen[0, 5]),
                    ]
                )
        parts.append("</section>")
    # Final-model uncertainty: one small histogram per run, only at pre-verdict.
    uncertainty_rows = []
    fig, axs = plt.subplots(
        1, len(runs), figsize=(max(5, 5 * len(runs)), 3.5), squeeze=False
    )
    found = False
    for ax, run in zip(axs.flat, runs, strict=True):
        selected_anchor = next(
            (
                a
                for a in ("mcp_pre_verdict", "pre_verdict", "pre_decision")
                if a in run["anchors"]
            ),
            None,
        )
        if selected_anchor is None:
            ax.text(0.1, 0.5, "Pre_verdict non disponibile")
            continue
        for correct, col, label in (
            (True, "#18864b", "corretti"),
            (False, "#e18b16", "errori"),
        ):
            vals = []
            for cid, c in run["cases"].items():
                v = run["data"].get((cid, selected_anchor, -1))
                if (
                    v is not None
                    and c["pred"] != "unknown"
                    and (c["gt"] == c["pred"]) == correct
                    and np.isfinite(v[M["chosen_conditional"]])
                ):
                    vals.append(v[M["chosen_conditional"]])
            if vals:
                ax.hist(
                    vals,
                    bins=np.linspace(0, 1, 11),
                    alpha=0.45,
                    color=col,
                    label=f"{label} n={len(vals)}",
                )
                found = True
        for conf in ("TP", "TN", "FP", "FN"):
            vals = [
                run["data"][(cid, selected_anchor, -1)]
                for cid, c in run["cases"].items()
                if c["outcome"] == conf and (cid, selected_anchor, -1) in run["data"]
            ]
            valid = [v for v in vals if np.isfinite(v[M["ab_entropy_bits"]])]
            uncertainty_rows.append(
                [
                    run["name"],
                    conf,
                    len(valid),
                    fmt(mean([v[M["ab_entropy_bits"]] for v in valid]))
                    if valid
                    else "—",
                    fmt(mean([v[M["chosen_conditional"]] for v in valid]))
                    if valid
                    else "—",
                    fmt(mean([v[M["pair_probability"]] for v in valid]), True)
                    if valid
                    else "—",
                ]
            )
        ax.set_title(run["name"], fontsize=8)
        ax.set_xlabel("p(verdetto emesso | A o B)")
        ax.set_ylabel("Esempi")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.tight_layout()
    parts += ["<h2>Incertezza del modello finale su pre_verdict</h2>"]
    if found:
        parts.append(figure_svg(fig))
    else:
        plt.close(fig)
        parts.append(
            "<p>Statistiche A/B del modello finale assenti. Nessuna stima sostitutiva dai top-k.</p>"
        )
    parts += [
        table(
            [
                "Run",
                "Esito",
                "n con A/B",
                "Entropia media (bit)",
                "p condizionata verdetto",
                "Massa assoluta A+B",
            ],
            uncertainty_rows,
        ),
        "<p>I logits del modello finale sono quelli salvati dalla pipeline di lettura, eventualmente ricomputati dopo la generazione. Queste quantità non sono probabilità calibrate di correttezza. I punteggi A/B sugli anchor lontani dal verdetto sono sonde controfattuali; rimangono nel CSV, ma non vengono presentati come certezza della decisione.</p>",
    ]
    parts.append(cross_position_section(runs, summaries, reps, seed, eps))
    paired, notices = paired_comparisons(runs, summaries, reps, seed)
    parts += [
        "<details><summary>Base/extended: differenze sugli stessi esempi</summary><p>Delta extended−base dopo la media sui layer per esempio, con bootstrap delle coppie. Masse in punti percentuali, SG in logit; si confrontano soltanto protocolli compatibili.</p>",
        table(
            ["Anchor", "GT", "n appaiati", "Metrica", "Delta", "IC95", "Confronto"],
            paired,
        ),
    ]
    parts += ["<p>" + html.escape(n) + "</p>" for n in notices]
    parts += [
        "</details>",
        "<details><summary>FP/FN: tutti gli esempi, senza un grafico per ciascuno</summary>",
        table(
            [
                "Run",
                "Anchor",
                "Caso",
                "Esito",
                "Massa safe",
                "Massa vuln",
                "Massa CWE",
                "SG vuln−safe",
                "Layer discordanti SG",
                "p verdetto finale A/B",
            ],
            error_rows,
        ),
        "</details>",
        "<details><summary>Per CWE: riepilogo descrittivo, senza heatmap</summary><p>Campioni piccoli: nessuna banda bootstrap per CWE. Le masse di gruppi sovrapposti non vanno sommate.</p>",
        table(
            ["Run", "Anchor", "CWE", "n", "Safe", "Vuln", "CWE attesa", "SG CWE"],
            cwe_rows,
        ),
        "</details>",
    ]
    manifest = {
        "version": VERSION,
        "bootstrap": reps,
        "seed": seed,
        "window": 1,
        "profile": "full",
        "inputs": [
            {
                "path": r["path"],
                "model": r["meta"].get("hf_id"),
                "layers": r["meta"].get("band"),
                "lens": r["meta"].get("lens_checkpoint"),
                "vocabulary_signature": r["meta"]
                .get("vocab_scores", {})
                .get("group_signature"),
            }
            for r in runs
        ],
    }
    parts += [
        "<details><summary>Riproducibilità e definizioni</summary><p>SG = media logits gruppo − media logits intero vocabolario. Contrasto SG = SG_vuln − SG_safe. Discordanza lessicale calcolata separatamente dal sostegno A/B. Assenza di un segnale lessicale non dimostra assenza della relativa informazione interna.</p><pre>"
        + html.escape(json.dumps(manifest, indent=2, ensure_ascii=False))
        + "</pre></details>"
    ]
    payload = json.dumps(exported, ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    parts += [
        '<script id="case-data" type="application/json">' + payload + "</script>",
        """<script>
function showAnchor(id){document.querySelectorAll('.anchor-panel').forEach(x=>x.classList.toggle('active',x.id===id));}
function downloadCases(){const rows=JSON.parse(document.getElementById('case-data').textContent);if(!rows.length)return;const h=Object.keys(rows[0]);function q(x){let s=x==null?'':String(x);if(/^[=+@]/.test(s)||/^-\\D/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';}const csv=[h.map(q).join(','),...rows.map(r=>h.map(k=>q(r[k])).join(','))].join('\\r\\n');const u=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='case_metrics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
</script></html>""",
    ]
    parts.insert(-1, "<script>" + INTERACTIVE_JS + "</script>")
    if reps == 0:
        parts.insert(
            2,
            '<p class="note">Bootstrap disabilitato: nessun intervallo di confidenza calcolato.</p>',
        )
    return "\n".join(parts)


INTERACTIVE_JS = "(function(){\n'use strict';\nconst NS='http://www.w3.org/2000/svg';\nfunction el(tag,attrs={},text){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!=null)e.textContent=text;return e;}\nfunction finite(x){return typeof x==='number'&&Number.isFinite(x);}\nfunction niceTicks(lo,hi,count=5){const raw=(hi-lo)/count;if(!(raw>0))return[lo];const base=10**Math.floor(Math.log10(raw));const ratio=raw/base;const step=(ratio<=1?1:ratio<=2?2:ratio<=5?5:10)*base;const a=[];for(let i=Math.ceil(lo/step);i*step<=hi+step*1e-8&&a.length<30;i++)a.push(i*step);return a;}\nfunction format(v){if(Math.abs(v)<1e-12)return'0';if(Math.abs(v)>=10000||Math.abs(v)<.001)return v.toExponential(1);return String(Number(v.toPrecision(4)));}\nfunction extent(values,fallback){const valid=values.filter(finite);if(!valid.length)return[...fallback];let lo=Math.min(...valid),hi=Math.max(...valid);const pad=hi>lo?(hi-lo)*.07:Math.max(Math.abs(lo)*.08,.01);return[lo-pad,hi+pad];}\nfunction pointsPath(points,sx,sy,close=false){let text='',pen=false;for(const p of points){if(!finite(p[0])||!finite(p[1])){pen=false;continue;}text+=(pen?' L ':'M ')+sx(p[0])+' '+sy(p[1]);pen=true;}return text+(close&&text?' Z':'');}\nlet nextId=0;\nfunction init(root){if(root.dataset.initialized)return;root.dataset.initialized='1';const d=JSON.parse(root.querySelector('.chart-spec').textContent);const canvas=root.querySelector('.chart-canvas');const controls=root.querySelector('.chart-controls');const id=++nextId;let scope='all',showBands=true,shared=false;\nconst initial=d.panels.map(p=>p.series.map(s=>s.visible));\nconst top=document.createElement('div');top.className='chart-top-controls';\nconst scopeLabel=document.createElement('label');scopeLabel.append('Seleziona linee per: ');const select=document.createElement('select');const all=new Option('Tutto il gruppo di grafici','all');select.add(all);d.panels.forEach((p,i)=>select.add(new Option('Pannello '+(i+1)+' — '+(p.title||p.ylabel||p.xlabel),String(i))));scopeLabel.append(select);top.append(scopeLabel);\nfunction checkbox(text,checked,fn){const l=document.createElement('label'),c=document.createElement('input');c.type='checkbox';c.checked=checked;c.onchange=()=>fn(c.checked);l.append(c,' '+text);top.append(l);return c;}\ncheckbox('Bande IC95',true,v=>{showBands=v;render();});\nif(d.sharedY.length)checkbox('Stessa scala Y per riga',false,v=>{shared=v;render();});\nconst buttons=document.createElement('span');for(const[label,action]of[['Tutte',true],['Nessuna',false],['Ripristina','reset']]){const b=document.createElement('button');b.type='button';b.textContent=label;b.onclick=()=>{indices().forEach(i=>d.panels[i].series.forEach((s,j)=>s.visible=action==='reset'?initial[i][j]:action));refresh();render();};buttons.append(b);}top.append(buttons);controls.append(top);\nconst choices=document.createElement('div');choices.className='chart-line-controls';controls.append(choices);\nconst hint=document.createElement('small');hint.textContent='Scala automatica sulle serie visibili e sulle bande attive. I valori e le statistiche non cambiano.';controls.append(hint);\nfunction indices(){return scope==='all'?d.panels.map((_,i)=>i):[Number(scope)];}\nfunction refresh(){choices.replaceChildren();const groups=new Map();for(const i of indices())for(const s of d.panels[i].series){if(!groups.has(s.key))groups.set(s.key,[]);groups.get(s.key).push(s);}for(const[key,ss]of groups){const label=document.createElement('label');label.className='chart-choice';const c=document.createElement('input');c.type='checkbox';c.checked=ss.every(s=>s.visible);c.indeterminate=ss.some(s=>s.visible)&&!c.checked;const swatch=document.createElement('span');swatch.className='chart-swatch';swatch.style.background=ss[0].color;label.append(c,swatch,document.createTextNode(key));c.onchange=()=>{ss.forEach(s=>s.visible=c.checked);refresh();render();};choices.append(label);}}\nselect.onchange=()=>{scope=select.value;refresh();};\nfunction bounds(p){const xs=[],ys=[];for(const s of p.series.filter(s=>s.visible)){if(s.kind==='bars'){for(const[x,y,w,h]of s.rects){xs.push(x,x+w);ys.push(y,y+h);}}else{for(const[x,y]of s.points){if(finite(x)&&finite(y)){xs.push(x);ys.push(y);}}if(showBands&&s.bands)for(const band of s.bands)for(const[x,y]of band)if(finite(x)&&finite(y)){xs.push(x);ys.push(y);}}}return{x:extent(xs,p.xlim),y:extent(ys,p.ylim)};}\nfunction render(){const svg=el('svg',{viewBox:'0 0 '+d.width+' '+d.height,role:'img','aria-label':d.title,'data-dynamic':'true','font-family':'system-ui, sans-serif'});svg.append(el('rect',{x:0,y:0,width:d.width,height:d.height,fill:'white'}));const defs=el('defs');svg.append(defs);svg.append(el('text',{x:d.width/2,y:18,'text-anchor':'middle','font-size':14,'font-weight':'600'},d.title));const domains=d.panels.map(bounds);if(shared)for(const group of d.sharedY){const active=group.filter(i=>d.panels[i].series.some(s=>s.visible));if(!active.length)continue;const lo=Math.min(...active.map(i=>domains[i].y[0])),hi=Math.max(...active.map(i=>domains[i].y[1]));group.forEach(i=>domains[i].y=[lo,hi]);}\nfor(let pi=0;pi<d.panels.length;pi++){const p=d.panels[pi],b=domains[pi],[x,y,w,h]=p.rect;const sx=v=>x+(v-b.x[0])/(b.x[1]-b.x[0])*w,sy=v=>y+h-(v-b.y[0])/(b.y[1]-b.y[0])*h;const active=p.series.filter(s=>s.visible);const ax=el('g',{'data-panel':pi,'data-ymin':b.y[0],'data-ymax':b.y[1],'data-xmin':b.x[0],'data-xmax':b.x[1]});svg.append(ax);ax.append(el('rect',{x,y,width:w,height:h,fill:p.background,stroke:'#687787','stroke-width':.7}));const clipId='chart-'+id+'-clip-'+pi;const clip=el('clipPath',{id:clipId});clip.append(el('rect',{x,y,width:w,height:h}));defs.append(clip);const plotted=el('g',{'clip-path':'url(#'+clipId+')'});ax.append(plotted);\nfor(const v of niceTicks(...b.x)){const xx=sx(v);plotted.append(el('line',{x1:xx,x2:xx,y1:y,y2:y+h,stroke:'#718096','stroke-opacity':.17}));ax.append(el('text',{x:xx,y:y+h+15,'text-anchor':'middle','font-size':10},format(v)));}\nfor(const v of niceTicks(...b.y)){const yy=sy(v);plotted.append(el('line',{x1:x,x2:x+w,y1:yy,y2:yy,stroke:'#718096','stroke-opacity':.17}));ax.append(el('text',{x:x-6,y:yy+3,'text-anchor':'end','font-size':10},format(v)));}\nfor(const ref of p.refs){if(!finite(ref.value))continue;if(ref.axis==='y'&&ref.value>=b.y[0]&&ref.value<=b.y[1])plotted.append(el('line',{x1:x,x2:x+w,y1:sy(ref.value),y2:sy(ref.value),stroke:'#768494','stroke-width':.8}));if(ref.axis==='x'&&ref.value>=b.x[0]&&ref.value<=b.x[1])plotted.append(el('line',{x1:sx(ref.value),x2:sx(ref.value),y1:y,y2:y+h,stroke:'#768494','stroke-width':.8}));}\nfor(const s of active){if(s.kind==='line'){if(showBands)for(const band of s.bands||[])plotted.append(el('path',{d:pointsPath(band,sx,sy,true),fill:s.color,'fill-opacity':.12,stroke:'none'}));const dash=s.style==='--'?'7 4':s.style===':'?'2 3':s.style==='-.'?'7 3 2 3':'';plotted.append(el('path',{d:pointsPath(s.points,sx,sy),fill:'none',stroke:s.color,'stroke-width':1.8,'stroke-dasharray':dash,'data-series':s.key}));}else if(s.kind==='scatter'){for(const[xp,yp]of s.points)if(finite(xp)&&finite(yp))plotted.append(el('circle',{cx:sx(xp),cy:sy(yp),r:2.8,fill:s.color,'fill-opacity':.65,'data-series':s.key}));}else for(const[xp,yp,ww,hh]of s.rects)plotted.append(el('rect',{x:sx(xp),y:sy(yp+hh),width:sx(xp+ww)-sx(xp),height:Math.abs(sy(yp)-sy(yp+hh)),fill:s.color,'fill-opacity':.45,'data-series':s.key}));}\nax.append(el('text',{x:x+w/2,y:y-11,'text-anchor':'middle','font-size':12,'font-weight':'600',fill:p.titleColor},p.title));ax.append(el('text',{x:x+w/2,y:y+h+32,'text-anchor':'middle','font-size':10},p.xlabel));if(p.ylabel)ax.append(el('text',{transform:'translate('+(x-46)+' '+(y+h/2)+') rotate(-90)','text-anchor':'middle','font-size':11},p.ylabel));\nif(!active.length)ax.append(el('text',{x:x+w/2,y:y+h/2,'text-anchor':'middle','font-size':12,fill:'#63758a'},p.notes.join(' · ')||'Nessuna serie selezionata'));\n// The compact legend is clickable; checkboxes above remain the accessible control.\nif(active.length){const fs=9,lw=Math.min(w-12,Math.max(75,...active.map(s=>s.label.length*fs*.53+30)));const legend=el('g');legend.append(el('rect',{x:x+5,y:y+5,width:lw,height:active.length*13+6,fill:'white','fill-opacity':.82,stroke:'#d0d7de'}));active.forEach((s,k)=>{const yy=y+16+k*13,g=el('g',{tabindex:0,role:'button','aria-label':'Nascondi '+s.label,style:'cursor:pointer'});g.append(el('line',{x1:x+10,x2:x+25,y1:yy-3,y2:yy-3,stroke:s.color,'stroke-width':2}));g.append(el('text',{x:x+30,y:yy,'font-size':fs},s.label));const hide=()=>{s.visible=false;refresh();render();};g.onclick=hide;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();hide();}};legend.append(g);});ax.append(legend);}}\ncanvas.replaceChildren(svg);}\nrefresh();render();}\nwindow.MCPCharts={init,extent,niceTicks};\nconst roots=document.querySelectorAll('.interactive-chart');if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){init(entry.target);observer.unobserve(entry.target);}},{rootMargin:'150px'});roots.forEach(r=>observer.observe(r));}else roots.forEach(init);\n})();\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--vocab",
        required=True,
        type=Path,
        help="Compiled vocabulary folder or token_groups.json; must match saved exact scores",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="One .html file, or one .zip containing report.html",
    )
    parser.add_argument(
        "--anchors", default="all", help="all, or comma-separated anchor names"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Pair/quartet replicates; 0 disables confidence intervals",
    )
    parser.add_argument("--seed", type=int, default=42017)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-9,
        help="Tie tolerance for discordance contrasts",
    )
    args = parser.parse_args()
    if args.bootstrap < 0 or not np.isfinite(args.epsilon) or args.epsilon < 0:
        parser.error("bootstrap and epsilon must be nonnegative")
    if args.out.suffix.lower() not in (".html", ".zip"):
        parser.error("--out must end with .html or .zip")
    if args.out.exists():
        parser.error("Output already exists: choose a new --out (nothing overwritten)")
    paths = []
    for p in args.inputs:
        if p.is_dir():
            parser.error(
                "Pass explicit JSONL files to avoid mixing old runs or marker pilots"
            )
        p = p.expanduser().resolve()
        if not p.is_file():
            parser.error(f"Input not found: {p}")
        paths.append(p)
    if len(set(paths)) != len(paths):
        parser.error("Duplicate input paths")
    requested = (
        None
        if args.anchors == "all"
        else {x.strip() for x in args.anchors.split(",") if x.strip()}
    )
    runs = load_inputs(paths, args.vocab.expanduser(), requested, args.epsilon)
    report = build_report(runs, requested, args.bootstrap, args.seed, args.epsilon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".zip":
        with zipfile.ZipFile(
            args.out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.writestr("report.html", report)
    else:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(report)
    print(
        f"Saved {args.out}: {args.out.stat().st_size / 1024**2:.2f} MiB; one offline report, {len(runs)} runs. No model/GPU loaded."
    )


if __name__ == "__main__":
    main()
