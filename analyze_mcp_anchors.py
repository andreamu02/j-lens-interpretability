from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import importlib.util
import itertools
import json
import math
import sqlite3
import sys
import zlib
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "3.0-uncertainty-discordance-errors"
MODES = {
    "overview",
    "gt",
    "confusion",
    "cwe",
    "datasets",
    "pairs",
    "outside",
    "layers",
    "errors",
}
KS = (1, 5, 10, 20, 50, 100, 200, 500)
METRICS = [
    "score_sum",
    "share_observed",
    "count",
    "reciprocal_rank",
    "best_score",
    "top1_contribution",
    "entropy_conditional",
    "effective_tokens",
]
BASE = ["run", "stream", "window", "profile", "layer", "group"]
_CATEGORIES = [
    "safety",
    "vuln",
    "generic",
    "union",
    "outside",
    "safety_only",
    "vuln_only",
    "polarity_overlap",
    "generic_only",
    "cwe_union",
    "expected_cwe",
    "other_cwe",
    "expected_cwe_exclusive",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield lineno, json.loads(line)
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: {e}") from e


def save(df, name, out, fmt):
    target = out / "tables"
    target.mkdir(exist_ok=True)
    if len(df.columns) == 0:
        df = pd.DataFrame(columns=["status"])
    if fmt in ("csv", "both"):
        df.to_csv(target / (name + ".csv.gz"), index=False, compression="gzip")
    if fmt in ("parquet", "both"):
        df.to_parquet(target / (name + ".parquet"), index=False, compression="zstd")


def normalize(value):
    return {
        "safe": "safe",
        "patched": "safe",
        "vuln": "vuln",
        "vulnerable": "vuln",
        "unsafe": "vuln",
    }.get(str(value).lower(), "unknown")


def surfaces(term):
    forms = {
        term,
        term.lower(),
        term.upper(),
        term.title(),
        term[:1].upper() + term[1:],
    }
    return {prefix + s for prefix in ("", " ") for s in forms}


class Vocabulary:
    def __init__(self, path):
        self.path = path
        self.compiled = json.loads((path / "token_groups.json").read_text())
        self.catalog = json.loads((path / "concepts.json").read_text())
        self.concepts = {c["term"]: c for c in self.catalog["concepts"]}
        self.forms = {}
        for term, concept in self.concepts.items():
            for text in surfaces(term) | set(concept.get("surface_aliases", [])):
                self.forms.setdefault(text, set()).add(term)
        self.cwes = sorted(self.catalog["meta"]["cwes"])
        self.cache = {}
        self.extra = {}

    def lookup(self, token_id, text, profiles):
        if token_id in self.cache:
            previous, groups = self.cache[token_id]
            if previous != text:
                raise ValueError(f"Superfici incompatibili per ID {token_id}")
            return groups
        row = self.compiled["tokens"].get(str(token_id))
        if row is not None and row["text"] != text:
            raise ValueError(f"ID/testo incompatibile con il vocabolario: {token_id}")
        terms = set(row.get("terms", [])) if row else set()
        terms.update(self.forms.get(text, ()))
        missing = terms - self.concepts.keys()
        if missing:
            raise ValueError(f"Concetti assenti dal catalogo: {missing}")
        if terms and row is None:
            self.extra[token_id] = {"text": text, "terms": sorted(terms)}
        result = {}
        for profile in profiles:
            selected = [self.concepts[t] for t in terms]
            if profile != "full":
                selected = [c for c in selected if c.get("variant_type") != "fragment"]
            if profile == "specific":
                selected = [
                    c for c in selected if not c.get("context_sensitive", False)
                ]
            result[profile] = {g for c in selected for g in c["groups"]}
        self.cache[token_id] = (text, result)
        return result


def entry_arrays(entry):
    ids, texts, lp = entry["ids"], entry["tokens"], entry["logp"]
    if (
        not ids
        or len(ids) != len(texts)
        or len(ids) != len(lp)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Top-k vuoto, duplicato o con lunghezze incompatibili")
    lp = np.asarray(lp, dtype=float)
    if np.isnan(lp).any() or np.isposinf(lp).any() or (lp > 1e-5).any():
        raise ValueError("Logprob non validi")
    order = np.argsort(-lp, kind="stable")
    return (
        np.asarray(ids)[order],
        np.asarray(texts, dtype=object)[order],
        lp[order],
        np.exp(lp[order]),
    )


def group_masks(memberships, cwes, expected):
    groups = {
        g: np.array([g in m for m in memberships])
        for g in ["generic", "safety", "vuln"] + cwes
    }
    groups["union"] = np.array([bool(m) for m in memberships])
    groups["outside"] = ~groups["union"]
    groups["safety_only"] = groups["safety"] & ~groups["vuln"]
    groups["vuln_only"] = groups["vuln"] & ~groups["safety"]
    groups["polarity_overlap"] = groups["safety"] & groups["vuln"]
    groups["cwe_union"] = np.array([bool(m & set(cwes)) for m in memberships])
    groups["expected_cwe"] = np.array([bool(m & expected) for m in memberships])
    groups["other_cwe"] = np.array(
        [bool(m & (set(cwes) - expected)) for m in memberships]
    )
    groups["expected_cwe_exclusive"] = groups["expected_cwe"] & ~groups["other_cwe"]
    groups["generic_only"] = groups["generic"] & ~(
        groups["safety"] | groups["vuln"] | groups["cwe_union"]
    )
    return groups


def measure(p, mask, ids, texts):
    pos = np.flatnonzero(mask)
    selected = p[mask]
    total = float(selected.sum())
    q = selected[selected > 0] / total if total else np.array([])
    entropy = float(-(q * np.log(q)).sum()) if len(q) else None
    row = {
        "score_sum": total,
        "share_observed": total / p.sum() if p.sum() else None,
        "count": len(pos),
        "first_rank": int(pos[0]) + 1 if len(pos) else None,
        "reciprocal_rank": 1 / (int(pos[0]) + 1) if len(pos) else 0,
        "best_score": float(selected[0]) if len(pos) else 0,
        "best_id": int(ids[pos[0]]) if len(pos) else None,
        "best_token": str(texts[pos[0]]) if len(pos) else None,
        "top1_contribution": float(selected[0] / total) if total else None,
        "top5_contribution": float(selected[:5].sum() / total) if total else None,
        "top10_contribution": float(selected[:10].sum() / total) if total else None,
        "entropy_conditional": entropy,
        "effective_tokens": math.exp(entropy) if entropy is not None else None,
        "hhi_conditional": float((q * q).sum()) if len(q) else None,
    }
    for k in KS:
        row[f"sum_at_{k}"] = float(p[:k][mask[:k]].sum()) if k <= len(p) else None
        row[f"count_at_{k}"] = int(mask[:k].sum()) if k <= len(p) else None
        row[f"hit_at_{k}"] = bool(mask[:k].any()) if k <= len(p) else None
    return row


def summarize(df, keys, columns, reps, rng):
    result = []
    for key, sub in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        head = dict(zip(keys, key, strict=False))
        values = sub[columns].to_numpy(float)
        codes, units = pd.factorize(sub["cluster"])
        ci = np.full((2, len(columns)), np.nan)
        valid_replicates = np.zeros(len(columns), dtype=int)
        if reps and len(units) > 1:
            sums = np.column_stack(
                [
                    np.bincount(
                        codes,
                        weights=np.where(np.isfinite(values[:, j]), values[:, j], 0.0),
                        minlength=len(units),
                    )
                    for j in range(len(columns))
                ]
            )
            counts = np.column_stack(
                [
                    np.bincount(
                        codes, weights=np.isfinite(values[:, j]), minlength=len(units)
                    )
                    for j in range(len(columns))
                ]
            )
            ix = rng.integers(0, len(units), (reps, len(units)))
            s, n = sums[ix].sum(axis=1), counts[ix].sum(axis=1)
            boots = np.divide(s, n, out=np.full_like(s, np.nan), where=n > 0)
            for j in range(len(columns)):
                b = boots[:, j]
                b = b[np.isfinite(b)]
                valid_replicates[j] = len(b)
                if len(b):
                    ci[:, j] = np.quantile(b, [0.025, 0.975])
        for j, metric in enumerate(columns):
            a = values[:, j]
            good = np.isfinite(a)
            a = a[good]
            row = head | {
                "metric": metric,
                "n": len(a),
                "clusters": len(set(codes[good])),
                "bootstrap_valid_replicates": int(valid_replicates[j]),
                "ci_status": "bootstrap_disabled"
                if not reps
                else "insufficient_clusters"
                if len(set(codes[good])) < 2
                else "pointwise_cluster_percentile",
            }
            if row["clusters"] < 2:
                ci[:, j] = np.nan
            if len(a):
                q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])
                row.update(
                    mean=float(a.mean()),
                    std=float(a.std(ddof=1)) if len(a) > 1 else None,
                    median=q[2],
                    q05=q[0],
                    q25=q[1],
                    q75=q[3],
                    q95=q[4],
                    min=a.min(),
                    max=a.max(),
                    ci_low=ci[0, j],
                    ci_high=ci[1, j],
                )
            result.append(row)
    return pd.DataFrame(
        result,
        columns=keys
        + [
            "metric",
            "n",
            "clusters",
            "bootstrap_valid_replicates",
            "ci_status",
            "mean",
            "std",
            "median",
            "q05",
            "q25",
            "q75",
            "q95",
            "min",
            "max",
            "ci_low",
            "ci_high",
        ],
    )


def contrasts(df, keys, field, comparisons, columns, reps, rng):
    rows = []
    for key, sub in df.groupby(keys, dropna=False, sort=True):
        for left, right in comparisons:
            selected = sub[sub[field].isin([left, right])]
            if selected[field].nunique() < 2:
                continue
            for metric in columns:
                data = selected[["cluster", field, metric]].dropna()
                a = data.loc[data[field] == left, metric].to_numpy(float)
                b = data.loc[data[field] == right, metric].to_numpy(float)
                if not len(a) or not len(b):
                    continue
                auc = (
                    pd.Series(np.r_[a, b]).rank().to_numpy()[: len(a)].sum()
                    - len(a) * (len(a) + 1) / 2
                ) / (len(a) * len(b))
                ci = [None, None]
                codes, units = pd.factorize(data.cluster)
                if reps and len(units) > 1:
                    cell = codes * 2 + (data[field].to_numpy() == right).astype(int)
                    sums = np.bincount(
                        cell,
                        weights=data[metric].to_numpy(float),
                        minlength=2 * len(units),
                    ).reshape(-1, 2)
                    counts = np.bincount(cell, minlength=2 * len(units)).reshape(-1, 2)
                    ix = rng.integers(0, len(units), (reps, len(units)))
                    s, n = sums[ix].sum(axis=1), counts[ix].sum(axis=1)
                    valid = (n > 0).all(axis=1)
                    means = s[valid] / n[valid]
                    if len(means):
                        ci = np.quantile(
                            means[:, 0] - means[:, 1], [0.025, 0.975]
                        ).tolist()
                rows.append(
                    dict(zip(keys, key, strict=False))
                    | {
                        "field": field,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        "n_left": len(a),
                        "n_right": len(b),
                        "mean_left": a.mean(),
                        "mean_right": b.mean(),
                        "difference_mean": a.mean() - b.mean(),
                        "difference_median": np.median(a) - np.median(b),
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                        "cliffs_delta": 2 * auc - 1,
                        "auc_left_high": auc,
                    }
                )
    return pd.DataFrame(rows)


def pairing(cases, groups, args):
    links = []
    for (run, _pair), sub in cases.dropna(subset=["pair_id"]).groupby(
        ["run", "pair_id"]
    ):
        a, b = sub[sub["gt"] == "vuln"], sub[sub["gt"] == "safe"]
        if len(a) == len(b) == 1:
            links.append(
                {
                    "kind": "vuln_minus_safe",
                    "run_a": run,
                    "case_a": a.iloc[0].case_id,
                    "run_b": run,
                    "case_b": b.iloc[0].case_id,
                }
            )
    for _, sub in cases.dropna(subset=["code_sha256"]).groupby(
        ["family", "config", "code_sha256", "gt"]
    ):
        a, b = sub[sub.variant == "extended"], sub[sub.variant == "base"]
        if len(a) == len(b) == 1:
            links.append(
                {
                    "kind": "extended_minus_base_exact_code",
                    "run_a": a.iloc[0].run,
                    "case_a": a.iloc[0].case_id,
                    "run_b": b.iloc[0].run,
                    "case_b": b.iloc[0].case_id,
                }
            )
    if args.pairs:
        links.extend(pd.read_csv(args.pairs, dtype=str).to_dict("records"))
    linkdf = pd.DataFrame(
        links, columns=["kind", "run_a", "case_a", "run_b", "case_b"]
    ).drop_duplicates()
    save(linkdf, "pair_links", args.out, args.format)
    indexed = {
        (run, cid): sub for (run, cid), sub in groups.groupby(["run", "case_id"])
    }
    caseindex = cases.set_index(["run", "case_id"])
    rows, coverage = [], []
    joins = ["stream", "window", "profile", "layer", "group"]
    for p in linkdf.to_dict("records"):
        ka, kb = (p["run_a"], p["case_a"]), (p["run_b"], p["case_b"])
        if ka not in caseindex.index or kb not in caseindex.index:
            raise ValueError(f"Estremo di coppia non presente nei JSONL: {p}")
        available = ka in indexed and kb in indexed
        coverage.append(
            p
            | {
                "readout_a": ka in indexed,
                "readout_b": kb in indexed,
                "status": "available" if available else "missing_or_inexact_readout",
            }
        )
        if not available:
            continue
        joined = indexed[ka][joins + METRICS].merge(
            indexed[kb][joins + METRICS],
            on=joins,
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        for r in joined.to_dict("records"):
            row = (
                p
                | {k: r[k] for k in joins}
                | {
                    "outcome_a": caseindex.loc[ka, "outcome"],
                    "outcome_b": caseindex.loc[kb, "outcome"],
                    "cluster": str(p["run_b"])
                    + "::"
                    + str(caseindex.loc[kb, "cluster"]),
                }
            )
            row.update({"delta_" + m: r[m + "_a"] - r[m + "_b"] for m in METRICS})
            rows.append(row)
    save(pd.DataFrame(coverage), "pair_link_coverage", args.out, args.format)
    diffs = pd.DataFrame(rows)
    save(diffs, "paired_differences", args.out, args.format)
    if len(diffs):
        s = diffs.groupby(["kind", "run_a", "run_b"] + joins)[
            ["delta_" + m for m in METRICS]
        ].agg(["count", "mean", "median", "std"])
        s.columns = ["_".join(c) for c in s.columns]
        save(s.reset_index(), "paired_summary", args.out, args.format)
        save(
            summarize(
                diffs,
                ["kind", "run_a", "run_b"] + joins,
                ["delta_" + m for m in METRICS],
                args.bootstrap,
                np.random.default_rng(args.seed),
            ),
            "paired_summary_with_intervals",
            args.out,
            args.format,
        )


def make_plots(groups, scalar, summaries, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = args.out / "plots"
    folder.mkdir(exist_ok=True)
    color = {
        "safety": "#278348",
        "vuln": "#c63333",
        "expected_cwe": "#2674bc",
        "other_cwe": "#d28b21",
        "outside": "#888888",
        "union": "#685095",
        "safe": "#278348",
        "TP": "#c63333",
        "TN": "#278348",
        "FP": "#d28b21",
        "FN": "#2674bc",
    }

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(folder / (name + ".png"), dpi=140)
        plt.close(fig)

    for (run, window, profile), sub in scalar[scalar.stream == "lens"].groupby(
        ["run", "window", "profile"]
    ):
        stem = f"{run}_w{window}_{profile}"
        ylabel = "Massa osservata" if window == 1 else "Somma punteggi max"
        for mode, field in [("gt", "gt"), ("confusion", "outcome")]:
            if mode not in args.compare:
                continue
            summary = summaries[mode]
            part = summary[
                (summary.run == run)
                & (summary.window == window)
                & (summary.profile == profile)
                & (summary.stream == "lens")
                & (summary.metric == "score_sum")
            ]
            cats = plot_categories(field, part[field].unique())
            fig, axes = plt.subplots(
                1, len(cats), figsize=(5 * len(cats), 4), squeeze=False, sharey=True
            )
            for ax, cat in zip(axes[0], cats, strict=False):
                for group in ["safety", "vuln", "expected_cwe", "other_cwe", "outside"]:
                    s = part[(part[field] == cat) & (part.group == group)].sort_values(
                        "layer"
                    )
                    ax.plot(s.layer, s["mean"], label=group, color=color[group])
                    ax.fill_between(
                        s.layer, s.ci_low, s.ci_high, color=color[group], alpha=0.12
                    )
                ax.set(
                    title=category_label(args, run, field, cat, sub),
                    xlabel="Layer",
                    ylabel=ylabel,
                )
                ax.legend(fontsize=7)
            finish(fig, stem + "_" + mode)
            layers = sorted(sub.layer.unique())
            nr = (len(layers) + 3) // 4
            fig, axes = plt.subplots(
                nr, 4, figsize=(16, 3 * nr), squeeze=False, sharex=True
            )
            for ax, layer in zip(axes.flat, layers, strict=False):
                s = sub[sub.layer == layer]
                for cat in cats:
                    ax.hist(
                        s.loc[s[field] == cat, "polarity_contrast"].dropna(),
                        bins=np.linspace(-1, 1, 26),
                        alpha=0.4,
                        label=category_label(args, run, field, cat, s),
                        color=color.get(cat, "#888888"),
                    )
                ax.set_title(f"L{layer}")
            for ax in list(axes.flat)[len(layers) :]:
                ax.set_visible(False)
            axes.flat[0].legend()
            fig.supxlabel("Contrasto vuln-safe")
            fig.supylabel("Casi")
            finish(fig, stem + "_" + mode + "_histograms")
        if "layers" in args.compare:
            pivot = sub.pivot(
                index="case_id", columns="layer", values="polarity_contrast"
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(
                pivot.to_numpy(dtype=float),
                aspect="auto",
                vmin=-1,
                vmax=1,
                cmap="RdYlGn_r",
            )
            ax.set_xticks(range(len(pivot.columns)), pivot.columns)
            ax.set(xlabel="Layer", ylabel="Caso")
            fig.colorbar(im, ax=ax)
            finish(fig, stem + "_trajectories")
        if "outside" in args.compare:
            fig, ax = plt.subplots(figsize=(6, 5))
            last = sub[sub.layer == sub.layer.max()]
            for gt, s in last.groupby("gt"):
                ax.scatter(
                    s.union_sum,
                    s.outside_best_score,
                    label=gt,
                    color=color.get(gt, "#888888"),
                    alpha=0.6,
                )
            ax.set(
                xlabel=ylabel + " nei vocabolari",
                ylabel="Primo token esterno",
                title=f"Layer {sub.layer.max()}",
            )
            ax.legend()
            finish(fig, stem + "_outside")
        if "cwe" in args.compare:
            cg = groups[
                (groups.run == run)
                & (groups.window == window)
                & (groups.profile == profile)
                & (groups.stream == "lens")
                & groups.group.str.startswith("CWE-")
            ]
            for layer, c in cg.groupby("layer"):
                mat = c.pivot_table(
                    index="cwe", columns="group", values="score_sum", aggfunc="mean"
                )
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(mat, aspect="auto", cmap="Blues")
                ax.set_xticks(
                    range(len(mat.columns)), mat.columns, rotation=90, fontsize=7
                )
                ax.set_yticks(range(len(mat.index)), mat.index, fontsize=7)
                ax.set(
                    title=f"Layer {layer}",
                    xlabel="Vocabolario CWE",
                    ylabel="CWE del caso",
                )
                fig.colorbar(im, ax=ax, label=ylabel)
                finish(fig, stem + f"_cwe_L{layer}")


def make_sg_plots(groups, summaries, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = args.out / "plots"
    folder.mkdir(exist_ok=True)
    colors = {
        "safety": "#278348",
        "vuln": "#c63333",
        "expected_cwe": "#2674bc",
        "union": "#685095",
    }
    for (run, window), frame in groups[groups.stream == "lens"].groupby(
        ["run", "window"]
    ):
        for mode, field in [("gt", "gt"), ("confusion", "outcome")]:
            if mode not in summaries:
                continue
            source = summaries[mode]
            part = source[
                (source.run == run)
                & (source.window == window)
                & (source.stream == "lens")
                & source.metric.isin(["sg", "sg_z", "exact_mass"])
            ]
            categories = plot_categories(field, part[field].unique())
            for metric in ["sg", "sg_z", "exact_mass"]:
                fig, axes = plt.subplots(
                    1,
                    len(categories),
                    figsize=(5 * len(categories), 4),
                    squeeze=False,
                    sharey=True,
                )
                for ax, category in zip(axes[0], categories, strict=False):
                    for group in ["safety", "vuln", "expected_cwe", "union"]:
                        s = part[
                            (part[field] == category)
                            & (part.metric == metric)
                            & (part.group == group)
                        ].sort_values("layer")
                        if not len(s):
                            continue
                        ax.plot(s.layer, s["mean"], label=group, color=colors[group])
                        ax.fill_between(
                            s.layer,
                            s.ci_low,
                            s.ci_high,
                            color=colors[group],
                            alpha=0.12,
                        )
                    ax.set(
                        title=category_label(args, run, field, category, frame),
                        xlabel="Layer",
                        ylabel=metric,
                    )
                    legend_if_present(ax, fontsize=7)
                fig.tight_layout()
                fig.savefig(folder / f"{run}_w{window}_{mode}_{metric}.png", dpi=140)
                plt.close(fig)
        if "cwe" in args.compare:
            selected = frame[frame.group.str.startswith("CWE-")]
            limit = max(1.0, float(selected.sg.abs().max()))
            for layer, s in selected.groupby("layer"):
                matrix = s.pivot_table(
                    index="cwe", columns="group", values="sg", aggfunc="mean"
                )
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(
                    matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
                )
                ax.set_xticks(
                    range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7
                )
                ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
                ax.set(
                    title=f"SG L{layer}",
                    xlabel="Vocabolario CWE",
                    ylabel="CWE del caso",
                )
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(folder / f"{run}_w{window}_sg_cwe_L{layer}.png", dpi=140)
                plt.close(fig)


def analyze(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(
            "Usa una cartella --out nuova o vuota per evitare risultati mescolati"
        )
    if (
        args.format in ("parquet", "both")
        and importlib.util.find_spec("pyarrow") is None
    ):
        raise ValueError("Per Parquet installa pyarrow, oppure usa --format csv")
    if not args.no_plots and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("Installa matplotlib oppure usa --no-plots")
    vocab = Vocabulary(args.vocab)
    args.out.mkdir(parents=True, exist_ok=True)
    db = (
        sqlite3.connect(args.out / "tokens.sqlite")
        if "outside" in args.compare
        else None
    )
    if db:
        db.execute(
            "CREATE TABLE tokens (run TEXT,case_id TEXT,stream TEXT,layer INTEGER,window INTEGER,token_id INTEGER,token TEXT,rank INTEGER,score REAL,groups_text TEXT,outside INTEGER,gt TEXT,outcome TEXT,cwe TEXT)"
        )
    records = []
    rows = []
    global_rows = []
    turnover = []
    issues = []
    inputs = []
    seen_files = set()
    seen_runs = set()
    sg_rows = []
    sg_runs = []
    windows = None if args.windows == "all" else set(map(int, args.windows.split(",")))
    layers = None if args.layers == "all" else set(map(int, args.layers.split(",")))
    mapping = json.loads(args.dataset_map.read_text()) if args.dataset_map else {}
    try:
        for path in args.inputs:
            h = sha(path)
            run = path.name.removesuffix(".gz").removesuffix(".jsonl")
            if h in seen_files or run in seen_runs:
                raise ValueError("Input duplicati per contenuto o nome")
            seen_files.add(h)
            seen_runs.add(run)
            meta = None
            seen = set()
            count = 0
            for _lineno, rec in read_records(path):
                if "_meta" in rec:
                    if meta is not None:
                        raise ValueError("Metadati ripetuti nello stesso JSONL")
                    meta = rec["_meta"]
                    if meta.get("hf_id") != vocab.compiled["meta"].get("tokenizer_id"):
                        raise ValueError(f"Modello incompatibile: {path}")
                    pilot_markers = (
                        meta.get("pilot", {}).get("frozen", {}).get("markers", [])
                    )
                    if len(pilot_markers) > 1:
                        raise ValueError(
                            "JSONL con marcatori multipli: analizzare separatamente per pilot_marker; vietato aggregarli come esempi indipendenti"
                        )
                    sg_meta = meta.get("vocab_scores")
                    if sg_meta:
                        if sg_meta.get("version") != "sg-full-vocab-v1":
                            raise ValueError("Unsupported SG version")
                        frozen = {
                            g: sorted(set(ids))
                            for g, ids in vocab.compiled["groups"].items()
                        }
                        frozen["union"] = sorted(
                            {i for ids in frozen.values() for i in ids}
                        )
                        signature = hashlib.sha256(
                            json.dumps(
                                frozen, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest()
                        if sg_meta.get("group_signature") != signature:
                            raise ValueError(
                                "SG group IDs differ from the selected compiled vocabulary"
                            )
                        if sg_meta.get("catalog_sha256") and sg_meta[
                            "catalog_sha256"
                        ] != sha(args.vocab / "concepts.json"):
                            raise ValueError(
                                "SG catalog differs from the selected vocabulary"
                            )
                        sg_runs.append(run)
                    elif args.require_sg:
                        raise ValueError(
                            f"SG not saved in {path}; rerun the patched pipeline"
                        )
                    continue
                if meta is None:
                    raise ValueError("Intestazione _meta assente")
                cid = str(rec["case_id"])
                if cid in seen:
                    raise ValueError(f"case_id duplicato: {run}/{cid}")
                seen.add(cid)
                count += 1
                gt, pred = (
                    normalize(rec.get("ground_truth")),
                    normalize(rec.get("pred", rec.get("pred_label"))),
                )
                outcome = {
                    ("vuln", "vuln"): "TP",
                    ("safe", "safe"): "TN",
                    ("safe", "vuln"): "FP",
                    ("vuln", "safe"): "FN",
                }.get((gt, pred), "unknown")
                expected = set(rec.get("cwes") or [rec.get("cwe")])
                expected.discard(None)
                unknown = expected - set(vocab.cwes)
                if unknown:
                    issues.append(
                        {
                            "run": run,
                            "case_id": cid,
                            "issue": "unknown_cwes",
                            "detail": "|".join(sorted(unknown)),
                        }
                    )
                expected &= set(vocab.cwes)
                name = str(meta.get("dataset") or run).lower()
                override = mapping.get(run, {})
                config = {
                    k: v
                    for k, v in meta.items()
                    if k not in ["dataset", "created", "limit", "dataset_kind"]
                }
                case = {
                    "run": run,
                    "case_id": cid,
                    "gt": gt,
                    "prediction": pred,
                    "outcome": outcome,
                    "cwe": "|".join(sorted(expected)),
                    "family": override.get(
                        "family", "dual" if "dual" in name else "controlled"
                    ),
                    "variant": override.get(
                        "variant", "extended" if "extended" in name else "base"
                    ),
                    "config": hashlib.sha256(
                        json.dumps(config, sort_keys=True).encode()
                    ).hexdigest(),
                    "pair_id": rec.get("pair_id"),
                    "quartet_id": rec.get("quartet_id"),
                    "cluster": str(rec.get("quartet_id") or rec.get("pair_id") or cid),
                    "code_sha256": rec.get("source_code_sha256") or None,
                    "truncated": rec.get("truncated"),
                    "carrier_ok": rec.get("carrier_ok"),
                    "response": rec.get("response", ""),
                    "model": meta.get("model", meta.get("hf_id")),
                    "pilot_marker": rec.get("pilot_marker"),
                    "anchor_available": False,
                    "anchor_exact": False,
                    "target_labels": json.dumps(
                        rec.get("target_labels", {}), sort_keys=True
                    ),
                }
                records.append(case)
                anchor = rec.get("anchor_readout")
                if anchor is None and "anchor_readout_packed" in rec:
                    anchor = unpack_readout(rec["anchor_readout_packed"])
                anchor = (anchor or {}).get("anchors", {}).get(args.anchor)
                if anchor is None:
                    issues.append(
                        {
                            "run": run,
                            "case_id": cid,
                            "issue": "missing_anchor",
                            "detail": args.anchor,
                        }
                    )
                    continue
                case["anchor_available"] = True
                case["anchor_exact"] = bool(anchor.get("boundary_exact", False))
                if not anchor.get("boundary_exact", False):
                    issues.append(
                        {
                            "run": run,
                            "case_id": cid,
                            "issue": "inexact_boundary",
                            "detail": args.anchor,
                        }
                    )
                    # Never pool shifted readouts with exact positions.
                    continue
                group_scores = anchor.get("vocab_scores")
                if run in sg_runs and group_scores is None and args.require_sg:
                    raise ValueError(f"SG anchor missing: {run}/{cid}")
                if group_scores:
                    if group_scores.get("profile") != "full":
                        raise ValueError("SG profile must be full")
                    sg_entries = [
                        ("lens", int(layer), ws)
                        for layer, ws in group_scores.get("lens", {}).items()
                    ]
                    if not args.no_model:
                        sg_entries.append(("model", -1, group_scores.get("model", {})))
                    for stream, layer, ws in sg_entries:
                        if (
                            stream == "lens"
                            and layers is not None
                            and layer not in layers
                        ):
                            continue
                        for w, e in ws.items():
                            window = int(w)
                            if windows is not None and window not in windows:
                                continue
                            if (
                                e.get("aggregation")
                                != "mean_of_per_position_statistics"
                            ):
                                raise ValueError("Unsupported SG window aggregation")
                            for group, values in e["groups"].items():
                                sg_rows.append(
                                    {
                                        "run": run,
                                        "case_id": cid,
                                        "stream": stream,
                                        "layer": layer,
                                        "window": window,
                                        "profile": "full",
                                        "group": group,
                                        "n_ids": values["n_ids"],
                                        "sg": values["sg"],
                                        "sg_z": values["sg_z"],
                                        "exact_mass": values["probability_mass"],
                                        "mean_logit": values["mean_logit"],
                                        "mean_logp": values["mean_logp"],
                                        "vocab_size": e["vocab_size"],
                                        "vocab_mean_logit": e["vocab_mean_logit"],
                                        "vocab_std_logit": e["vocab_std_logit"],
                                        "sg_aggregation": e["aggregation"],
                                    }
                                )
                            if len(expected) == 1:
                                target = next(iter(expected))
                                values = e["groups"].get(target)
                                if values:
                                    sg_rows.append(
                                        {
                                            "run": run,
                                            "case_id": cid,
                                            "stream": stream,
                                            "layer": layer,
                                            "window": window,
                                            "profile": "full",
                                            "group": "expected_cwe",
                                            "n_ids": values["n_ids"],
                                            "sg": values["sg"],
                                            "sg_z": values["sg_z"],
                                            "exact_mass": values["probability_mass"],
                                            "mean_logit": values["mean_logit"],
                                            "mean_logp": values["mean_logp"],
                                            "vocab_size": e["vocab_size"],
                                            "vocab_mean_logit": e["vocab_mean_logit"],
                                            "vocab_std_logit": e["vocab_std_logit"],
                                            "sg_aggregation": e["aggregation"],
                                        }
                                    )
                entries = [
                    ("lens", int(l), int(w), e)
                    for l, ws in anchor.get("lens", {}).items()
                    for w, e in ws.items()
                ]
                if not args.no_model:
                    entries += [
                        ("model", -1, int(w), e)
                        for w, e in anchor.get("model", {}).items()
                    ]
                previous = {}
                batch = []
                for stream, layer, window, entry in sorted(
                    entries, key=lambda e: (e[0], e[2], e[1])
                ):
                    if windows is not None and window not in windows:
                        continue
                    if stream == "lens" and layers is not None and layer not in layers:
                        continue
                    ids, texts, _lp, p = entry_arrays(entry)
                    is_mass = window == 1 and entry.get("used_positions", window) == 1
                    if is_mass and p.sum() > 1.0001:
                        raise ValueError(f"Massa top-k >1: {run}/{cid}/{layer}")
                    memberships = [
                        vocab.lookup(int(i), str(t), args.profiles)
                        for i, t in zip(ids, texts, strict=False)
                    ]
                    if db:
                        for rank, (i, t, prob, m) in enumerate(
                            zip(ids, texts, p, memberships, strict=False), 1
                        ):
                            groups = (
                                m["full"]
                                if "full" in m
                                else vocab.lookup(int(i), str(t), args.profiles)[
                                    args.profiles[0]
                                ]
                            )
                            batch.append(
                                (
                                    run,
                                    cid,
                                    stream,
                                    layer,
                                    window,
                                    int(i),
                                    str(t),
                                    rank,
                                    float(prob),
                                    "|".join(sorted(groups)),
                                    int(not groups),
                                    gt,
                                    outcome,
                                    case["cwe"],
                                )
                            )
                    for profile in args.profiles:
                        key = {
                            "run": run,
                            "case_id": cid,
                            "stream": stream,
                            "layer": layer,
                            "window": window,
                            "profile": profile,
                        }
                        masks = group_masks(
                            [m[profile] for m in memberships], vocab.cwes, expected
                        )
                        gm = {
                            g: measure(p, mask, ids, texts) for g, mask in masks.items()
                        }
                        rows.extend(key | {"group": g} | r for g, r in gm.items())
                        safe, vuln = gm["safety"]["score_sum"], gm["vuln"]["score_sum"]
                        so, vo = (
                            gm["safety_only"]["score_sum"],
                            gm["vuln_only"]["score_sum"],
                        )
                        cwe_scores = {g: gm[g]["score_sum"] for g in vocab.cwes}
                        best = max((cwe_scores[g] for g in expected), default=0)
                        rank = (
                            1 + sum(s > best for s in cwe_scores.values())
                            if best > 0
                            else None
                        )
                        global_rows.append(
                            key
                            | {
                                "is_probability_mass": is_mass,
                                "used_positions": entry.get("used_positions", window),
                                "saved_k": len(ids),
                                "saved_sum": float(p.sum()),
                                "residual_mass": max(0.0, 1 - float(p.sum()))
                                if is_mass
                                else None,
                                "boundary_exact": anchor.get("boundary_exact"),
                                "anchor_last_token": (
                                    anchor.get("tail_tokens") or [""]
                                )[-1],
                                "safety_sum": safe,
                                "vuln_sum": vuln,
                                "polarity_difference": vuln - safe,
                                "polarity_contrast": (vuln - safe) / (vuln + safe)
                                if vuln + safe
                                else None,
                                "exclusive_contrast": (vo - so) / (vo + so)
                                if vo + so
                                else None,
                                "union_sum": gm["union"]["score_sum"],
                                "outside_sum": gm["outside"]["score_sum"],
                                "outside_best_score": gm["outside"]["best_score"],
                                "inside_best_score": gm["union"]["best_score"],
                                "best_inside_minus_outside": gm["union"]["best_score"]
                                - gm["outside"]["best_score"],
                                "outside_before_inside": gm["union"]["first_rank"] - 1
                                if gm["union"]["first_rank"]
                                else len(ids),
                                "top1_token": str(texts[0]),
                                "top1_probability_or_score": float(p[0]),
                                "top1_outside": bool(masks["outside"][0]),
                                "expected_cwe_rank": rank,
                                "expected_cwe_ties": sum(
                                    v == best for v in cwe_scores.values()
                                )
                                if best
                                else None,
                                "expected_cwe_margin": best
                                - max(
                                    (
                                        s
                                        for g, s in cwe_scores.items()
                                        if g not in expected
                                    ),
                                    default=0,
                                ),
                                "expected_cwe_share": gm["expected_cwe"]["score_sum"]
                                / gm["cwe_union"]["score_sum"]
                                if gm["cwe_union"]["score_sum"]
                                else None,
                                "cwe_winners": "|".join(
                                    g
                                    for g, s in cwe_scores.items()
                                    if s > 0 and s == max(cwe_scores.values())
                                ),
                            }
                        )
                    if stream == "lens" and "layers" in args.compare:
                        if window in previous:
                            prev_layer, prev_ids = previous[window]
                            for k in KS:
                                if k > min(len(ids), len(prev_ids)):
                                    continue
                                a, b = set(prev_ids[:k]), set(ids[:k])
                                turnover.append(
                                    {
                                        "run": run,
                                        "case_id": cid,
                                        "window": window,
                                        "layer_from": prev_layer,
                                        "layer_to": layer,
                                        "consecutive": layer == prev_layer + 1,
                                        "k": k,
                                        "jaccard": len(a & b) / len(a | b),
                                        "entered": len(b - a),
                                        "exited": len(a - b),
                                    }
                                )
                        previous[window] = (layer, ids)
                if db and batch:
                    db.executemany(
                        "INSERT INTO tokens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
                    )
                    db.commit()
                if count % 10 == 0:
                    print(f"{run}: {count} casi", flush=True)
            inputs.append(
                {
                    "path": str(path.resolve()),
                    "sha256": h,
                    "cases": count,
                    "metadata": meta,
                }
            )
        if not rows:
            raise ValueError("Nessun anchor/layer/finestra selezionato trovato")
        cases = pd.DataFrame(records)
        args.case_counts = cases[
            ["run", "case_id", "gt", "outcome", "anchor_available", "anchor_exact"]
        ].copy()
        quality_tables(cases, args)
        groups = pd.DataFrame(rows)
        scalar = pd.DataFrame(global_rows)
        if sg_rows:
            if set(sg_runs) != set(seen_runs):
                raise ValueError(
                    "Do not mix runs with and without SG in the same analysis"
                )
            sg = pd.DataFrame(sg_rows)
            key = ["run", "case_id", "stream", "layer", "window", "profile", "group"]
            if sg.duplicated(key).any():
                raise ValueError("Duplicate SG observation")
            observation_key = ["run", "case_id", "stream", "layer", "window", "profile"]
            expected_observations = set(map(tuple, scalar[observation_key].to_numpy()))
            actual_observations = set(
                map(tuple, sg[sg.group == "union"][observation_key].to_numpy())
            )
            if expected_observations != actual_observations:
                raise ValueError(
                    "SG does not cover all selected case/layer/window observations"
                )
            save(sg, "sg_group_metrics", args.out, args.format)
            groups = groups.merge(sg, on=key, how="left", validate="one_to_one")
            groups["unobserved_mass"] = np.where(
                groups.window == 1, groups.exact_mass - groups.score_sum, np.nan
            )
            if (groups.unobserved_mass.dropna() < -1e-4).any():
                raise ValueError(
                    "Observed top-k mass exceeds exact SG vocabulary mass; check ID membership"
                )
            groups["unobserved_mass"] = groups.unobserved_mass.clip(lower=0)
            for metric in ["sg", "sg_z", "exact_mass", "unobserved_mass"]:
                if metric not in METRICS:
                    METRICS.append(metric)
            safe = sg[sg.group == "safety"]
            vuln = sg[sg.group == "vuln"]
            global_key = ["run", "case_id", "stream", "layer", "window", "profile"]
            polar = safe[global_key + ["sg", "sg_z", "exact_mass"]].merge(
                vuln[global_key + ["sg", "sg_z", "exact_mass"]],
                on=global_key,
                suffixes=("_safe", "_vuln"),
                validate="one_to_one",
            )
            polar["sg_polarity_difference"] = polar.sg_vuln - polar.sg_safe
            polar["sg_z_polarity_difference"] = polar.sg_z_vuln - polar.sg_z_safe
            denom = polar.exact_mass_vuln + polar.exact_mass_safe
            polar["exact_mass_polarity_contrast"] = np.divide(
                polar.exact_mass_vuln - polar.exact_mass_safe,
                denom,
                out=np.full(len(polar), np.nan),
                where=denom > 0,
            )
            scalar = scalar.merge(
                polar[
                    global_key
                    + [
                        "sg_polarity_difference",
                        "sg_z_polarity_difference",
                        "exact_mass_polarity_contrast",
                    ]
                ],
                on=global_key,
                how="left",
                validate="one_to_one",
            )

        for name, df in [
            ("cases", cases),
            ("group_metrics", groups),
            ("case_layer_metrics", scalar),
            ("quality_issues", pd.DataFrame(issues)),
        ]:
            save(df, name, args.out, args.format)
        groups = groups.merge(cases, on=["run", "case_id"], validate="many_to_one")
        scalar = scalar.merge(cases, on=["run", "case_id"], validate="many_to_one")
        rng = np.random.default_rng(args.seed)
        summaries = {}
        scalar_metrics = [
            "saved_sum",
            "residual_mass",
            "safety_sum",
            "vuln_sum",
            "polarity_contrast",
            "exclusive_contrast",
            "union_sum",
            "outside_sum",
            "outside_best_score",
            "best_inside_minus_outside",
            "top1_outside",
            "expected_cwe_rank",
            "expected_cwe_margin",
            "expected_cwe_share",
        ]
        scalar_metrics += [
            m
            for m in [
                "sg_polarity_difference",
                "sg_z_polarity_difference",
                "exact_mass_polarity_contrast",
            ]
            if m in scalar.columns
        ]
        global_keys = ["run", "stream", "window", "profile", "layer"]
        for mode, field in [
            ("overview", None),
            ("gt", "gt"),
            ("confusion", "outcome"),
            ("cwe", "cwe"),
        ]:
            if mode not in args.compare:
                continue
            print("Riepiloghi: " + mode, flush=True)
            extra = [field] if field else []
            summaries[mode] = summarize(
                groups,
                BASE + extra,
                METRICS,
                0 if mode == "cwe" else args.bootstrap,
                rng,
            )
            save(summaries[mode], "summary_" + mode, args.out, args.format)
            save(
                summarize(
                    scalar,
                    global_keys + extra,
                    scalar_metrics,
                    0 if mode == "cwe" else args.bootstrap,
                    rng,
                ),
                "global_" + mode,
                args.out,
                args.format,
            )
        if "gt" in args.compare or "confusion" in args.compare:
            save(
                cases.groupby(["run", "gt", "prediction", "outcome"])
                .size()
                .reset_index(name="n"),
                "behavior",
                args.out,
                args.format,
            )
            for mode, field, pairs in [
                ("gt", "gt", [("vuln", "safe")]),
                (
                    "confusion",
                    "outcome",
                    [("TP", "FN"), ("TN", "FP"), ("TP", "FP"), ("TN", "FN")],
                ),
            ]:
                if mode not in args.compare:
                    continue
                print("Confronti: " + mode, flush=True)
                save(
                    contrasts(groups, BASE, field, pairs, METRICS, args.bootstrap, rng),
                    "contrasts_" + mode,
                    args.out,
                    args.format,
                )
                save(
                    contrasts(
                        scalar,
                        global_keys,
                        field,
                        pairs,
                        scalar_metrics,
                        args.bootstrap,
                        rng,
                    ),
                    "global_contrasts_" + mode,
                    args.out,
                    args.format,
                )
        if "cwe" in args.compare:
            macro = (
                groups.groupby(BASE + ["gt", "cwe"])
                .score_sum.mean()
                .groupby(BASE + ["gt"])
                .agg(["mean", "count"])
                .reset_index()
            )
            save(macro, "macro_cwe", args.out, args.format)
            overlap = []
            for a, ids_a in vocab.compiled["groups"].items():
                for b, ids_b in vocab.compiled["groups"].items():
                    a_set, b_set = set(ids_a), set(ids_b)
                    overlap.append(
                        {
                            "group_a": a,
                            "group_b": b,
                            "size_a": len(a_set),
                            "size_b": len(b_set),
                            "intersection": len(a_set & b_set),
                            "jaccard": len(a_set & b_set) / len(a_set | b_set)
                            if a_set | b_set
                            else None,
                        }
                    )
            save(pd.DataFrame(overlap), "vocab_overlap", args.out, args.format)
        if "datasets" in args.compare:
            cross = groups.copy()
            cross["cluster"] = cross.run + "::" + cross.cluster
            keys = ["family", "config", "stream", "window", "profile", "layer", "group"]
            for split in [None, "gt", "outcome"]:
                save(
                    contrasts(
                        cross,
                        keys + ([split] if split else []),
                        "variant",
                        [("extended", "base")],
                        ["score_sum", "reciprocal_rank"]
                        + [
                            m
                            for m in ["sg", "sg_z", "exact_mass"]
                            if m in cross.columns
                        ],
                        args.bootstrap,
                        rng,
                    ),
                    "datasets_" + (split or "all"),
                    args.out,
                    args.format,
                )
            save(
                summarize(
                    groups, ["family", "variant"] + BASE[1:], ["score_sum"], 0, rng
                ),
                "dataset_overview",
                args.out,
                args.format,
            )
            if sg_rows:
                cross_global = scalar.copy()
                cross_global["cluster"] = cross_global.run + "::" + cross_global.cluster
                for split in [None, "gt", "outcome"]:
                    save(
                        contrasts(
                            cross_global,
                            ["family", "config", "stream", "window", "profile", "layer"]
                            + ([split] if split else []),
                            "variant",
                            [("extended", "base")],
                            [
                                "sg_polarity_difference",
                                "sg_z_polarity_difference",
                                "exact_mass_polarity_contrast",
                            ],
                            args.bootstrap,
                            rng,
                        ),
                        "datasets_global_sg_" + (split or "all"),
                        args.out,
                        args.format,
                    )

        if "pairs" in args.compare:
            pairing(cases, groups, args)
        if "layers" in args.compare:
            save(pd.DataFrame(turnover), "layer_turnover", args.out, args.format)
            keys = ["run", "case_id", "stream", "window", "profile"]
            delta = scalar.sort_values(keys + ["layer"]).copy()
            delta["layer_from"] = delta.groupby(keys).layer.shift()
            delta["consecutive"] = delta.layer - delta.layer_from == 1
            for m in [
                "safety_sum",
                "vuln_sum",
                "polarity_contrast",
                "union_sum",
                "expected_cwe_margin",
            ]:
                delta["delta_" + m] = delta.groupby(keys)[m].diff()
            save(delta, "layer_deltas", args.out, args.format)
            trajectories = []
            for key, s in scalar[scalar.stream == "lens"].groupby(
                ["run", "case_id", "window", "profile"]
            ):
                s = s.sort_values("layer")
                v = s.polarity_contrast.dropna()
                sign = np.sign(v.to_numpy())
                sign = sign[sign != 0]
                trajectories.append(
                    dict(
                        zip(["run", "case_id", "window", "profile"], key, strict=False)
                    )
                    | {
                        "first_observed_layer": s.layer.min(),
                        "last_observed_layer": s.layer.max(),
                        "peak_vuln_layer": s.loc[s.vuln_sum.idxmax(), "layer"]
                        if s.vuln_sum.max() > 0
                        else None,
                        "first_vuln_hit_layer": s.loc[s.vuln_sum > 0, "layer"].min(),
                        "sign_changes": int((sign[1:] != sign[:-1]).sum()),
                        "positive_contrast_fraction": (v > 0).mean(),
                    }
                )
            save(pd.DataFrame(trajectories), "trajectories", args.out, args.format)
        if db:
            print("Statistiche token interni/esterni", flush=True)
            for split in ["gt", "outcome", "cwe"]:
                stats = pd.read_sql_query(
                    f"SELECT run,stream,window,layer,{split},token_id,token,groups_text,outside,COUNT(*) AS occurrences,SUM(score) AS score_sum,MIN(rank) AS best_rank,AVG(rank) AS mean_rank_present,SUM(rank=1) AS top1_count FROM tokens GROUP BY run,stream,window,layer,{split},token_id,token,groups_text,outside",
                    db,
                )
                denom = (
                    scalar[scalar.profile == args.profiles[0]]
                    .groupby(["run", "stream", "window", "layer", split])
                    .size()
                    .rename("n_cases")
                    .reset_index()
                )
                stats = stats.merge(
                    denom,
                    on=["run", "stream", "window", "layer", split],
                    validate="many_to_one",
                )
                stats["mean_score_all_cases"] = stats.score_sum / stats.n_cases
                save(stats, "tokens_" + split, args.out, args.format)
        if not args.no_plots:
            make_plots(groups, scalar, summaries, args)
        if sg_rows and not args.no_plots:
            make_sg_plots(groups, summaries, args)
        manifest = {
            "version": VERSION,
            "inputs": inputs,
            "vocab_meta": vocab.compiled["meta"],
            "catalog_sha256": sha(args.vocab / "concepts.json"),
            "token_groups_sha256": sha(args.vocab / "token_groups.json"),
            "script_sha256": sha(Path(__file__)),
            "comparisons": sorted(args.compare),
            "profiles": args.profiles,
            "windows": args.windows,
            "layers": args.layers,
            "anchor": args.anchor,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "signal_epsilon": args.signal_epsilon,
            "uncertainty": "95% pointwise percentile cluster bootstrap; units pair_id/quartet_id, never layer rows; AB entropy is conditional and not calibrated correctness confidence",
            "n_cases": len(cases),
            "n_group_rows": len(groups),
            "observed_additional_ids": vocab.extra,
            "issues": issues,
            "token_database_profile": args.profiles[0],
            "sg_runs": sg_runs,
            "sg_group_rows": len(sg_rows),
            "sg_metrics": "full-vocabulary; mean over positions; raw SG and population-std SG_z",
            "limitations": [
                "score_sum: massa top-k osservata; exact_mass: gruppo completo, normalizzato su tutto V. Il residuo top-k non e classificabile per gruppo senza statistiche esatte.",
                "Finestre >1: somme di massimi, non probabilita.",
                "CI pointwise bootstrap per pair/quartet, esplorative; per-CWE senza CI.",
                "CWE rank ottimistico in caso di pari merito: controllare expected_cwe_ties.",
                "AUROC nelle tabelle dei contrasti e esplorativa, senza soglie selezionate.",
                "Dataset distinti sono trattati come cluster indipendenti nei confronti non appaiati; usare paired_differences per coppie verificate.",
                "Il codice fornito assume logp normalizzati prima del top-k, come nella pipeline originale.",
            ],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        pics = sorted((args.out / "plots").glob("*.png"))
        page = '<html lang="it"><meta charset="utf-8"><title>Code-end metrics</title><style>body{font:15px system-ui;max-width:1300px;margin:32px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Code-end metrics</h1>'
        page = page.replace("Code-end metrics", "Anchor: " + html.escape(args.anchor))
        page += f"<p>{len(cases)} casi. Vocabolario: {html.escape(str(vocab.compiled['meta'].get('version')))}</p><p>Masse osservate nei top-k; finestre multiple: punteggi max. I confronti mancanti restano vuoti. exact_mass usa il gruppo completo e la normalizzazione su tutto V; SG e SG_z usano tutti i logits globali. Le bande mostrano CI 95% pointwise per coppia/quartetto, non variabilita tra singoli esempi.</p>"
        page += (
            "<h2>Tabelle</h2><ul>"
            + "".join(
                f'<li><a href="tables/{html.escape(p.name)}">{html.escape(p.name)}</a></li>'
                for p in sorted((args.out / "tables").iterdir())
            )
            + "</ul>"
        )
        page += "".join(
            f'<h2>{html.escape(p.stem)}</h2><img loading="lazy" src="plots/{html.escape(p.name)}">'
            for p in pics
        )
        page += "<h2>Interpretazione</h2><pre>" + html.escape(HELP) + "</pre></html>"
        (args.out / "report.html").write_text(page)
        print(f"Completato: {args.out.resolve()} ({len(cases)} casi)", flush=True)
    finally:
        if db:
            db.close()


HELP = """
Default: tre analisi complete e confronti appaiati:
01_code_end, 02_mcp_record, 03_mcp_pre_verdict, 04_positions_compare.
Richiede la nuova estrazione con top-500, SG e statistiche A/B per tutte le posizioni.
--anchor code_end permette ancora l'analisi singola dei vecchi JSONL.
Usare sempre --profiles full e --windows 1 per masse probabilistiche.
--vocab indica la cartella con concepts.json e token_groups.json compilati e invariati.
Contrasti fra quattro gruppi: generic, safety, vuln, CWE attesa; sei confronti,
con gruppi sovrapposti e differenze delle medie dei logits centrati.
SG non è probabilità; A/B della lens non è confidenza calibrata.
AUROC GT: vulnerabile contro safe. AUROC errori: errato contro corretto.
Intervalli bootstrap pointwise per coppia/quartetto; risultati esplorativi sullo
stesso campione, senza scelta automatica di layer, soglia o classificatore.
Le posizioni sono state proposte sull'extended: serve validazione indipendente.
La suite e le analisi errori usano CSV gzip. Parquet e disponibile per analisi singole senza errors.
Nuovo default: bootstrap 1000, CI 95% pointwise per pair/quartet; --bootstrap 0 disabilita i CI.
Con --compare all si aggiungono discordance_*, fp_fn_*, error_*, run_quality_with_intervals,
readout_coverage, entropy bounds e una pagina error_review.html per ogni anchor.
Le categorie vuote sono mostrate con n=0; unknown e separato da TP/TN/FP/FN.
--error-examples 6 limita i grafici individuali, non le tabelle.
Il segno vuln-safe e una direzione descrittiva, non un classificatore validato.
1-max(qA,qB) = ambiguita intrinseca A/B; chosen_opposition=1-q(verdetto finale).
L'entropia A/B non e l'entropia dell'intero vocabolario. La massa A+B va letta separatamente.
Le nuove discordanze usano finestra 1, full e tutti i layer selezionati senza privilegiarne uno.
I file con marcatori multipli vengono rifiutati per evitare di mescolare ripetizioni dipendenti.
Nessun modulo locale richiesto dall'analizzatore.
Esempio: python3 analyze_mcp_anchors.py results/qwen38_controlled*v6_mcp500.jsonl --vocab vocab/qwen3.8-27b --out analisi_v6 --require-sg --bootstrap 1000
"""


ANCHOR_FOLDERS = {
    "code_end": "01_code_end",
    "mcp_record": "02_mcp_record",
    "mcp_pre_verdict": "03_mcp_pre_verdict",
}


def _auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    labels, scores = labels[valid], scores[valid]
    p, n = scores[labels == 1], scores[labels == 0]
    if not len(p) or not len(n):
        return np.nan
    return float(
        ((p[:, None] > n).sum() + 0.5 * (p[:, None] == n).sum()) / (len(p) * len(n))
    )


def _auc_interval(labels, scores, clusters, reps, rng):
    labels, scores, clusters = (
        np.asarray(labels),
        np.asarray(scores, float),
        np.asarray(clusters),
    )
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    labels, scores, clusters = labels[valid], scores[valid], clusters[valid]
    point = _auc(labels, scores)
    codes, units = pd.factorize(clusters)
    if not reps or len(units) < 2 or not np.isfinite(point):
        return point, np.nan, np.nan
    weights = np.zeros((reps, len(units)), dtype=int)
    for row, ix in enumerate(rng.integers(0, len(units), (reps, len(units)))):
        weights[row] = np.bincount(ix, minlength=len(units))
    wp, wn = weights[:, codes[labels == 1]], weights[:, codes[labels == 0]]
    p, n = scores[labels == 1], scores[labels == 0]
    order = (p[:, None] > n).astype(float) + 0.5 * (p[:, None] == n)
    denominator = wp.sum(1) * wn.sum(1)
    estimates = np.divide(
        ((wp @ order) * wn).sum(1),
        denominator,
        out=np.full(reps, np.nan),
        where=denominator > 0,
    )
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return point, np.nan, np.nan
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def _table(folder, name):
    return pd.read_csv(folder / "tables" / (name + ".csv.gz"))


def _choice_rows(inputs, anchors=None):
    """Read available A/B scores; invalid predictions remain visible in cases/quality."""
    result, issues = [], []
    columns = [
        "run",
        "case_id",
        "anchor",
        "stream",
        "layer",
        "window",
        "profile",
        "probability_A",
        "probability_B",
        "conditional_A",
        "conditional_B",
        "margin_A_minus_B",
        "pair_probability",
        "conditional_entropy",
        "chosen_probability",
        "chosen_conditional",
        "chosen_margin",
        "uncertainty",
        "chosen_opposition",
        "intrinsic_uncertainty",
        "conditional_entropy_normalized",
        "absolute_margin",
        "error",
    ]
    for path in inputs:
        run = path.name.removesuffix(".gz").removesuffix(".jsonl")
        for _lineno, rec in read_records(path):
            if "_meta" in rec:
                continue
            gt = normalize(rec.get("ground_truth"))
            pred = normalize(rec.get("pred", rec.get("pred_label")))
            chosen = "A" if pred == "vuln" else "B" if pred == "safe" else None
            tree = rec.get("anchor_readout", {})
            if not tree and rec.get("anchor_readout_packed"):
                tree = unpack_readout(rec["anchor_readout_packed"])
            for anchor in anchors or ANCHOR_FOLDERS:
                a = tree.get("anchors", {}).get(anchor, {})
                if not a.get("boundary_exact") or not a.get("choice_scores"):
                    issues.append(
                        {
                            "run": run,
                            "case_id": rec["case_id"],
                            "anchor": anchor,
                            "issue": "missing_or_inexact_choice_readout",
                        }
                    )
                    continue
                scores = a["choice_scores"]
                entries = [
                    ("model", -1, w, v) for w, v in scores.get("model", {}).items()
                ]
                entries += [
                    ("lens", int(l), w, v)
                    for l, ws in scores.get("lens", {}).items()
                    for w, v in ws.items()
                ]
                for stream, layer, window, v in entries:
                    qa = float(v["conditional_probability"]["A"])
                    qb = float(v["conditional_probability"]["B"])
                    pa, pb = float(v["probability"]["A"]), float(v["probability"]["B"])
                    if (
                        not np.isfinite([qa, qb, pa, pb]).all()
                        or min(qa, qb, pa, pb) < 0
                        or max(qa, qb, pa, pb) > 1 + 1e-6
                        or abs(qa + qb - 1) > 1e-5
                        or abs(pa + pb - float(v["pair_probability"])) > 1e-5
                    ):
                        raise ValueError(
                            f"Invalid A/B statistics: {run}/{rec['case_id']}/{anchor}"
                        )
                    margin = float(v["logit_margin_A_minus_B"])
                    entropy = -sum(q * math.log(q) for q in [qa, qb] if q > 0)
                    # For windows>1 probabilities/margins are per-position means. Keep
                    # saved mean entropy: H(mean q) is not mean H(q).
                    saved_entropy = float(v["conditional_entropy_nats"])
                    if int(window) == 1 and abs(entropy - saved_entropy) > 2e-5:
                        raise ValueError(
                            "A/B conditional entropy inconsistent with probabilities"
                        )
                    opposition = (
                        1 - v["conditional_probability"][chosen] if chosen else np.nan
                    )
                    result.append(
                        {
                            "run": run,
                            "case_id": rec["case_id"],
                            "anchor": anchor,
                            "stream": stream,
                            "layer": layer,
                            "window": int(window),
                            "profile": "full",
                            "probability_A": pa,
                            "probability_B": pb,
                            "conditional_A": qa,
                            "conditional_B": qb,
                            "margin_A_minus_B": margin,
                            "pair_probability": v["pair_probability"],
                            "conditional_entropy": saved_entropy,
                            "chosen_probability": v["probability"][chosen]
                            if chosen
                            else np.nan,
                            "chosen_conditional": v["conditional_probability"][chosen]
                            if chosen
                            else np.nan,
                            "chosen_margin": margin
                            if chosen == "A"
                            else -margin
                            if chosen
                            else np.nan,
                            "uncertainty": opposition,  # legacy alias; not intrinsic uncertainty
                            "chosen_opposition": opposition,
                            "intrinsic_uncertainty": 1 - max(qa, qb),
                            "conditional_entropy_normalized": saved_entropy
                            / math.log(2),
                            "absolute_margin": abs(margin),
                            "error": int(gt != pred)
                            if gt in {"safe", "vuln"} and chosen
                            else np.nan,
                        }
                    )
    frame = pd.DataFrame(result, columns=columns)
    frame.attrs["issues"] = issues
    return frame


def _add_anchor_analysis(args, choices):
    cases, sg = _table(args.out, "cases"), _table(args.out, "sg_group_metrics")
    key = ["run", "case_id", "stream", "layer", "window", "profile"]
    wide = sg[sg.group.isin(["generic", "safety", "vuln", "expected_cwe"])].pivot(
        index=key, columns="group", values=["sg", "sg_z", "exact_mass"]
    )
    frames = []
    for left, right in itertools.combinations(
        ["generic", "safety", "vuln", "expected_cwe"], 2
    ):
        d = pd.DataFrame(index=wide.index)
        for metric in ["sg", "sg_z", "exact_mass"]:
            d["delta_" + metric] = wide[(metric, left)] - wide[(metric, right)]
        d = d.reset_index()
        d["contrast"] = left + "_minus_" + right
        frames.append(d)
    cross = pd.concat(frames, ignore_index=True).merge(
        cases, on=["run", "case_id"], validate="many_to_one"
    )
    save(cross, "vocabulary_cross_case_metrics", args.out, args.format)
    rng = np.random.default_rng(args.seed)
    base = ["run", "stream", "layer", "window", "profile", "contrast"]
    for mode, extra in [
        ("overview", []),
        ("gt", ["gt"]),
        ("confusion", ["outcome"]),
        ("cwe", ["cwe"]),
        ("datasets", ["family", "variant"]),
    ]:
        save(
            summarize(
                cross,
                base + extra,
                ["delta_sg", "delta_sg_z", "delta_exact_mass"],
                0 if mode == "cwe" else args.bootstrap,
                rng,
            ),
            "vocabulary_cross_" + mode,
            args.out,
            args.format,
        )
    save(
        contrasts(
            cross,
            base,
            "gt",
            [("vuln", "safe")],
            ["delta_sg", "delta_sg_z", "delta_exact_mass"],
            args.bootstrap,
            rng,
        ),
        "vocabulary_cross_contrasts_gt",
        args.out,
        args.format,
    )
    save(
        contrasts(
            cross,
            base,
            "outcome",
            [("FP", "TN"), ("FN", "TP")],
            ["delta_sg", "delta_sg_z", "delta_exact_mass"],
            args.bootstrap,
            rng,
        ),
        "vocabulary_cross_error_vs_correct",
        args.out,
        args.format,
    )
    choice_issues = choices.attrs.get("issues", [])
    choices = choices[choices.anchor == args.anchor].copy()
    selected = set(map(tuple, sg[key].drop_duplicates().to_numpy()))
    choices = choices[
        [tuple(row) in selected for row in choices[key].to_numpy()]
    ].merge(cases, on=["run", "case_id"], validate="many_to_one")
    if choices.duplicated(key).any():
        raise ValueError("Duplicate A/B score observation")
    actual = set(map(tuple, choices[key].to_numpy()))
    coverage = pd.DataFrame(
        [
            dict(zip(key, row, strict=True)) | {"choice_available": row in actual}
            for row in sorted(selected)
        ]
    )
    save(coverage, "choice_coverage", args.out, args.format)
    save(pd.DataFrame(choice_issues), "choice_quality_issues", args.out, args.format)
    save(choices, "choice_case_metrics", args.out, args.format)
    metrics = [
        "probability_A",
        "probability_B",
        "conditional_A",
        "conditional_B",
        "margin_A_minus_B",
        "pair_probability",
        "conditional_entropy",
        "chosen_probability",
        "chosen_conditional",
        "chosen_margin",
        "uncertainty",
        "chosen_opposition",
        "intrinsic_uncertainty",
        "conditional_entropy_normalized",
        "absolute_margin",
    ]
    for mode, extra in [
        ("overview", []),
        ("gt", ["gt"]),
        ("confusion", ["outcome"]),
        ("cwe", ["cwe"]),
        ("datasets", ["family", "variant"]),
    ]:
        save(
            summarize(
                choices,
                ["run", "stream", "layer", "window", "profile"] + extra,
                metrics,
                0 if mode == "cwe" else args.bootstrap,
                rng,
            ),
            "choice_" + mode,
            args.out,
            args.format,
        )
    scalar = _table(args.out, "case_layer_metrics").merge(
        cases, on=["run", "case_id"], validate="many_to_one"
    )
    auc_rows = []
    for source, metrics, target in [
        (scalar, ["sg_polarity_difference", "exact_mass_polarity_contrast"], "gt"),
        (choices, ["margin_A_minus_B"], "gt"),
        (
            choices,
            ["chosen_opposition", "intrinsic_uncertainty", "conditional_entropy"],
            "error",
        ),
    ]:
        for k, g in source.groupby(["run", "stream", "layer", "window", "profile"]):
            labels = (g["gt"] == "vuln").astype(int) if target == "gt" else g.error
            for metric in metrics:
                point, low, high = _auc_interval(
                    labels, g[metric], g.cluster, args.bootstrap, rng
                )
                auc_rows.append(
                    dict(
                        zip(
                            ["run", "stream", "layer", "window", "profile"],
                            k,
                            strict=True,
                        )
                    )
                    | {
                        "target": target,
                        "metric": metric,
                        "auroc": point,
                        "ci_low": low,
                        "ci_high": high,
                        "n": int(
                            (np.isfinite(g[metric]) & np.isin(labels, [0, 1])).sum()
                        ),
                        "n_positive": int(
                            ((labels == 1) & np.isfinite(g[metric])).sum()
                        ),
                        "n_negative": int(
                            ((labels == 0) & np.isfinite(g[metric])).sum()
                        ),
                        "status": "exploratory_same_sample",
                    }
                )
    auc_df = pd.DataFrame(auc_rows)
    save(auc_df, "auroc_with_intervals", args.out, args.format)
    if not args.no_plots:
        import matplotlib.pyplot as plt

        for run, g in auc_df[auc_df.stream == "lens"].groupby("run"):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, target in zip(axes, ["gt", "error"], strict=True):
                for metric, h in g[(g.target == target) & (g.window == 1)].groupby(
                    "metric"
                ):
                    h = h.sort_values("layer")
                    ax.plot(h.layer, h.auroc, label=metric)
                    ax.fill_between(h.layer, h.ci_low, h.ci_high, alpha=0.12)
                ax.axhline(0.5, color="gray", ls=":")
                ax.set(
                    title="AUROC: " + target,
                    xlabel="Layer",
                    ylabel="AUROC",
                    ylim=(0, 1),
                )
                ax.legend(fontsize=7)
            fig.suptitle(args.anchor + " / " + run + " / exploratory")
            fig.tight_layout()
            fig.savefig(args.out / "plots" / (run + "_auroc_targets.png"), dpi=150)
            plt.close(fig)
    if not args.no_plots:
        import matplotlib.pyplot as plt

        cross_gt = _table(args.out, "vocabulary_cross_gt")
        for run, all_rows in cross_gt[
            (cross_gt.stream == "lens")
            & (cross_gt.window == 1)
            & (cross_gt.metric == "delta_sg")
        ].groupby("run"):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            for ax, (contrast, sub) in zip(
                axes.flat, all_rows.groupby("contrast"), strict=True
            ):
                for gt, points in sub.groupby("gt"):
                    points = points.sort_values("layer")
                    ax.plot(points.layer, points["mean"], label="GT " + gt)
                    ax.fill_between(
                        points.layer, points.ci_low, points.ci_high, alpha=0.12
                    )
                ax.axhline(0, color="gray", lw=0.8)
                ax.set(
                    title=contrast, xlabel="Layer", ylabel="Differenza logits centrati"
                )
                ax.legend(fontsize=8)
            fig.suptitle(args.anchor + " / " + run + " / vocabolari sovrapposti")
            fig.tight_layout()
            fig.savefig(args.out / "plots" / (run + "_vocabulary_cross.png"), dpi=150)
            plt.close(fig)
    if not args.no_plots:
        import matplotlib.pyplot as plt

        choice_summary = _table(args.out, "choice_confusion")
        colors = {"TP": "#c63333", "TN": "#278348", "FP": "#d28b21", "FN": "#2674bc"}
        for run, all_rows in choice_summary[
            (choice_summary.stream == "lens") & (choice_summary.window == 1)
        ].groupby("run"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for ax, metric in zip(
                axes.flat,
                [
                    "chosen_conditional",
                    "pair_probability",
                    "chosen_margin",
                    "conditional_entropy",
                ],
                strict=True,
            ):
                for outcome in plot_categories("outcome", all_rows.outcome.unique()):
                    points = all_rows[
                        (all_rows.metric == metric) & (all_rows.outcome == outcome)
                    ].sort_values("layer")
                    ax.plot(
                        points.layer,
                        points["mean"],
                        color=colors.get(outcome, "gray"),
                        label=category_label(args, run, "outcome", outcome, choices),
                    )
                    ax.fill_between(
                        points.layer,
                        points.ci_low,
                        points.ci_high,
                        alpha=0.1,
                        color=colors.get(outcome, "gray"),
                    )
                ax.set(title=metric, xlabel="Layer")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.15)
            fig.suptitle(
                args.anchor
                + " / "
                + run
                + " / sostegno al verdetto, non confidenza calibrata"
            )
            fig.tight_layout()
            fig.savefig(args.out / "plots" / (run + "_choice_support.png"), dpi=150)
            plt.close(fig)
    report = args.out / "report.html"
    extra = "<h2>Estensioni</h2><p>Tabelle vocabulary_cross_*: sei contrasti fra gruppi sovrapposti. choice_*: A/B espliciti. auroc_with_intervals: target GT ed errore distinti, CI pointwise bootstrap per cluster. Nessuna validazione held-out o selezione automatica del layer migliore.</p>"
    if not args.no_plots:
        extra += "".join(
            '<img style="max-width:100%" src="plots/' + p.name + '">'
            for p in sorted((args.out / "plots").glob("*.png"))
            if p.name.endswith(
                ("_auroc_targets.png", "_vocabulary_cross.png", "_choice_support.png")
            )
        )
    current = report.read_text()
    report.write_text(insert_report_section(current, extra))
    return sg, scalar, choices


def analyze_suite(args):
    if args.format != "csv":
        raise ValueError("The three-anchor suite requires --format csv")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new or empty output folder")
    choices = _choice_rows(args.inputs)
    outputs = {}
    for anchor, folder in ANCHOR_FOLDERS.items():
        local = argparse.Namespace(**vars(args))
        local.anchor, local.out = anchor, args.out / folder
        analyze(local)
        outputs[anchor] = _add_anchor_analysis(local, choices)
        if "errors" in local.compare:
            add_error_analysis(local, outputs[anchor][2])
    out = args.out / "04_positions_compare"
    out.mkdir()
    rng = np.random.default_rng(args.seed)
    for index, name in enumerate(["groups", "scalar", "choices"]):
        frames = []
        for left, right in itertools.combinations(ANCHOR_FOLDERS, 2):
            a, b = outputs[left][index], outputs[right][index]
            keys = ["run", "case_id", "stream", "layer", "window", "profile"] + (
                ["group"] if index == 0 else []
            )
            metrics = (
                ["sg", "sg_z", "exact_mass"]
                if index == 0
                else [
                    "sg_polarity_difference",
                    "exact_mass_polarity_contrast",
                    "union_sum",
                    "outside_sum",
                ]
                if index == 1
                else [
                    "margin_A_minus_B",
                    "chosen_margin",
                    "chosen_conditional",
                    "pair_probability",
                    "uncertainty",
                ]
            )
            m = b.merge(
                a[keys + metrics],
                on=keys,
                suffixes=("", "_reference"),
                validate="one_to_one",
            )
            m["comparison"] = right + "_minus_" + left
            if index == 0:
                m = m.merge(
                    _table(args.out / ANCHOR_FOLDERS[left], "cases"),
                    on=["run", "case_id"],
                    validate="many_to_one",
                )
            for metric in metrics:
                m["delta_" + metric] = m[metric] - m[metric + "_reference"]
            frames.append(m)
        d = pd.concat(frames, ignore_index=True)
        save(d, "paired_" + name + "_differences", out, "csv")
        base = ["comparison", "run", "stream", "layer", "window", "profile"] + (
            ["group"] if index == 0 else []
        )
        for mode, extra in [
            ("overview", []),
            ("gt", ["gt"]),
            ("confusion", ["outcome"]),
            ("cwe", ["cwe"]),
        ]:
            save(
                summarize(
                    d,
                    base + extra,
                    ["delta_" + m for m in metrics],
                    0 if mode == "cwe" else args.bootstrap,
                    rng,
                ),
                "paired_" + name + "_" + mode,
                out,
                "csv",
            )
    if "errors" in args.compare:
        compare_discordance_positions(args, out)
    auc = []
    for anchor, folder in ANCHOR_FOLDERS.items():
        d = _table(args.out / folder, "auroc_with_intervals")
        d["anchor"] = anchor
        auc.append(d)
    save(pd.concat(auc, ignore_index=True), "auroc_positions", out, "csv")
    if not args.no_plots:
        import matplotlib.pyplot as plt

        d = _table(out, "paired_groups_gt")
        colors = {
            "generic": "#777777",
            "safety": "#278348",
            "vuln": "#c63333",
            "expected_cwe": "#2674bc",
        }
        for run, all_rows in d[
            (d.stream == "lens") & (d.window == 1) & d.group.isin(colors)
        ].groupby("run"):
            comparisons = sorted(all_rows.comparison.unique())
            fig, axes = plt.subplots(
                len(comparisons), 2, figsize=(13, 4 * len(comparisons)), squeeze=False
            )
            for row, comparison in enumerate(comparisons):
                for col, metric in enumerate(["delta_exact_mass", "delta_sg"]):
                    ax = axes[row, col]
                    a = all_rows[
                        (all_rows.comparison == comparison)
                        & (all_rows.metric == metric)
                    ]
                    for (group, gt), points in a.groupby(["group", "gt"]):
                        points = points.sort_values("layer")
                        scale = 100 if metric == "delta_exact_mass" else 1
                        ax.plot(
                            points.layer,
                            points["mean"] * scale,
                            color=colors[group],
                            ls="-" if gt == "vuln" else "--",
                            label=group + " / GT " + gt,
                        )
                        ax.fill_between(
                            points.layer.to_numpy(float),
                            points.ci_low.to_numpy(float) * scale,
                            points.ci_high.to_numpy(float) * scale,
                            color=colors[group],
                            alpha=0.10,
                        )
                    ax.axhline(0, color="gray", lw=0.8)
                    ax.set(
                        title=comparison,
                        xlabel="Layer",
                        ylabel="Differenza massa (punti percentuali)"
                        if metric == "delta_exact_mass"
                        else "Differenza logits centrati",
                    )
                    legend_if_present(ax, fontsize=7)
                    ax.grid(alpha=0.15)
            fig.suptitle(run + " — differenze appaiate, tutti i layer")
            fig.tight_layout()
            out.joinpath("plots").mkdir(exist_ok=True)
            fig.savefig(out / "plots" / (run + "_position_differences.png"), dpi=150)
            plt.close(fig)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "anchors": ANCHOR_FOLDERS,
                "seed": args.seed,
                "bootstrap": args.bootstrap,
                "pairing": "same run and case_id across anchors",
                "auc_status": "exploratory, pointwise cluster intervals",
            },
            indent=2,
        )
    )
    notes = "Nuove tabelle discordance_positions_*: direzioni concordi/opposte e differenze appaiate nelle quote di sostegno alla GT e al verdetto. Differenze appaiate sullo stesso esempio: posizione destra meno riferimento. Nessun layer principale. AUROC GT: vulnerabile contro safe; AUROC errori: errato contro corretto. CI pointwise, non simultanei. Punti scelti esplorativamente: risultati non held-out. Vocabolari sovrapposti; generic e union possono coincidere. A/B su code_end non ha lo stesso ruolo next-token di pre-verdetto. Finestre >1: SG e A/B sono medie per posizione, top-k usa massimi e non è massa."
    (out / "report.html").write_text(
        '<meta charset="utf-8"><h1>Confronti fra posizioni</h1><p>'
        + html.escape(notes)
        + "</p>"
        + "".join(
            '<img style="max-width:100%" src="plots/' + p.name + '">'
            for p in sorted((out / "plots").glob("*.png"))
        )
    )
    (args.out / "report.html").write_text(
        '<meta charset="utf-8"><h1>Analisi MCP</h1><ul>'
        + "".join(
            '<li><a href="' + folder + '/report.html">' + anchor + "</a></li>"
            for anchor, folder in ANCHOR_FOLDERS.items()
        )
        + '<li><a href="04_positions_compare/report.html">Confronti appaiati</a></li></ul>'
    )


# Uncertainty, descriptive discordances and error review (CPU-only).
OUTCOMES = ["TP", "FP", "TN", "FN"]
OUTCOME_COLORS = {"TP": "#c63333", "FP": "#d28b21", "TN": "#278348", "FN": "#2674bc"}
SIGNAL_TITLES = {
    "sg_polarity_difference": "SG vuln − safe (logits)",
    "sg_z_polarity_difference": "SG_z vuln − safe",
    "exact_mass_polarity_contrast": "Contrasto massa esatta vuln − safe",
    "polarity_contrast": "Contrasto massa osservata vuln − safe",
    "ab_margin": "Margine A − B",
}


def legend_if_present(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **kwargs)


def legend_from_axes(axes, **kwargs):
    for ax in axes:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            legend_if_present(ax, **kwargs)
            break


def plot_categories(field, values):
    values = set(values)
    return (
        OUTCOMES + (["unknown"] if "unknown" in values else [])
        if field == "outcome"
        else sorted(values)
    )


def category_label(args, run, field, category, available=None):
    cases = getattr(args, "case_counts", pd.DataFrame())
    if cases.empty:
        return str(category)
    selected = cases[(cases.run == run) & (cases[field] == category)]
    n = selected.case_id.nunique()
    used = n
    if available is not None and "case_id" in available and field in available:
        used = available.loc[
            (available.run == run) & (available[field] == category), "case_id"
        ].nunique()
    return f"{category} (n={n}" + (f"; letti={used}" if used != n else "") + ")"


def insert_report_section(current, extra):
    for closing in ["</body>", "</html>"]:
        if closing in current:
            return current.replace(closing, extra + closing, 1)
    return current + extra


def quality_tables(cases, args):
    """Case counts never use rows multiplied by layers, windows or vocabularies."""
    counts, rows = [], []
    rng = np.random.default_rng(args.seed)
    for run, frame in cases.groupby("run"):
        for outcome in OUTCOMES + ["unknown"]:
            sub = frame[frame.outcome == outcome]
            counts.append(
                {
                    "run": run,
                    "outcome": outcome,
                    "n": len(sub),
                    "n_exact_anchor": int(sub.anchor_exact.sum()),
                    "n_clusters": sub.cluster.nunique(),
                    "n_cwe": sub.cwe.nunique(),
                }
            )
        valid_gt = frame["gt"].isin(["safe", "vuln"])
        classified = frame.prediction.isin(["safe", "vuln"])
        d = frame.copy()
        d["accuracy_all_known_gt"] = np.where(
            valid_gt, (frame["gt"] == frame.prediction).astype(float), np.nan
        )
        d["accuracy_classified"] = np.where(
            valid_gt & classified,
            (frame["gt"] == frame.prediction).astype(float),
            np.nan,
        )
        d["prediction_coverage"] = np.where(valid_gt, classified.astype(float), np.nan)
        d["invalid_prediction_rate"] = np.where(
            valid_gt, (~classified).astype(float), np.nan
        )
        d["exact_anchor_coverage"] = frame.anchor_exact.astype(float)
        d["truncated_rate"] = frame.truncated.map({True: 1.0, False: 0.0})
        d["carrier_valid_rate"] = frame.carrier_ok.map({True: 1.0, False: 0.0})
        d["recall_classified"] = np.where(
            valid_gt & classified & (frame["gt"] == "vuln"),
            (frame.prediction == "vuln").astype(float),
            np.nan,
        )
        d["specificity_classified"] = np.where(
            valid_gt & classified & (frame["gt"] == "safe"),
            (frame.prediction == "safe").astype(float),
            np.nan,
        )
        d["fp_rate_classified_safe"] = np.where(
            valid_gt & classified & (frame["gt"] == "safe"),
            (frame.prediction == "vuln").astype(float),
            np.nan,
        )
        d["fn_rate_classified_vuln"] = np.where(
            valid_gt & classified & (frame["gt"] == "vuln"),
            (frame.prediction == "safe").astype(float),
            np.nan,
        )
        metrics = [c for c in d.columns if c not in frame.columns]
        rows.append(summarize(d, ["run"], metrics, args.bootstrap, rng))
    save(pd.DataFrame(counts), "outcome_counts", args.out, args.format)
    save(
        pd.concat(rows, ignore_index=True),
        "run_quality_with_intervals",
        args.out,
        args.format,
    )
    save(
        cases.groupby(["run", "cwe", "outcome"], dropna=False)
        .size()
        .rename("n")
        .reset_index(),
        "outcome_counts_by_cwe",
        args.out,
        args.format,
    )


def _finite_sign(values, epsilon):
    x = np.asarray(values, float)
    return np.where(
        np.isfinite(x),
        np.where(x > epsilon, 1.0, np.where(x < -epsilon, -1.0, 0.0)),
        np.nan,
    )


def build_discordance_rows(scalar, choices, epsilon):
    """Direction is prespecified: positive = vulnerable/A. Zero is no direction."""
    key = ["run", "case_id", "stream", "layer", "window", "profile"]
    frames = []
    for metric in SIGNAL_TITLES:
        if metric == "ab_margin":
            if choices.empty:
                continue
            frame = choices.copy()
            frame["value"] = frame.margin_A_minus_B
            frame = frame[frame.window == 1].copy()
        else:
            if metric not in scalar:
                continue
            frame = scalar.copy()
            frame["value"] = frame[metric]
            # Only single-position top-k sums are probability masses.
            frame = frame[frame.window == 1].copy()
        frame["signal"] = metric
        frame["sign"] = _finite_sign(frame.value, epsilon)
        frame["signal_label"] = np.select(
            [frame.sign > 0, frame.sign < 0, frame.sign == 0],
            ["vuln", "safe", "neutral"],
            default="unavailable",
        )
        pred_valid = frame.prediction.isin(["safe", "vuln"])
        gt_valid = frame["gt"].isin(["safe", "vuln"])
        directional = frame.sign.notna() & (frame.sign != 0)
        finite = frame.sign.notna()
        # Ties count as neutral, not as agreement. Missing observations stay NaN.
        frame["neutral"] = np.where(finite, (frame.sign == 0).astype(float), np.nan)
        frame["opposes_final"] = np.where(
            finite & pred_valid,
            (directional & (frame.signal_label != frame.prediction)).astype(float),
            np.nan,
        )
        frame["agrees_gt"] = np.where(
            finite & gt_valid,
            (directional & (frame.signal_label == frame["gt"])).astype(float),
            np.nan,
        )
        frame["agrees_final"] = np.where(
            finite & pred_valid,
            (directional & (frame.signal_label == frame.prediction)).astype(float),
            np.nan,
        )
        frame["opposes_final_directional"] = np.where(
            directional & pred_valid,
            (frame.signal_label != frame.prediction).astype(float),
            np.nan,
        )
        frame["error"] = np.where(
            gt_valid & pred_valid,
            (frame["gt"] != frame.prediction).astype(float),
            np.nan,
        )
        frame["error_counterevidence"] = np.where(
            (frame.error == 1) & finite, frame.agrees_gt, np.nan
        )
        frame["supports_wrong_verdict"] = np.where(
            (frame.error == 1) & finite, frame.agrees_final, np.nan
        )
        frame["correct_but_opposed"] = np.where(
            (frame.error == 0) & finite, frame.opposes_final, np.nan
        )
        frame["direction_rule"] = (
            "positive=vuln/A; negative=safe/B; neutral within numerical epsilon"
        )
        if frame.duplicated(key).any():
            raise ValueError(f"Duplicate signal observations: {metric}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def entropy_bounds(scalar, groups):
    """Bounds, not reconstructed full-vocabulary entropy, from a saved subset."""
    key = ["run", "case_id", "stream", "layer", "window", "profile"]
    columns = key + [
        "saved_k",
        "saved_sum",
        "residual_mass",
        "top1_probability_or_score",
    ]
    out = scalar[scalar.window == 1][columns].copy()
    union = groups[groups.group == "union"]
    if "vocab_size" not in union:
        out["status"] = "vocab_size_not_saved; entropy bounds unavailable"
        return out
    out = out.merge(union[key + ["vocab_size"]], on=key, validate="one_to_one")
    # group entropies use each token once through disjoint union/outside masks.
    parts = groups[
        (groups.window == 1) & groups.group.isin(["union", "outside"])
    ].copy()
    p = parts.score_sum.to_numpy(float)
    ent = parts.entropy_conditional.fillna(0).to_numpy(float)
    parts["entropy_observed_part"] = np.where(
        p > 0, p * ent - p * np.log(np.maximum(p, 1e-300)), 0
    )
    observed = (
        parts.groupby(key)
        .entropy_observed_part.sum()
        .rename("entropy_observed_nats")
        .reset_index()
    )
    out = out.merge(observed, on=key, validate="one_to_one")
    r = out.residual_mass.to_numpy(float)
    n = out.vocab_size.to_numpy(float) - out.saved_k.to_numpy(float)
    lumped = np.where(r > 0, -r * np.log(np.maximum(r, 1e-300)), 0)
    out["full_entropy_lower_nats"] = out.entropy_observed_nats + lumped
    out["full_entropy_upper_nats"] = out.full_entropy_lower_nats + np.where(
        n > 0, r * np.log(np.maximum(n, 1)), 0
    )
    invalid = ~np.isfinite(n) | (n < 0) | ((n == 0) & (r > 1e-5))
    out.loc[invalid, ["full_entropy_lower_nats", "full_entropy_upper_nats"]] = np.nan
    out["status"] = np.where(
        invalid,
        "invalid_or_missing_vocab_size",
        "bounds_only; residual lumped vs uniformly spread",
    )
    return out


def _sign_trajectory(g):
    g = g.sort_values("layer")
    v = g[g.sign.notna()]
    directional = v[v.sign != 0]
    signs = directional.sign.to_numpy(float)
    return {
        "layers_available": len(v),
        "layers_directional": len(directional),
        "first_observed_layer": int(g.layer.min()),
        "last_observed_layer": int(g.layer.max()),
        "mean_signal": g.value.mean(),
        "median_signal": g.value.median(),
        "fraction_layers_opposing_final": v.opposes_final.mean(),
        "fraction_layers_agreeing_gt": v.agrees_gt.mean(),
        "fraction_layers_neutral": v.neutral.mean(),
        "any_layer_opposes_final": bool((v.opposes_final == 1).any()),
        "all_available_layers_oppose_final": bool(
            len(v) and (v.opposes_final == 1).all()
        ),
        "sign_changes_ignoring_neutral": int((signs[1:] != signs[:-1]).sum()),
        "note": "descriptive selected-layer trajectory; layers are not independent samples",
    }


def add_error_analysis(args, choices):
    cases = _table(args.out, "cases")
    args.case_counts = cases
    scalar = _table(args.out, "case_layer_metrics").merge(
        cases, on=["run", "case_id"], validate="many_to_one"
    )
    groups = _table(args.out, "group_metrics").merge(
        cases, on=["run", "case_id"], validate="many_to_one"
    )
    signals = build_discordance_rows(scalar, choices, args.signal_epsilon)
    if signals.empty:
        return
    coverage = (
        scalar.groupby(
            ["run", "stream", "window", "profile", "layer", "outcome"], dropna=False
        )
        .agg(
            n_cases=("case_id", "nunique"),
            n_clusters=("cluster", "nunique"),
            saved_k_min=("saved_k", "min"),
            saved_k_max=("saved_k", "max"),
            saved_mass_mean=("saved_sum", "mean"),
            residual_mass_mean=("residual_mass", "mean"),
        )
        .reset_index()
    )
    save(coverage, "readout_coverage", args.out, args.format)
    # Keep a compact, auditable long table rather than replicating every feature.
    keys = ["run", "case_id", "stream", "layer", "window", "profile", "signal"]
    fields = [
        "value",
        "sign",
        "signal_label",
        "gt",
        "prediction",
        "outcome",
        "cwe",
        "cluster",
        "neutral",
        "opposes_final",
        "agrees_gt",
        "agrees_final",
        "opposes_final_directional",
        "error",
        "error_counterevidence",
        "supports_wrong_verdict",
        "correct_but_opposed",
        "direction_rule",
    ]
    signals = signals[keys + fields]
    save(signals, "discordance_case_layer", args.out, args.format)
    rng = np.random.default_rng(args.seed)
    metrics = [
        "value",
        "neutral",
        "opposes_final",
        "agrees_gt",
        "agrees_final",
        "opposes_final_directional",
        "error_counterevidence",
        "supports_wrong_verdict",
        "correct_but_opposed",
    ]
    base = ["run", "stream", "layer", "window", "profile", "signal"]
    for mode, extra in [
        ("overview", []),
        ("gt", ["gt"]),
        ("confusion", ["outcome"]),
        ("cwe", ["cwe"]),
    ]:
        summary = summarize(
            signals, base + extra, metrics, 0 if mode == "cwe" else args.bootstrap, rng
        )
        save(summary, "discordance_" + mode, args.out, args.format)
    save(
        contrasts(
            signals,
            base,
            "outcome",
            [("FP", "TN"), ("FN", "TP")],
            ["value", "opposes_final", "agrees_gt", "neutral"],
            args.bootstrap,
            rng,
        ),
        "error_vs_correct_same_gt",
        args.out,
        args.format,
    )
    # This table separates statistical uncertainty from per-example ambiguity.
    bounds = entropy_bounds(scalar, groups)
    save(bounds, "token_distribution_entropy_bounds", args.out, args.format)
    if not choices.empty:
        uncertainties = [
            "intrinsic_uncertainty",
            "conditional_entropy_normalized",
            "absolute_margin",
            "chosen_opposition",
            "pair_probability",
            "chosen_conditional",
        ]
        save(
            contrasts(
                choices,
                ["run", "stream", "layer", "window", "profile"],
                "outcome",
                [("FP", "TN"), ("FN", "TP")],
                uncertainties,
                args.bootstrap,
                rng,
            ),
            "error_uncertainty_vs_correct_same_gt",
            args.out,
            args.format,
        )
    trajectories = []
    tkeys = ["run", "case_id", "stream", "window", "profile", "signal"]
    for key, frame in signals[signals.stream == "lens"].groupby(tkeys):
        trajectories.append(
            dict(zip(tkeys, key, strict=True))
            | _sign_trajectory(frame)
            | {
                c: frame.iloc[0][c]
                for c in ["gt", "prediction", "outcome", "cwe", "cluster"]
            }
        )
    trajectories = pd.DataFrame(trajectories)
    save(trajectories, "discordance_trajectories", args.out, args.format)
    save(
        trajectories[trajectories.outcome.isin(["FP", "FN"])],
        "fp_fn_trajectories",
        args.out,
        args.format,
    )
    error_keys = cases.loc[cases.outcome.isin(["FP", "FN"]), ["run", "case_id"]]
    error_layers = scalar.merge(
        error_keys, on=["run", "case_id"], validate="many_to_one"
    )
    if not choices.empty:
        abcols = [
            "margin_A_minus_B",
            "chosen_margin",
            "chosen_conditional",
            "chosen_opposition",
            "intrinsic_uncertainty",
            "conditional_entropy",
            "pair_probability",
        ]
        error_layers = error_layers.merge(
            choices[keys[:-1] + abcols], on=keys[:-1], how="left", validate="one_to_one"
        )
    for group in ["safety", "vuln", "expected_cwe", "outside"]:
        cols = [
            c
            for c in [
                "best_token",
                "best_score",
                "first_rank",
                "score_sum",
                "sg",
                "sg_z",
                "exact_mass",
            ]
            if c in groups
        ]
        selected = groups[groups.group == group][keys[:-1] + cols].rename(
            columns={c: group + "_" + c for c in cols}
        )
        error_layers = error_layers.merge(
            selected, on=keys[:-1], how="left", validate="one_to_one"
        )
    save(error_layers, "fp_fn_layer_details", args.out, args.format)
    save(
        cases[cases.outcome.isin(["FP", "FN", "unknown"])],
        "error_case_index",
        args.out,
        args.format,
    )
    if not args.no_plots:
        plot_error_analysis(args, signals, choices, error_layers)
    _write_error_review(args, cases, error_layers, trajectories)


def plot_error_analysis(args, signals, choices, error_layers):
    import matplotlib.pyplot as plt

    summary = _table(args.out, "discordance_confusion")
    folder = args.out / "plots"
    folder.mkdir(exist_ok=True)
    for (run, window, profile), frame in summary[summary.stream == "lens"].groupby(
        ["run", "window", "profile"]
    ):
        for metric, ylabel in [
            ("opposes_final", "Quota che contrasta il verdetto"),
            ("agrees_gt", "Quota nella direzione della GT"),
        ]:
            fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), sharey=True)
            for ax, outcome in zip(axes, OUTCOMES, strict=True):
                part = frame[(frame.outcome == outcome) & (frame.metric == metric)]
                for signal, points in part.groupby("signal"):
                    points = points.sort_values("layer")
                    ax.plot(points.layer, points["mean"], label=SIGNAL_TITLES[signal])
                    ax.fill_between(
                        points.layer.to_numpy(float),
                        points.ci_low.to_numpy(float),
                        points.ci_high.to_numpy(float),
                        alpha=0.12,
                    )
                ax.set(
                    title=category_label(args, run, "outcome", outcome, signals),
                    xlabel="Layer",
                    ylim=(-0.03, 1.03),
                )
                ax.grid(alpha=0.2)
            axes[0].set_ylabel(ylabel)
            legend_from_axes(axes, fontsize=6)
            fig.suptitle(
                f"{args.anchor} / {run} — direzioni descrittive; CI 95% pointwise per coppia"
            )
            fig.tight_layout()
            fig.savefig(
                folder / f"{run}_w{window}_{profile}_discordance_{metric}.png", dpi=145
            )
            plt.close(fig)
    if not choices.empty:
        summary = (
            _table(args.out, "choice_confusion")
            if (args.out / "tables" / "choice_confusion.csv.gz").exists()
            else None
        )
        # All four outcomes are panels, including n=0; counts come from JSONL cases.
        for run, frame in choices[
            (choices.stream == "lens") & (choices.window == 1)
        ].groupby("run"):
            fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), sharey=True)
            selected_metrics = [
                "intrinsic_uncertainty",
                "conditional_entropy_normalized",
                "chosen_opposition",
                "pair_probability",
            ]
            stats = summarize(
                frame,
                ["outcome", "layer"],
                selected_metrics,
                args.bootstrap,
                np.random.default_rng(args.seed),
            )
            for ax, outcome in zip(axes, OUTCOMES, strict=True):
                for metric in selected_metrics:
                    part = stats[
                        (stats.outcome == outcome) & (stats.metric == metric)
                    ].sort_values("layer")
                    ax.plot(part.layer, part["mean"], label=metric)
                    ax.fill_between(
                        part.layer.to_numpy(float),
                        part.ci_low.to_numpy(float),
                        part.ci_high.to_numpy(float),
                        alpha=0.1,
                    )
                ax.set(
                    title=category_label(args, run, "outcome", outcome, frame),
                    xlabel="Layer",
                    ylim=(-0.03, 1.03),
                )
                ax.grid(alpha=0.2)
            axes[0].set_ylabel("Misure A/B (non calibrate)")
            legend_from_axes(axes, fontsize=6)
            fig.suptitle(
                f"{args.anchor} / {run} — ambiguità A/B e opposizione al verdetto sono misure diverse"
            )
            fig.tight_layout()
            fig.savefig(folder / f"{run}_choice_uncertainty_confusion.png", dpi=145)
            plt.close(fig)
    # Deterministic review subset; all errors stay in the exported tables.
    for (run, outcome), frame in error_layers[
        (error_layers.stream == "lens") & (error_layers.window == 1)
    ].groupby(["run", "outcome"]):
        for cid in sorted(frame.case_id.unique())[: args.error_examples]:
            g = frame[frame.case_id == cid].sort_values("layer")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for metric in ["exact_mass_polarity_contrast", "polarity_contrast"]:
                if metric in g:
                    axes[0].plot(g.layer, g[metric], label=metric)
            for metric in ["sg_polarity_difference"]:
                if metric in g:
                    axes[1].plot(g.layer, g[metric], label=metric)
            axes[1].set_ylabel("Differenza in unita di logit")
            if "chosen_conditional" in g:
                for metric in [
                    "chosen_conditional",
                    "pair_probability",
                    "intrinsic_uncertainty",
                ]:
                    axes[2].plot(g.layer, g[metric], label=metric)
                axes[2].set_ylim(-0.03, 1.03)
            for ax in axes:
                ax.axhline(0, color="gray", lw=0.7)
                ax.set_xlabel("Layer")
                ax.legend(fontsize=6)
                ax.grid(alpha=0.2)
            fig.suptitle(
                f"{outcome} / {cid} / {args.anchor} — GT={g.iloc[0]['gt']}, risposta={g.iloc[0].prediction}"
            )
            fig.tight_layout()
            token = hashlib.sha256((run + "\0" + cid).encode()).hexdigest()[:12]
            fig.savefig(folder / f"{run}_error_{outcome}_{token}.png", dpi=145)
            plt.close(fig)


def _write_error_review(args, cases, details, trajectories):
    notes = """Le bande sono intervalli bootstrap al 95% per coppia/quartetto, pointwise: non sono intervalli di previsione per un singolo esempio e non correggono la selezione su molti layer. Le curve per-CWE restano descrittive (campioni piccoli). I layer non vengono trattati come osservazioni indipendenti.
Una discordanza e definita come segno del contrasto opposto al verdetto finale: non dimostra da sola una contraddizione semantica o un errore causale. SG e massa esatta possono avere segni diversi. Safe/vuln possono sovrapporsi; non sono due classi probabilistiche disgiunte. I segnali nulli restano neutrali, non sono voti safe.
Su FP/FN si distingue controevidenza nella direzione della GT dal sostegno alla risposta errata. Si confrontano FP con TN (GT safe), FN con TP (GT vulnerable). Nessun classificatore o soglia viene addestrato su questi risultati.
L'entropia A/B e condizionata alle due alternative; la massa A+B e riportata separatamente. Su code_end/record A/B sono letture diagnostiche; solo immediatamente prima del verdetto hanno il ruolo di alternative del prossimo token. Non sono confidenza calibrata di correttezza. 'uncertainty' e conservato come alias legacy di chosen_opposition = 1-q(verdetto finale); l'incertezza intrinseca simmetrica e 1-max(qA,qB). Gli errori ad alta preferenza restano possibili. chosen_opposition e le discordanze rispetto alla risposta usano il verdetto finale osservato: non sono feature disponibili dalla sola code_end prima della generazione; un eventuale predittore anticipato richiede un protocollo separato senza accesso al verdetto futuro.
L'entropia su tutto V non e ricostruibile esattamente dai top-k: vengono salvati soltanto limiti inferiore/superiore compatibili con massa residua e dimensione V. Non sono intervalli bootstrap. Le discordance nuove usano soltanto finestra 1.
I casi nei grafici individuali sono scelti per ID, fino al limite --error-examples per FP/FN; non per picchi o layer favorevoli. Le tabelle includono tutti gli errori e tutti i layer selezionati. Le risposte non valide sono 'unknown', escluse da TP/TN/FP/FN ma riportate nei conteggi e nell'accuratezza su tutti i GT noti.
I confronti base/extended non sono automaticamente appaiati solo perche i numeri di esempi coincidono. Usare --pairs con corrispondenze verificate; l'appaiamento tra posizioni usa invece lo stesso run/case_id. Un segnale CWE attesa non prova da solo che il codice sia vulnerabile: puo comparire anche nella patch. Un primo token fuori vocabolario negli strati finali puo riflettere il formato di output richiesto; non prova da solo assenza di segnale di sicurezza.
"""
    pics = sorted((args.out / "plots").glob("*_error_*.png"))
    page = '<meta charset="utf-8"><title>FP/FN e discordanze</title><style>body{font:14px system-ui;margin:28px}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>FP/FN e discordanze</h1>'
    page += "<pre>" + html.escape(notes) + "</pre><h2>Conteggi dei JSONL</h2>"
    page += _table(args.out, "outcome_counts").to_html(index=False, escape=True)
    page += "<h2>Qualita e metriche con intervalli</h2>" + _table(
        args.out, "run_quality_with_intervals"
    ).to_html(index=False, escape=True)
    page += "<h2>Indice di tutti gli errori e risposte non valide</h2>"
    cols = [
        "run",
        "case_id",
        "gt",
        "prediction",
        "outcome",
        "cwe",
        "anchor_available",
        "anchor_exact",
    ]
    page += cases[cases.outcome.isin(["FP", "FN", "unknown"])][cols].to_html(
        index=False, escape=True
    )
    page += "<h2>Evoluzioni per errore</h2>"
    page += "".join(
        f'<img loading="lazy" src="plots/{html.escape(p.name)}">' for p in pics
    )
    page += "<h2>Dettagli per tutti i layer</h2>"
    show = [
        c
        for c in [
            "run",
            "case_id",
            "outcome",
            "stream",
            "layer",
            "top1_token",
            "top1_probability_or_score",
            "top1_outside",
            "expected_cwe_rank",
            "expected_cwe_margin",
            "sg_polarity_difference",
            "exact_mass_polarity_contrast",
            "chosen_conditional",
            "pair_probability",
            "safety_best_token",
            "vuln_best_token",
            "expected_cwe_best_token",
            "outside_best_token",
        ]
        if c in details
    ]
    page += (
        details[show]
        .sort_values(["run", "case_id", "layer"])
        .to_html(index=False, escape=True)
    )
    (args.out / "error_review.html").write_text(page, encoding="utf-8")
    report = args.out / "report.html"
    section = '<h2>Incertezza e discordanze</h2><p><a href="error_review.html">Apri revisione FP/FN, conteggi e definizioni</a>. Tabelle discordance_*, fp_fn_*, error_* e token_distribution_entropy_bounds.</p>'
    section += "".join(
        f'<img loading="lazy" style="max-width:100%" src="plots/{html.escape(p.name)}">'
        for p in sorted((args.out / "plots").glob("*.png"))
        if "_discordance_" in p.name
        or p.name.endswith("_choice_uncertainty_confusion.png")
    )
    report.write_text(
        insert_report_section(report.read_text(), section), encoding="utf-8"
    )
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["error_analysis"] = {
        "signal_epsilon": args.signal_epsilon,
        "direction": "positive=vuln/A, negative=safe/B; ties neutral",
        "review_selection": "lexicographic case_id, not peaks",
        "plot_limit_per_error_type": args.error_examples,
        "all_error_cases_in_tables": True,
        "uncertainty_legacy_alias": "chosen_opposition",
        "windows": [1],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def compare_discordance_positions(args, out):
    frames = []
    keys = ["run", "case_id", "stream", "layer", "window", "profile", "signal"]
    for left, right in itertools.combinations(ANCHOR_FOLDERS, 2):
        a = _table(args.out / ANCHOR_FOLDERS[left], "discordance_case_layer")
        b = _table(args.out / ANCHOR_FOLDERS[right], "discordance_case_layer")
        cols = ["value", "sign", "opposes_final", "agrees_gt"]
        joined = b.merge(
            a[keys + cols], on=keys, suffixes=("", "_reference"), validate="one_to_one"
        )
        joined["comparison"] = right + "_minus_" + left
        valid = joined.sign.notna() & joined.sign_reference.notna()
        both_directional = valid & (joined.sign != 0) & (joined.sign_reference != 0)
        joined["opposite_directions"] = np.where(
            both_directional,
            (joined.sign != joined.sign_reference).astype(float),
            np.nan,
        )
        joined["any_neutral"] = np.where(
            valid,
            ((joined.sign == 0) | (joined.sign_reference == 0)).astype(float),
            np.nan,
        )
        joined["delta_opposes_final"] = (
            joined.opposes_final - joined.opposes_final_reference
        )
        joined["delta_agrees_gt"] = joined.agrees_gt - joined.agrees_gt_reference
        joined["delta_value"] = joined.value - joined.value_reference
        frames.append(joined)
    frame = pd.concat(frames, ignore_index=True)
    save(frame, "discordance_positions_case_layer", out, "csv")
    rng = np.random.default_rng(args.seed)
    base = ["run", "comparison", "stream", "layer", "window", "profile", "signal"]
    metrics = [
        "opposite_directions",
        "any_neutral",
        "delta_opposes_final",
        "delta_agrees_gt",
        "delta_value",
    ]
    for mode, extra in [("overview", []), ("gt", ["gt"]), ("confusion", ["outcome"])]:
        save(
            summarize(frame, base + extra, metrics, args.bootstrap, rng),
            "discordance_positions_" + mode,
            out,
            "csv",
        )
    if args.no_plots:
        return
    import matplotlib.pyplot as plt

    stats = _table(out, "discordance_positions_confusion")
    cases = _table(args.out / ANCHOR_FOLDERS["code_end"], "cases")
    args.case_counts = cases
    for (run, comparison), part in stats[
        (stats.stream == "lens") & (stats.metric == "delta_agrees_gt")
    ].groupby(["run", "comparison"]):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), sharey=True)
        observed = frame[(frame.run == run) & (frame.comparison == comparison)]
        for ax, outcome in zip(axes, OUTCOMES, strict=True):
            for signal, points in part[part.outcome == outcome].groupby("signal"):
                points = points.sort_values("layer")
                ax.plot(points.layer, points["mean"], label=SIGNAL_TITLES[signal])
                ax.fill_between(
                    points.layer.to_numpy(float),
                    points.ci_low.to_numpy(float),
                    points.ci_high.to_numpy(float),
                    alpha=0.1,
                )
            ax.set(
                title=category_label(args, run, "outcome", outcome, observed),
                xlabel="Layer",
                ylim=(-1.03, 1.03),
            )
            ax.axhline(0, color="gray", lw=0.7)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Delta quota nella direzione della GT")
        legend_from_axes(axes, fontsize=6)
        fig.suptitle(
            run + " / " + comparison + " — stesso esempio, non campioni indipendenti"
        )
        fig.tight_layout()
        (out / "plots").mkdir(exist_ok=True)
        fig.savefig(
            out / "plots" / (run + "_" + comparison + "_discordance.png"), dpi=145
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Metriche lens e confronti in un solo file",
        epilog=HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compare", default="all")
    parser.add_argument("--profiles", default="full", choices=["full"])
    parser.add_argument(
        "--require-sg",
        action="store_true",
        help="Reject JSONL files without frozen full-vocabulary SG scores",
    )
    parser.add_argument("--windows", default="1")
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--anchor",
        default="all",
        choices=["all", "code_end", "mcp_record", "mcp_pre_verdict"],
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--signal-epsilon",
        type=float,
        default=1e-12,
        help="Numerical zero tolerance only; never fitted to labels",
    )
    parser.add_argument(
        "--error-examples",
        type=int,
        default=6,
        help="Maximum per-case plots per run and error type; all cases remain in tables",
    )
    parser.add_argument("--seed", type=int, default=42017)
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="csv")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--dataset-map", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()
    args.compare = (
        MODES.copy() if args.compare == "all" else set(args.compare.split(","))
    )
    if not args.compare or args.compare - MODES:
        parser.error("Modalita --compare non valida")
    args.profiles = list(dict.fromkeys(args.profiles.split(",")))
    if set(args.profiles) - {"full", "no_fragments", "specific"}:
        parser.error("Profilo non valido")
    if "full" in args.profiles:
        args.profiles = ["full"] + [p for p in args.profiles if p != "full"]
    if not np.isfinite(args.signal_epsilon) or args.signal_epsilon < 0:
        parser.error("--signal-epsilon must be finite and >=0")
    if args.error_examples < 0:
        parser.error("--error-examples must be >=0")
    if args.bootstrap < 0:
        parser.error("--bootstrap deve essere >=0")
    paths = []
    for path in args.inputs:
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")) + sorted(path.glob("*.jsonl.gz")))
        else:
            paths.append(path)
    if not paths:
        parser.error("Nessun JSONL trovato")
    args.inputs = paths
    for name in ["windows", "layers"]:
        value = getattr(args, name)
        if value != "all":
            try:
                vals = list(map(int, value.split(",")))
                assert all(n >= (1 if name == "windows" else 0) for n in vals)
            except (ValueError, AssertionError):
                parser.error(f"--{name}: usa all o numeri separati da virgola")
    if args.format != "csv" and (args.anchor == "all" or "errors" in args.compare):
        parser.error("La suite e le nuove analisi errori richiedono --format csv")
    try:
        if args.anchor == "all":
            analyze_suite(args)
        else:
            analyze(args)
            if args.format == "csv":
                choices = _choice_rows(args.inputs, anchors=[args.anchor])
                if (
                    args.format == "csv"
                    and (args.out / "tables" / "sg_group_metrics.csv.gz").exists()
                ):
                    _, _, choices = _add_anchor_analysis(args, choices)
                else:
                    cases = _table(args.out, "cases")
                    choices = (
                        choices.merge(
                            cases, on=["run", "case_id"], validate="many_to_one"
                        )
                        if len(choices)
                        else choices
                    )
                if "errors" in args.compare:
                    add_error_analysis(args, choices)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Errore: {exc}\n")


_MARKER = "__anchor_topk_slice_v1__"

_CODES = {"uint16": "H", "uint32": "I", "float32": "f", "float64": "d"}

_WIDTHS = {"uint16": 2, "uint32": 4, "float32": 4, "float64": 8}


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


if __name__ == "__main__":
    main()
