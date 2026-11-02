from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def variants(term):
    forms = {
        term,
        term.lower(),
        term.upper(),
        term.title(),
        term[:1].upper() + term[1:],
    }
    return sorted({prefix + form for prefix in ("", " ") for form in forms})


def observed_tokens(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    pairs = {}
    model = None

    def add(token_id, token):
        if token_id in pairs and pairs[token_id] != token:
            raise ValueError(f"Conflicting decoded strings for token {token_id}")
        pairs[token_id] = token

    def visit(value):
        if isinstance(value, list):
            for v in value:
                visit(v)
        elif isinstance(value, dict):
            if isinstance(value.get("ids"), list) and isinstance(
                value.get("tokens"), list
            ):
                if len(value["ids"]) != len(value["tokens"]):
                    raise ValueError("Token array length mismatch")
                for i, t in zip(value["ids"], value["tokens"]):
                    add(i, t)
            for k, v in value.items():
                if k not in {"ids", "tokens", "logp"}:
                    visit(v)

    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "_meta" in rec:
                found = rec["_meta"].get("hf_id")
                if model is not None and found != model:
                    raise ValueError("Mixed model metadata")
                model = found
                continue
            for id_key, text_key in (("topk_vocab_ids", "topk_vocab"),):
                ids, texts = rec.get(id_key, []), rec.get(text_key, [])
                if len(ids) != len(texts):
                    raise ValueError("Vocabulary length mismatch")
                for i, t in zip(ids, texts):
                    add(i, t)
            for k in ("anchor_readout_packed", "prompt_tail_readout_packed"):
                if k in rec:
                    for i, t in zip(rec[k]["vocab_ids"], rec[k]["vocab_tokens"]):
                        add(i, t)
            for k in ("anchor_readout", "prompt_tail_readout"):
                visit(rec.get(k))
    if not model:
        raise ValueError("Missing hf_id: cannot bind observed IDs to a tokenizer")
    return pairs, model


def compile_catalog(
    catalog_path, output, tokenizer_path=None, jsonl_path=None, model_id=None
):
    raw = catalog_path.read_bytes()
    catalog = json.loads(raw)
    concepts = catalog["concepts"]
    forms = defaultdict(set)
    by_term = {c["term"]: c for c in concepts}
    for c in concepts:
        for form in variants(c["term"]):
            forms[form].add(c["term"])
    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        pairs = {}
        membership = defaultdict(set)
        for form, terms in forms.items():
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                token_id = ids[0]
                pairs[token_id] = tokenizer.decode([token_id])
                membership[token_id].update(terms)
        coverage = "tokenizer_compiled"
        bound_model = model_id or tokenizer_path
    else:
        pairs, bound_model = observed_tokens(jsonl_path)
        membership = {i: forms[text] for i, text in pairs.items() if text in forms}
        coverage = "observed_only"
    tokens = {}
    found_terms = set()
    group_ids = {
        "generic": [],
        "safety": [],
        "vuln": [],
        **{c: [] for c in catalog["meta"]["cwes"]},
    }
    for token_id, terms in sorted(membership.items()):
        if not terms:
            continue
        found_terms.update(terms)
        groups = sorted({g for term in terms for g in by_term[term]["groups"]})
        tokens[str(token_id)] = {
            "text": pairs[token_id],
            "groups": groups,
            "terms": sorted(terms),
            "context_sensitive": any(by_term[t]["context_sensitive"] for t in terms),
            "typo": any(by_term[t]["variant_type"] == "typo" for t in terms),
        }
        for g in groups:
            group_ids[g].append(token_id)
    meta = {
        **catalog["meta"],
        "catalog_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizer_id": bound_model,
        "coverage": coverage,
        "surface_variants": "original/lower/upper/title/capitalized; zero or one leading ASCII space",
        "observed_vocabulary_size": len(pairs) if jsonl_path else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "groups").mkdir(exist_ok=True)
    for group in group_ids:
        selected = [c["term"] for c in concepts if group in c["groups"]]
        (output / "groups" / (group + ".txt")).write_text(
            "".join(term + "\n" for term in selected)
        )
    (output / "concepts.json").write_bytes(raw)
    with (output / "concepts.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(
            [
                "term",
                "groups",
                "role",
                "context_sensitive",
                "variant_type",
                "canonical_spelling",
                "source",
            ]
        )
        for c in concepts:
            writer.writerow(
                [
                    c["term"],
                    "|".join(c["groups"]),
                    c["role"],
                    c["context_sensitive"],
                    c["variant_type"],
                    c["canonical_spelling"],
                    c["source"],
                ]
            )
    (output / "token_groups.json").write_text(
        json.dumps(
            {"meta": meta, "groups": group_ids, "tokens": tokens},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    with (output / "token_membership.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(
            ["token_id", "surface", "groups", "terms", "context_sensitive", "typo"]
        )
        for i, row in tokens.items():
            writer.writerow(
                [
                    i,
                    row["text"],
                    "|".join(row["groups"]),
                    "|".join(row["terms"]),
                    row["context_sensitive"],
                    row["typo"],
                ]
            )
    with (output / "concept_coverage.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["term", "status"])
        for c in concepts:
            writer.writerow(
                [
                    c["term"],
                    (
                        "available"
                        if c["term"] in found_terms
                        else (
                            "not_observed"
                            if coverage == "observed_only"
                            else "no_single_token_variant"
                        )
                    ),
                ]
            )
    result = {
        "coverage": coverage,
        "tokenizer_id": bound_model,
        "matching_token_ids": len(tokens),
        "concepts_with_ids": len(found_terms),
        "concepts_without_ids": len(concepts) - len(found_terms),
        "group_token_counts": {g: len(v) for g, v in group_ids.items()},
    }
    (output / "compile_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", type=Path, default=Path(__file__).parent / "vocab/concepts.json"
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "vocab")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tokenizer")
    source.add_argument("--from-jsonl", type=Path)
    parser.add_argument("--model-id")
    args = parser.parse_args()
    print(
        json.dumps(
            compile_catalog(
                args.catalog, args.out, args.tokenizer, args.from_jsonl, args.model_id
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
