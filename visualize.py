from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from contextlib import nullcontext
from pathlib import Path

import intervention
import pipeline_controlled as pipeline
import transformers
from dataset_and_scoring import load_cases

import jlens
from jlens.vis import build_page, compute_slice


def load_run(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    meta, records = None, []
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "_meta" in row:
                if meta is not None or records:
                    raise ValueError(
                        f"{path}:{number}: metadata duplicati o fuori posizione."
                    )
                meta = row["_meta"]
            else:
                records.append(row)
    if meta is None:
        raise ValueError(f"{path}: metadata _meta mancanti.")
    return meta, records


def find_case(records, case_id):
    matches = [row for row in records if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(
            f"Atteso un solo record per {case_id!r}; trovati {len(matches)}."
        )
    return matches[0]


def get_code(record, meta):

    code = record.get("code")
    if code is None:
        if not meta.get("dataset"):
            raise ValueError("Codice e percorso dataset mancanti.")
        cases = load_cases(
            str(Path(meta["dataset"]).expanduser()),
            kind=meta.get("dataset_kind", "auto"),
        )
        found = [case for case in cases if case.case_id == record["case_id"]]
        if len(found) != 1:
            raise ValueError("Esempio non trovato univocamente nel dataset.")
        code = found[0].code
    digest = record.get("source_code_sha256")
    if digest and hashlib.sha256(code.encode()).hexdigest() != digest:
        raise ValueError(
            "Il codice non coincide con quello della run; ricostruzione rifiutata."
        )
    return code


def reconstruct_full_prompt(record, meta, tok, spec):
    if record.get("full_prompt"):
        return record["full_prompt"], "full_prompt salvato"
    if record.get("verdict_prompt"):
        return record["verdict_prompt"] + (
            record.get("response") or ""
        ), "verdict_prompt salvato + response"
    if "condition" not in meta:
        raise ValueError(
            "Run storica senza prompt esatto e condition: non ricostruisco il template del file eliminato."
        )

    code = get_code(record, meta)
    condition = meta["condition"]
    chat = meta.get("chat_template", True)
    if condition == "carrier":
        if meta.get("carrier_prompt_version") != "v5-controlled-target-scores":
            raise ValueError(
                "Versione carrier non riconosciuta: serve il full_prompt originale."
            )
        carrier = meta["carrier"]
        text = meta.get("carrier_text")
        if text is None:
            raise ValueError("carrier_text originale mancante.")
        task = meta.get("carrier_task_text")
        if task is None:
            raise ValueError("carrier_task_text originale mancante.")
        prompt, _ = pipeline.build_carrier_prompt(
            code, chat, tok, spec, carrier, text, task
        )
    elif condition in ("silent", "cot"):
        template = pipeline.PROMPTS[meta["prompt"]]
        prefix, parts = pipeline.build_prompts(
            template, code, chat, condition == "silent", tok, spec
        )
        prompt = (
            prefix
            if condition == "silent"
            else pipeline.verdict_prompt(
                template, prefix, parts, record.get("reasoning") or "", chat, tok, spec
            )
        )
    else:
        raise ValueError(f"Condition non riconosciuta: {condition}")
    return (
        prompt + (record.get("response") or ""),
        "ricostruito con pipeline_controlled attuale e metadata originali; prompt esatto non salvato",
    )


def safe_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value)).strip(".")
    if not cleaned:
        cleaned = "case"
    if cleaned != str(value):
        cleaned += "_" + hashlib.sha256(str(value).encode()).hexdigest()[:8]
    return cleaned


def output_paths(run_path, case_id, out_dir, mode):
    base = Path(out_dir) if out_dir else run_path.parent / "vis" / run_path.stem
    name = safe_name(case_id)
    if mode == "fetch":
        directory = base / name
        return directory, directory / "index.html"
    return None, base / (name + ".html")


def intervention_context(meta, model, lens, wrapped, tok):

    if meta.get("intervention_scope") != "all_positions_prefill_and_decode":
        raise ValueError(
            "Run intervento legacy senza scope verificabile; usa --variant baseline se presente."
        )
    parsed = intervention.parse_terms(meta["terms"], meta.get("default_op", "ablate"))
    layers = meta["layers"]
    model_blocks = intervention.blocks(model)
    if any(
        layer not in lens.source_layers or layer >= len(model_blocks)
        for layer in layers
    ):
        raise ValueError("Layer intervento non disponibili.")
    wanted = {
        term for op, item in parsed for term in (item[:2] if op == "swap" else item[:1])
    }
    ids = pipeline.single_ids(tok, wanted)
    if set(ids) != wanted or meta.get("token_ids") != ids:
        raise ValueError("Gli ID dei termini non coincidono con la run originale.")
    per_layer = {}
    for layer in layers:
        vectors = {
            term: intervention.direction(lens, wrapped, layer, ids[term])
            for term in wanted
        }
        per_layer[layer] = [
            (
                op,
                (vectors[item[0]], vectors[item[1]], item[2])
                if op == "swap"
                else (vectors[item[0]], item[1]),
            )
            for op, item in parsed
        ]
    return intervention.active_hooks(model_blocks, per_layer)


def main():
    parser = argparse.ArgumentParser(
        description="HTML J-Lens da run controlled o intervention; non richiede lens_pipeline.py."
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--variant", choices=("baseline", "intervention"), default="intervention"
    )
    parser.add_argument("--out-dir")
    parser.add_argument("--mode", choices=("fetch", "embed"), default="fetch")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--last-n", type=int)
    parser.add_argument("--max-tracked", type=int)
    parser.add_argument("--no-mask-display", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--lens-file")
    parser.add_argument("--lens-repo")
    args = parser.parse_args()
    meta, records = load_run(args.run)
    if args.list_cases:
        for record in records:
            print(record.get("case_id"))
        return
    if not args.case_id:
        parser.error("Serve --case oppure --list-cases.")
    if (
        args.top_n < 1
        or args.last_n is not None
        and args.last_n < 1
        or args.max_tracked is not None
        and args.max_tracked < 1
    ):
        parser.error("top-n, last-n e max-tracked devono essere positivi.")
    record = find_case(records, args.case_id)
    is_intervention = meta.get("kind") == "intervention"
    if args.variant == "baseline":
        if not is_intervention or not isinstance(record.get("baseline"), dict):
            parser.error("Baseline appaiata non presente nella run.")
        record = {**record, **record["baseline"]}

    model_name = args.model or meta.get("model")
    if model_name not in pipeline.MODELS:
        raise ValueError(
            "Modello non presente in pipeline_controlled.py; specifica --model."
        )
    spec = dict(pipeline.MODELS[model_name], model_name=model_name)
    if meta.get("hf_id") and meta["hf_id"] != spec["hf_id"]:
        raise ValueError("HF model della run diverso da quello selezionato.")
    spec["quantize"] = meta.get("quantize", spec["quantize"])
    spec["effort_explicit_none"] = meta.get("reasoning_effort_requested") == "none"
    if "reasoning_effort_effective" in meta:
        spec["reasoning_effort"] = meta["reasoning_effort_effective"]
    lens_file = args.lens_file or meta.get("lens_file")
    if not lens_file:
        raise ValueError("Lens mancante nei metadata: specifica --lens-file.")
    lens_repo = (
        args.lens_repo
        or meta.get("lens_repo")
        or spec.get("lens_repo", pipeline.LENS_REPO)
    )
    if (
        is_intervention
        and args.variant == "intervention"
        and (lens_file != meta.get("lens_file") or lens_repo != meta.get("lens_repo"))
    ):
        raise ValueError(
            "Replay intervento richiede la stessa lens della run originale."
        )
    tok = transformers.AutoTokenizer.from_pretrained(spec["hf_id"])
    full_prompt, prompt_source = reconstruct_full_prompt(record, meta, tok, spec)
    max_seq_len = int(meta.get("max_seq_len") or spec["max_seq_len"])
    n_tokens = len(tok.encode(full_prompt))
    if n_tokens > max_seq_len:
        raise ValueError(
            f"{n_tokens} token oltre max_seq_len={max_seq_len}; nessun troncamento implicito."
        )
    hf = pipeline.load_model(spec)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    context = (
        intervention_context(meta, hf, lens, model, tok)
        if is_intervention and args.variant == "intervention"
        else nullcontext()
    )
    with context:
        data = compute_slice(
            model,
            lens,
            full_prompt,
            top_n=args.top_n,
            max_tracked=args.max_tracked,
            layer_stride=1,
            last_n_tokens=args.last_n,
            max_seq_len=max_seq_len,
            mask_display=not args.no_mask_display,
        )
    label = args.case_id + ("_" + args.variant if is_intervention else "")
    directory, html_path = output_paths(args.run, label, args.out_dir, args.mode)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    description = f"{model_name} | condition={meta.get('condition')} | GT={record.get('ground_truth')} | pred={record.get('pred')} | variant={args.variant if is_intervention else 'original'} | {prompt_source}"
    page, raw_bytes, payload_bytes = build_page(
        data,
        full_prompt,
        title=f"{args.case_id} — {model_name}",
        description=description,
        mode=args.mode,
        out_dir=directory,
    )
    html_path.write_text(page, encoding="utf-8")
    print(
        f"Prompt: {prompt_source}\nTokens: {n_tokens}\nHTML: {html_path}\nRaw: {raw_bytes} bytes\nPayload: {payload_bytes} bytes"
    )
    if args.mode == "fetch":
        print(
            f"uv run python3 -m http.server 8766 --bind 127.0.0.1 --directory {html_path.parent}"
        )
        print("Apri http://localhost:8766/ dopo il forward SSH.")


if __name__ == "__main__":
    main()
