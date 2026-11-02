from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import pipeline_controlled as pipeline
import torch
import transformers
from dataset_and_scoring import load_cases, predict_label, score_terms

import jlens

OPS = ("ablate", "reduce", "amplify", "inject", "swap")


def blocks(hf):
    for path in (
        "model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.model.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
    ):
        obj = hf
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("Blocchi del modello non trovati.")


def parse_layers(value):
    layers = set()
    for item in value.split(","):
        ends = item.strip().split("-")
        if len(ends) == 1:
            layers.add(int(ends[0]))
        elif len(ends) == 2:
            lo, hi = map(int, ends)
            if lo > hi:
                raise ValueError("Intervallo layer invertito.")
            layers.update(range(lo, hi + 1))
        else:
            raise ValueError("Layer: usa 40-58 oppure 40,45,50.")
    if not layers or min(layers) < 0:
        raise ValueError("Layer non validi.")
    return sorted(layers)


def term_strength(value, default):
    term, sep, amount = value.rpartition(":")
    if not sep:
        term, amount = value, default
    else:
        amount = float(amount)
    if not term.strip() or not math.isfinite(float(amount)):
        raise ValueError("Termine vuoto o forza non finita.")
    return term.strip(), float(amount)


def parse_terms(items, default_op):
    parsed = []
    for item in items:
        op, sep, value = item.partition("=")
        if not sep:
            op, value = default_op, item
        if op not in OPS:
            raise ValueError(f"Operazione non valida: {op}")
        default = 0.5 if op == "reduce" else 1.0
        if op == "swap":
            first, sep, second = value.partition("::")
            if not sep or not first.strip():
                raise ValueError("Swap richiede swap=source::target[:forza].")
            target, strength = term_strength(second, default)
            parsed.append((op, (first.strip(), target, strength)))
        else:
            term, strength = term_strength(value, default)
            if op == "ablate" and strength != 1.0:
                raise ValueError(
                    "Ablate usa forza 1; per una rimozione parziale usa reduce."
                )
            if op == "reduce" and not 0 <= strength <= 1:
                raise ValueError("Reduce richiede forza tra 0 e 1.")
            parsed.append((op, (term, strength)))
    return parsed


def direction(lens, wrapped, layer, token_id):

    jacobian = lens.jacobians[layer].detach().float().cpu()
    head = wrapped._lm_head.weight[token_id].detach().float().cpu()
    norm = wrapped._final_norm
    weight = getattr(norm, "weight", None)
    gain = torch.ones_like(head) if weight is None else weight.detach().float().cpu()
    if "gemma" in type(norm).__name__.lower():
        gain = gain + 1
    vector = jacobian.T @ (gain * head)
    length = vector.norm()
    if (
        not torch.isfinite(vector).all()
        or not torch.isfinite(length)
        or length <= 1e-12
    ):
        raise ValueError(f"Direzione nulla o non finita a L{layer}, token {token_id}.")
    return vector / length


def make_hook(items):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        for op, payload in items:
            if op == "swap":
                source, target, strength = payload
                source, target = (
                    source.to(h.device, h.dtype),
                    target.to(h.device, h.dtype),
                )
                projection = (h @ source).unsqueeze(-1)
                h = h - projection * source + strength * projection * target
            else:
                vector, strength = payload
                vector = vector.to(h.device, h.dtype)
                if op == "inject":
                    scale = h.norm(dim=-1, keepdim=True) / math.sqrt(h.shape[-1])
                    h = h + strength * scale * vector
                else:
                    projection = (h @ vector).unsqueeze(-1)
                    sign = -1 if op in ("ablate", "reduce") else 1
                    h = h + sign * strength * projection * vector
        return (h,) + output[1:] if isinstance(output, tuple) else h

    return hook


@contextmanager
def active_hooks(model_blocks, per_layer):
    handles = []
    try:
        for layer, items in per_layer.items():
            handles.append(model_blocks[layer].register_forward_hook(make_hook(items)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def run_case(hf, tok, spec, case, args):

    prompt = pipeline.PROMPTS[args.prompt]
    if args.condition == "carrier":
        prefix, anchors = pipeline.build_carrier_prompt(
            case.code,
            args.chat_template,
            tok,
            spec,
            args.carrier,
            pipeline.CARRIERS[args.carrier]["text"],
            pipeline.CARRIER_TASKS[args.carrier_task],
        )
        reasoning, reasoning_cut, output_prompt = "", False, prefix
    else:
        prefix, parts = pipeline.build_prompts(
            prompt, case.code, args.chat_template, args.condition == "silent", tok, spec
        )
        anchors = {"assistant_start": len(prefix)}
        reasoning, reasoning_cut, output_prompt = "", False, prefix
        if args.condition == "cot":
            reasoning, reasoning_cut = pipeline.generate_batch(
                hf, tok, [prefix], args.max_reasoning
            )[0]
            output_prompt = pipeline.verdict_prompt(
                prompt, prefix, parts, reasoning, args.chat_template, tok, spec
            )
            anchors = {"assistant_start": len(output_prompt)}
    max_output = 96 if args.condition == "carrier" else 8
    if len(tok.encode(output_prompt)) + max_output > spec["max_seq_len"]:
        raise ValueError(
            f"{case.case_id}: prompt troppo lungo; nessun troncamento implicito."
        )
    response, response_cut = pipeline.generate_batch(
        hf, tok, [output_prompt], max_output
    )[0]
    bad, good, prob, summary, valid = {}, {}, None, None, None
    if args.condition == "carrier":
        pred, valid = pipeline.parse_mcp_verdict(response)
    else:
        scores = score_terms(hf, tok, output_prompt, prompt["bad"] + prompt["good"])
        bad = {t: scores[t] for t in prompt["bad"] if t in scores}
        good = {t: scores[t] for t in prompt["good"] if t in scores}
        prob = predict_label(bad, good)
        summary = pipeline.last_summary(reasoning)
        pred = (
            pipeline.parse_pred(response, prompt)
            or prob
            or pipeline.parse_pred(summary, prompt)
        )
    return {
        "reasoning": reasoning,
        "summary": summary,
        "response": response,
        "pred": pred,
        "pred_prob": prob,
        "scores_bad": bad,
        "scores_good": good,
        "carrier_ok": valid,
        "truncated": bool(reasoning_cut or response_cut),
        "correct": pred == case.label if pred is not None else None,
        "verdict_prompt": output_prompt,
        "full_prompt": output_prompt + response,
        "focus_anchors": anchors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Interventi J-Lens con baseline appaiata; usa pipeline_controlled.py."
    )
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--prompt", default="security")
    parser.add_argument(
        "--condition", choices=("carrier", "silent", "cot"), default="carrier"
    )
    parser.add_argument("--carrier", choices=("mcp-verdict",), default="mcp-verdict")
    parser.add_argument("--carrier-task", default="neutral")
    parser.add_argument(
        "--no-chat-template", dest="chat_template", action="store_false", default=True
    )
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--dataset-kind", choices=("auto", "controlled", "secllmholmes"), default="auto"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lens-file")
    parser.add_argument("--lens-repo")
    parser.add_argument("--max-reasoning", type=int)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh")
    )
    parser.add_argument("--op", choices=OPS, default="ablate")
    parser.add_argument("--terms", nargs="*", default=[])
    parser.add_argument("--terms-file", type=Path)
    parser.add_argument("--layers")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.silent:
        args.condition = "silent"
    if args.limit < 0:
        parser.error("--limit deve essere >=0; 0 significa tutti.")
    raw = list(args.terms)
    if args.terms_file:
        raw += [
            line.strip()
            for line in args.terms_file.read_text().splitlines()
            if line.strip()
        ]
    if not raw:
        parser.error("Serve --terms o --terms-file.")
    try:
        parsed = parse_terms(raw, args.op)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out.exists():
        parser.error(f"Output già presente: {args.out}")

    if (
        args.model not in pipeline.MODELS
        or args.prompt not in pipeline.PROMPTS
        or args.carrier_task not in pipeline.CARRIER_TASKS
    ):
        parser.error(
            "Modello, prompt o carrier-task non presente in pipeline_controlled.py."
        )
    spec = dict(
        pipeline.MODELS[args.model],
        model_name=args.model,
        effort_explicit_none=args.reasoning_effort == "none",
    )
    effort = args.reasoning_effort
    spec["reasoning_effort"] = (
        None
        if effort == "none" or effort is None and args.condition == "carrier"
        else effort or spec.get("reasoning_effort")
    )
    if effort not in (None, "none") and effort not in pipeline.SUPPORTED_REASONING.get(
        args.model, set()
    ):
        parser.error("Reasoning effort non supportato dal modello.")
    if not args.chat_template and effort is not None:
        parser.error("--reasoning-effort richiede il chat template.")
    args.max_reasoning = (
        args.max_reasoning if args.max_reasoning is not None else spec["max_reasoning"]
    )
    if args.max_reasoning < 1:
        parser.error("--max-reasoning deve essere positivo.")
    cases = load_cases(str(Path(args.dataset).expanduser()), kind=args.dataset_kind)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        parser.error("Dataset senza esempi.")
    layers = parse_layers(args.layers or f"{spec['band'][0]}-{spec['band'][1]}")
    hf = pipeline.load_model(spec)
    tok = transformers.AutoTokenizer.from_pretrained(spec["hf_id"])
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError("Tokenizer senza pad/eos.")
        tok.pad_token = tok.eos_token
    lens_repo = args.lens_repo or spec.get("lens_repo", pipeline.LENS_REPO)
    lens_file = args.lens_file or spec["lens_file"]
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    model_blocks = blocks(hf)
    if any(
        layer not in lens.source_layers or layer >= len(model_blocks)
        for layer in layers
    ):
        raise ValueError(
            "Uno o più layer richiesti non sono disponibili; nessuna selezione parziale implicita."
        )
    wanted = sorted(
        {
            term
            for op, item in parsed
            for term in (item[:2] if op == "swap" else item[:1])
        }
    )
    ids = pipeline.single_ids(tok, wanted)
    if set(ids) != set(wanted):
        raise ValueError(
            f"Non sono single-token con spazio iniziale: {sorted(set(wanted) - set(ids))}"
        )
    wrapped = jlens.from_hf(hf, tok)
    per_layer = {}
    for layer in layers:
        vectors = {term: direction(lens, wrapped, layer, ids[term]) for term in wanted}
        per_layer[layer] = [
            (
                op,
                (vectors[item[0]], vectors[item[1]], item[2])
                if op == "swap"
                else (vectors[item[0]], item[1]),
            )
            for op, item in parsed
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "intervention",
        "model": args.model,
        "hf_id": spec["hf_id"],
        "quantize": spec["quantize"],
        "prompt": args.prompt,
        "condition": args.condition,
        "chat_template": args.chat_template,
        "carrier": args.carrier if args.condition == "carrier" else None,
        "carrier_text": pipeline.CARRIERS[args.carrier]["text"]
        if args.condition == "carrier"
        else None,
        "carrier_task": args.carrier_task,
        "reasoning_effort_requested": args.reasoning_effort,
        "reasoning_effort_effective": spec.get("reasoning_effort"),
        "max_reasoning": args.max_reasoning,
        "max_seq_len": spec["max_seq_len"],
        "dataset": args.dataset,
        "dataset_kind": args.dataset_kind,
        "lens_repo": lens_repo,
        "lens_file": lens_file,
        "band": layers,
        "layers": layers,
        "default_op": args.op,
        "terms": raw,
        "token_ids": ids,
        "intervention_scope": "all_positions_prefill_and_decode",
        "direction_formula": "unit(J.T @ (final_norm_gain * unembedding_row))",
        "baseline_paired": True,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with args.out.open("x") as stream:
        stream.write(json.dumps({"_meta": metadata}, ensure_ascii=False) + "\n")
        for index, case in enumerate(cases, 1):
            baseline = run_case(hf, tok, spec, case, args)
            with active_hooks(model_blocks, per_layer):
                result = run_case(hf, tok, spec, case, args)
            record = {
                "case_id": case.case_id,
                "cwe": case.cwe,
                "pair_id": case.pair_id,
                "ground_truth": case.label,
                "code": case.code,
                "source_code_sha256": hashlib.sha256(case.code.encode()).hexdigest(),
                "baseline": baseline,
                "prediction_changed": baseline["pred"] != result["pred"],
                **result,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"{index}/{len(cases)} baseline={baseline['pred']} intervention={result['pred']} GT={case.label}"
            )
    print(f"Scritto: {args.out}")


if __name__ == "__main__":
    main()
