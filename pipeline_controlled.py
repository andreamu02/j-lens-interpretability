from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import sys
import time
import zlib
from array import array
from pathlib import Path

import torch
import transformers
from anchor_codec import pack_record
from dataset_and_scoring import load_cases, predict_label, score_terms

import jlens

DATASET = "~/jspace/data/secllmholmes/datasets/hand-crafted/dataset"
LENS_REPO = "neuronpedia/jacobian-lens"
L = "jlens/Salesforce-wikitext"
MODELS = {
    "olmo3-32b": {
        "hf_id": "allenai/Olmo-3-1125-32B",
        "quantize": "4bit",
        "band": (25, 58),
        "lens_file": f"olmo-3-1125-32b/{L}/Olmo-3-1125-32B_jacobian_lens.pt",
        "jpp_file": "olmo-3-1125-32b/jpp/Salesforce-wikitext/Olmo-3-1125-32B_jpp_lens.pt",
        "max_reasoning": 256,
        "max_seq_len": 4096,
        "layer_chunk": 1,
        "batch_size": 1,
        "base_model": True,
        "no_explicit_reasoning": True,
        "min_transformers": "5.17.0",
        "stop_at_mcp_default": True,
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "quantize": "none",
        "band": (17, 23),
        "lens_file": f"qwen2.5-7b-it/{L}/Qwen2.5-7B-Instruct_jacobian_lens.pt",
        "max_reasoning": 1500,
        "max_seq_len": 4096,
        "layer_chunk": 2,
        "batch_size": 8,
    },
    "gemma3-12b": {
        "hf_id": "google/gemma-3-12b-it",
        "quantize": "4bit",
        "band": (18, 39),
        "lens_file": "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt",
        "max_reasoning": 2000,
        "max_seq_len": 4096,
        "layer_chunk": 1,
        "batch_size": 1,
    },
    "gemma3-27b": {
        "hf_id": "google/gemma-3-27b-it",
        "quantize": "4bit",
        "band": (28, 52),
        "lens_file": f"gemma-3-27b-it/{L}/gemma-3-27b-it_jacobian_lens.pt",
        "max_reasoning": 2000,
        "max_seq_len": 4096,
        "layer_chunk": 1,
        "batch_size": 1,
    },
    "gemma4-31b": {
        "hf_id": "google/gemma-4-31B",
        "quantize": "4bit",
        "band": (26, 53),
        "lens_file": f"gemma-4-31b/{L}/gemma-4-31B_jacobian_lens.pt",
        "jpp_file": "gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt",
        "max_reasoning": 2000,
        "max_seq_len": 4096,
        "layer_chunk": 1,
        "batch_size": 1,
        "image_text": True,
        "base_model": True,
    },
    "qwen3.6-27b": {
        "hf_id": "Qwen/Qwen3.6-27B",
        "quantize": "4bit",
        "band": (36, 58),
        "lens_file": f"qwen3.6-27b/{L}/Qwen3.6-27B_jacobian_lens_n1000.pt",
        "jpp_file": "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt",
        "max_reasoning": 3900,
        "max_seq_len": 4096,
        "layer_chunk": 1,
        "batch_size": 1,
        "image_text": True,
    },
    "gpt-oss-20b": {
        "hf_id": "openai/gpt-oss-20b",
        "quantize": "mxfp4",
        "band": (11, 20),
        "lens_file": f"gpt-oss-20b/{L}/gpt-oss-20b_jacobian_lens.pt",
        "max_reasoning": 3900,
        "max_seq_len": 8192,
        "layer_chunk": 1,
        "batch_size": 1,
        "reasoning_effort": "medium",
    },
    "qwen3.8-27b": {
        "hf_id": "Qwen/Qwen3.8-27B",
        "quantize": "4bit",
        "band": (36, 58),
        "lens_repo": "eyes-ml/Qwen3.8-27B_jacobian-lens",
        "lens_file": "Qwen3.8-27B_jacobian_lens.pt",
        "max_reasoning": 3900,
        "max_seq_len": 8192,
        "layer_chunk": 1,
        "batch_size": 1,
        "image_text": True,
        "reasoning_effort": "medium",
    },
}
PROMPTS = {
    "security": {
        "role": "a security expert who analyzes source code for security weaknesses",
        "task": "Analyze the function above step by step, following your process. Focus only on security-relevant vulnerabilities and determine whether the function is vulnerable or safe. Do not invent behavior that is not supported by the code.",
        "silent_task": "Analyze the function above carefully. Focus only on security-relevant vulnerabilities and determine whether the function is vulnerable or safe. Do not invent behavior that is not supported by the code.",
        "tail": "\n\nBased on this analysis, the expert's final answer, one word, safe or vulnerable, is",
        "bad": [
            "vulnerable",
            "insecure",
            "unsafe",
            "susceptible",
            "prone",
            "flawed",
            "risky",
        ],
        "good": ["safe", "secure", "sound", "correct", "fine", "acceptable", "robust"],
    },
    "usable": {
        "role": "a code expert who analyzes source code for problems",
        "task": "Analyze the function above step by step, following your process. Focus only on relevant problems and determine whether the function is usable or unusable. Do not invent behavior that is not supported by the code.",
        "silent_task": "Analyze the function above carefully. Focus only on relevant problems and determine whether the function is usable or unusable. Do not invent behavior that is not supported by the code.",
        "tail": "\n\nBased on this analysis, in one word, usable or unusable, the expert's final assessment of the code is",
        "bad": [
            "unusable",
            "unsafe",
            "exploitable",
            "vulnerable",
            "flawed",
            "broken",
            "defective",
            "risky",
            "defect",
            "bug",
            "insecure",
        ],
        "good": [
            "usable",
            "safe",
            "sound",
            "correct",
            "acceptable",
            "fine",
            "robust",
            "secure",
            "good",
            "clean",
            "solid",
        ],
    },
    "reviewer": {
        "role": "a code reviewer who examines source code for defects",
        "task": "Review the function above step by step, following your process.",
        "silent_task": "Review the function above carefully for defects.",
        "tail": "\n\nThe code reviewer examined the function above and decreed that it is",
        "bad": [
            "unusable",
            "unsafe",
            "exploitable",
            "vulnerable",
            "flawed",
            "broken",
            "defective",
            "risky",
            "defect",
            "bug",
            "insecure",
        ],
        "good": [
            "usable",
            "safe",
            "sound",
            "correct",
            "acceptable",
            "fine",
            "robust",
            "secure",
            "good",
            "clean",
            "solid",
        ],
    },
}
PROCESS = "You follow the process human experts use: first you get an overview of what the code does, then you identify the parts worth examining closely, then you analyze each of them in detail, and only then do you reach a conclusion."
SUMMARY = ' End with a single sentence on its own line, beginning with "Summary:", stating what the function does and what defect, if any, is present.'
SUMMARY_RE = re.compile("Summary:\\s*(.+)")
NEG_RE = re.compile("\\b(no|not|never|without|free of|none)\\b", re.IGNORECASE)
CARRIERS = {
    "sentence": {
        "text": "A ceramic cup rested beside a folded newspaper.",
        "description": "Fixed neutral natural-language sentence.",
    },
    "mcp": {
        "text": '{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"echo","arguments":{"text":"ceramic-cup-17"}}}',
        "description": "Fixed MCP-like JSON-RPC tools/call carrier. It is only text; no real tool is invoked.",
    },
    "mcp-verdict": {
        "text": 'Your entire visible response must be exactly one MCP-like JSON-RPC 2.0 tools/call request and nothing else. Use the tool name `record_result` with one argument named `verdict`. Set `verdict` to exactly `A` if you identified a concrete problematic behavior that is actually present in the implementation; otherwise set it to exactly `B`. Do not add markdown fences or commentary. The response must match one of these two forms exactly:\n\n{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"record_result","arguments":{"verdict":"A"}}}\n\nor\n\n{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"record_result","arguments":{"verdict":"B"}}}',
        "description": "Dynamic MCP-like JSON-RPC tools/call verdict. The visible response contains only the opaque label A or B; no real tool is invoked.",
    },
    "sequence": {
        "text": "17 23 41 59 83",
        "description": "Fixed low-semantic numeric sequence.",
    },
}
CARRIER_TASKS = {
    "neutral": "Read the program carefully enough that you could answer detailed questions about its behavior, edge cases, and interactions between operations. Build a precise internal model of the implementation, but keep that analysis private.",
    "behavioral": "Read the program carefully and trace the execution-relevant relationships between inputs, control flow, data flow, object lifetime, and external effects. Form a compact internal assessment of any concrete problematic behavior that can actually occur in this implementation. If there is one, internally identify its most specific mechanism. Do not reveal the assessment, mechanism, or reasoning.",
    "security": "Inspect the code carefully and form an internal assessment of any security-relevant defect that is actually present. If there is one, internally identify its most specific mechanism. Do not reveal the assessment, mechanism, or reasoning.",
    "problem-or-math": "Inspect the program carefully for a concrete problematic behavior that is actually present in this implementation. If you identify one, focus your internal reasoning strongly on that specific problem: trace the relevant operations and relationships in detail and keep the most specific mechanism active in mind. If you do not identify any concrete problem in the program, instead reason internally through the calculation 9 x 9 - 6 x 6 step by step, keeping its intermediate arithmetic active in mind. Do not reveal which branch you followed, the program assessment, the mechanism, the calculation, its intermediate steps, or its result.",
    "problem-focus": "Inspect the program carefully for a concrete problematic behavior that is actually present in this implementation. If you identify one, focus your internal reasoning strongly on that specific problem: trace the relevant operations and relationships in detail and keep the most specific mechanism active in mind.",
}
CONDITIONS = ("carrier", "cot", "silent")
DEFAULT_LAST_TOKEN_WINDOWS = (1,)
REASONING_CHOICES = ("none", "low", "medium", "high", "xhigh")
SUPPORTED_REASONING = {
    "gpt-oss-20b": {"low", "medium", "high"},
    "qwen3.8-27b": {"low", "medium", "xhigh"},
}


# Additive readout profiles. These never change the prompt or the generated verdict.
PERSONALIZED_ANCHORS = {
    "gemma3-27b": {
        "code_post_space": "After the single literal space immediately following </code>",
        "mcp_jsonrpc_2": "After 2 in the generated jsonrpc version 2.0",
        "mcp_verdict_ver": "After ver in the generated verdict key",
    },
    "gemma4-31b": {
        "mcp_id_1": "After the first digit 1 in the generated id value 17",
        "mcp_verdict_ver": "After ver in the generated verdict key",
    },
    "gpt-oss-20b": {
        "prompt_follow_the": "After the in the post-code instruction follow the previous",
        "mcp_jsonrpc_dot": "After the decimal point in generated jsonrpc version 2.0",
        "mcp_verdict_ver": "After ver in the generated verdict key",
    },
}


def personalized_definitions(spec):
    if not spec.get("personalized_anchors"):
        return {}
    return PERSONALIZED_ANCHORS.get(spec["model_name"], {})


def personalized_prompt_anchors(prefix, spec):
    definitions = personalized_definitions(spec)
    anchors = {}
    close = prefix.rfind("</code>")
    if close < 0 and definitions:
        raise ValueError("Personalized anchors require the closing code marker")
    end = close + len("</code>")
    if "code_post_space" in definitions:
        if prefix[end : end + 1] != " ":
            raise ValueError("Expected a literal space immediately after </code>")
        anchors["code_post_space"] = end + 1
    if "prompt_follow_the" in definitions:
        matches = list(
            re.finditer(
                r"Plese follow(?P<target> the) previous instruction", prefix[end:]
            )
        )
        if len(matches) != 1:
            raise ValueError("Cannot uniquely locate post-code follow the instruction")
        anchors["prompt_follow_the"] = end + matches[0].end("target")
    return anchors


def personalized_output_anchors(prefix, response, spec):
    definitions = personalized_definitions(spec)
    if not definitions or not parse_mcp_verdict(response)[1]:
        return {}
    anchors = {}
    patterns = {
        "mcp_jsonrpc_2": (r'"jsonrpc"\s*:\s*"(?P<target>2)\.0"', "target"),
        "mcp_jsonrpc_dot": (r'"jsonrpc"\s*:\s*"2(?P<target>\.)0"', "target"),
        "mcp_id_1": (r'"id"\s*:\s*(?P<target>1)7(?=\s*[,}])', "target"),
        "mcp_verdict_ver": (r'"(?P<target>ver)dict"\s*:\s*"[AB]"', "target"),
    }
    for name in definitions:
        if name not in patterns:
            continue
        pattern, group = patterns[name]
        matches = list(re.finditer(pattern, response))
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely locate {name} in actual MCP output")
        anchors[name] = len(prefix) + matches[0].end(group)
    return anchors


class _ActualTokenModel:
    """Read the actual generated sequence rather than re-tokenizing across its boundary."""

    def __init__(self, model, text, token_ids):
        self.model = model
        self.text = text
        self.token_ids = token_ids

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode(self, text, *, max_length=512):
        if text != self.text or len(self.token_ids) > max_length:
            raise ValueError(
                "Actual-token readout cannot change or truncate the sequence"
            )
        return torch.tensor(
            [self.token_ids], dtype=torch.long, device=self.model.input_device
        )


def actual_generation_tokenization(tok, prefix, generated_ids, prompt_ids):
    matches = [
        item
        for item in _tokenize_prompt_with_offsets(tok, prefix)
        if item[0] == prompt_ids
    ]
    if not matches:
        raise ValueError(
            "Cannot match actual generation prompt IDs to character offsets"
        )
    _, offsets, special = matches[0]
    offsets = list(offsets)
    previous = ""
    for index in range(len(generated_ids)):
        text = tok.decode(generated_ids[: index + 1], skip_special_tokens=True)
        if not text.startswith(previous):
            raise ValueError(
                "Non-monotonic output decoding: refusing guessed token offsets"
            )
        offsets.append((len(prefix) + len(previous), len(prefix) + len(text)))
        previous = text
    return prefix + previous, (prompt_ids + generated_ids, offsets, special)


def cache_action(remove=None):
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    if remove is None:
        for repo in sorted(cache.repos, key=lambda item: item.repo_id):
            print(f"{repo.repo_id}: {repo.size_on_disk / 1024**3:.2f} GiB")
        return
    spec = MODELS.get(remove)
    if spec is None:
        raise SystemExit(f"Modello sconosciuto: {remove}")
    repo_id = spec["hf_id"]
    # if repo_id == MODELS["qwen3.8-27b"]["hf_id"]:
    #     raise SystemExit("Qwen3.8 è protetto: nessun file eliminato")
    repos = [
        repo
        for repo in cache.repos
        if repo.repo_type == "model" and repo.repo_id == repo_id
    ]
    revisions = [revision.commit_hash for repo in repos for revision in repo.revisions]
    if not revisions:
        print(f"Nessuna copia in cache di {repo_id}")
        return
    strategy = cache.delete_revisions(*revisions)
    print(
        f"Rimozione cache del solo {repo_id}: {strategy.expected_freed_size / 1024**3:.2f} GiB",
        flush=True,
    )
    strategy.execute()
    print("Cache del modello rimossa. Lens e risultati conservati.")


def load_run_lens(spec, filename, revision=None, refresh=False, kind="jlens"):
    from huggingface_hub import hf_hub_download

    repo = spec.get("lens_repo", LENS_REPO)
    local = Path(filename).expanduser()
    if local.is_file():
        if refresh:
            raise ValueError(
                "--refresh-lens richiede una lens remota, non --lens-file locale"
            )
        path = local.resolve()
        commit = None
    else:
        path = Path(
            hf_hub_download(
                repo, filename=filename, revision=revision, force_download=refresh
            )
        )
        parts = path.parts
        commit = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    provenance = payload.get("provenance", {})
    if kind == "jpp":
        if provenance.get("capture_point") != "block_output":
            raise ValueError(f"J++ capture_point non supportato: {provenance}")
        if provenance.get("stream_reduce", "none") not in (None, "none"):
            raise ValueError("Questa pipeline richiede una singola residual stream")
    lens = jlens.JacobianLens(
        jacobians=payload["J"],
        n_prompts=payload["n_prompts"],
        d_model=payload["d_model"],
    )
    info = {
        "method": kind,
        "provenance": json.loads(json.dumps(provenance, default=str)),
        "readout_filter": "letter_or_digit_ranking_only" if kind == "jpp" else "none",
        "metric_distribution": "full_unfiltered_vocabulary",
        "repo": repo,
        "file": filename,
        "revision_requested": revision,
        "revision_resolved": commit,
        "sha256": digest,
        "refresh_requested": refresh,
    }
    print(f"Lens: {repo}/{filename} revision={commit} sha256={digest}", flush=True)
    return lens, info


def selected_layers(value, spec, lens):
    if value is None:
        if spec["band"] is None:
            return sorted(int(layer) for layer in lens.source_layers)
        return [
            layer
            for layer in range(spec["band"][0], spec["band"][1] + 1)
            if layer in lens.source_layers
        ]
    if value == "all":
        return sorted(int(layer) for layer in lens.source_layers)
    result = set()
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) == 1:
            result.add(int(bounds[0]))
        elif len(bounds) == 2:
            begin, end = map(int, bounds)
            if end < begin:
                raise ValueError("Intervallo layer invertito")
            result.update(range(begin, end + 1))
        else:
            raise ValueError("Layer: usa all oppure 40-58 oppure 40,45,50")
    missing = result - set(lens.source_layers)
    if not result or missing:
        raise ValueError(f"Layer assenti dalla lens: {sorted(missing)}")
    return sorted(result)


def check_anchor_boundaries(
    tok, spec, cases, chat, carrier_name, carrier_text, carrier_task
):
    if carrier_name != "mcp-verdict":
        raise ValueError("--check-anchors richiede --carrier mcp-verdict")
    failed = False
    for case in cases:
        prefix, anchors = build_carrier_prompt(
            case.code, chat, tok, spec, carrier_name, carrier_text, carrier_task
        )
        for label in ("A", "B"):
            response = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 17,
                    "method": "tools/call",
                    "params": {
                        "name": "record_result",
                        "arguments": {"verdict": label},
                    },
                },
                separators=(",", ":"),
            )
            output, _ = _mcp_output_anchors(prefix, response)
            output.update(personalized_output_anchors(prefix, response, spec))
            prompt_ids = [int(x) for x in tok.encode(prefix)]
            generated_ids = [
                int(x) for x in tok.encode(response, add_special_tokens=False)
            ]
            full, tokenization = actual_generation_tokenization(
                tok, prefix, generated_ids, prompt_ids
            )
            ids = tokenization[0]
            resolved = _resolve_char_anchors(
                tok, full, anchors | output, ids, tokenization
            )
            report = {}
            for name, item in resolved.items():
                end = item["end"]
                exact = item["boundary_exact"] and end > 0
                failed |= not exact
                report[name] = {
                    **item,
                    "tail_tokens": tok.batch_decode(
                        [[i] for i in ids[max(0, end - 3) : end]]
                    ),
                    "following_tokens": tok.batch_decode(
                        [[i] for i in ids[end : end + 2]]
                    ),
                }
            print(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "verdict": label,
                        "hf_id": spec["hf_id"],
                        "chat_template": chat,
                        "anchors": report,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if failed:
        raise SystemExit("Almeno un anchor non è esatto: non avviare la run completa")


def reasoning_template_kwargs(spec, *, condition):
    effort = spec.get("reasoning_effort")
    model_name = spec.get("model_name")
    if model_name == "qwen3.6-27b":
        return {
            "enable_thinking": condition != "carrier"
            and not spec.get("effort_explicit_none")
        }
    if model_name == "qwen3.8-27b":
        if effort is None:
            return (
                {"enable_thinking": False}
                if condition == "carrier" or spec.get("effort_explicit_none")
                else {}
            )
        return {"enable_thinking": True, "reasoning_effort": effort}
    if model_name == "gpt-oss-20b":
        return {"reasoning_effort": effort} if effort is not None else {}
    return {}


def build_carrier_prompt(
    code, chat, tok, spec, carrier_name, carrier_text, carrier_task
):
    chat_kwargs = reasoning_template_kwargs(spec, condition="carrier")
    if carrier_name == "mcp-verdict":
        output_spec = carrier_text
    else:
        output_spec = f"Your entire visible response must be exactly the literal text below, regardless of the result of the analysis. Do not add markdown fences or commentary.\n\n{carrier_text}"
    system = f"You are a careful code analyst. {carrier_task}\n\nKeep your reasoning and explanation private. Follow the output specification below for your entire visible response.\n\n{output_spec}"
    if carrier_name == "mcp-verdict":
        instruction_tail = "Plese follow the previous instruction specified in the system message and nothing else. Return exactly one JSON-RPC request using record_result. Your response must start with { and contain only the JSON request. Do not write analysis, explanations, or markdown."
    else:
        instruction_tail = "Return exactly the fixed carrier text specified in the system message, regardless of your assessment of the program. Do not write analysis, explanations, or markdown."
    code_block = f"<code>\n{code}\n</code> "
    user = f"Analyze the following program.\n\n{code_block}\n\n{instruction_tail}"
    if chat:
        prefix = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        if spec.get("gpt_oss_direct_final"):
            if not prefix.endswith("<|start|>assistant"):
                raise RuntimeError("Unexpected GPT-OSS generation prefix")
            prefix += "<|channel|>final<|message|>"
    else:
        prefix = f"{system}\n\n{user}\n\nAssistant:"

    def char_end_after(marker, *, last=True):
        idx = prefix.rfind(marker) if last else prefix.find(marker)
        if idx < 0:
            raise RuntimeError(
                f"carrier anchor marker not found in rendered prompt: {marker!r}"
            )
        return idx + len(marker)

    anchors = {
        "code_end": char_end_after("</code>"),
    }
    if spec.get("code_position_anchors"):
        block_start = prefix.rfind(code_block)
        if block_start < 0:
            raise RuntimeError("Code block not preserved in rendered prompt")
        content = code.rstrip()
        if not content:
            raise ValueError("Cannot locate a code-content anchor in empty code")
        anchors["code_content_end"] = block_start + len("<code>\n") + len(content)
        whitespace_end = anchors["code_end"]
        while whitespace_end < len(prefix) and prefix[whitespace_end].isspace():
            whitespace_end += 1
        anchors["code_whitespace_end"] = whitespace_end
    anchors.update(personalized_prompt_anchors(prefix, spec))
    if carrier_name != "mcp-verdict":
        anchors.update(
            instruction_end=char_end_after(instruction_tail),
            assistant_start=len(prefix),
        )
    think_idx = prefix.rfind("<think>")
    if think_idx >= 0 and carrier_name != "mcp-verdict":
        anchors["assistant_prethink"] = think_idx
    return (prefix, anchors)


def carrier_quality(text, carrier_text):
    return (text or "").strip() == carrier_text.strip()


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_mcp_verdict(text):
    try:
        obj = json.loads((text or "").strip(), object_pairs_hook=_unique_json_pairs)
    except (ValueError, TypeError):
        return (None, False)
    expected_top = {"jsonrpc", "id", "method", "params"}
    if not isinstance(obj, dict) or set(obj) != expected_top:
        return (None, False)
    if obj.get("jsonrpc") != "2.0" or obj.get("id") != 17:
        return (None, False)
    if obj.get("method") != "tools/call":
        return (None, False)
    params = obj.get("params")
    if not isinstance(params, dict) or set(params) != {"name", "arguments"}:
        return (None, False)
    if params.get("name") != "record_result":
        return (None, False)
    args = params.get("arguments")
    if not isinstance(args, dict) or set(args) != {"verdict"}:
        return (None, False)
    verdict = args.get("verdict")
    if verdict == "A":
        return ("vulnerable", True)
    elif verdict == "B":
        return ("safe", True)
    return (None, False)


def build_prompts(p, code, chat, silent, tok, spec):
    system = f"You are {p['role']}. {PROCESS}"
    chat_kwargs = reasoning_template_kwargs(
        spec, condition="silent" if silent else "cot"
    )
    if silent:
        user = f"{code}\n\n{p['silent_task']} Do not provide your reasoning, analysis, or explanation. Give only the requested final assessment.{p['tail']}"
        if chat:
            prefix = tok.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            )
        else:
            prefix = f"{system}\n\n{user}"
        return (prefix, None)
    user = f"{code}\n\n{p['task']} Do not state a conclusion yet.{SUMMARY}\n\nAnalysis:"
    if chat:
        prefix = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
    else:
        prefix = f"{system}\n\n{user}"
    return (prefix, (system, user))


def verdict_prompt(p, prefix, parts, reasoning, chat, tok, spec):
    if not chat or parts is None:
        return prefix + reasoning + p["tail"]
    system, user = parts
    chat_kwargs = (
        {"enable_thinking": False}
        if spec.get("model_name") in {"qwen3.8-27b", "qwen3.6-27b"}
        else reasoning_template_kwargs(spec, condition="cot")
    )
    return tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": p["tail"].strip()},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )


def validate_model_request(spec, condition, reasoning_effort):
    if not spec.get("no_explicit_reasoning"):
        return
    if condition == "cot":
        raise ValueError(
            f"{spec['hf_id']}: CoT disabilitato per questa configurazione; "
            "usa --condition carrier oppure silent"
        )
    if reasoning_effort not in (None, "none"):
        raise ValueError(
            f"{spec['hf_id']}: nessuna modalita thinking; usa --reasoning-effort none"
        )


def validate_model_runtime(spec):
    minimum = spec.get("min_transformers")
    if minimum:
        from packaging.version import Version

        installed = transformers.__version__
        if Version(installed) < Version(minimum):
            raise RuntimeError(
                f"{spec['hf_id']}: transformers {installed} non consentito; "
                f"usa transformers>={minimum} per questa configurazione. "
                "Puoi usare uv run --with 'transformers>=5.17,<6' senza "
                "modificare il lockfile del progetto."
            )


def load_model(spec):
    validate_model_runtime(spec)
    q, hf_id = (spec["quantize"], spec["hf_id"])
    if spec.get("image_text") and q in ("4bit", "bnb4"):
        bnb = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return transformers.AutoModelForImageTextToText.from_pretrained(
            hf_id, quantization_config=bnb, device_map="auto"
        ).eval()
    if q == "mxfp4":
        cfg = transformers.Mxfp4Config(dequantize=False)
        return transformers.AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=cfg, device_map="auto"
        ).eval()
    if q in ("4bit", "bnb4"):
        kw = {}
        if q == "bnb4":
            c = transformers.AutoConfig.from_pretrained(hf_id)
            c.quantization_config = None
            kw["config"] = c
        bnb = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return transformers.AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb, device_map="auto", **kw
        ).eval()
    return transformers.AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, device_map="auto"
    ).eval()


def _eos_token_ids(hf, tok):
    eos = getattr(hf.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tok.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(x) for x in eos}


def _completed_mcp_prefix(text):
    start = len(text) - len(text.lstrip())
    if not text[start:].startswith("{"):
        return None
    try:
        _, end = json.JSONDecoder(object_pairs_hook=_unique_json_pairs).raw_decode(
            text, start
        )
    except (ValueError, TypeError):
        return None
    prefix = text[:end]
    if not parse_mcp_verdict(prefix)[1]:
        return None
    return prefix, text[end:]


class CompleteMCPStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, tok, prompt_width):
        self.tok = tok
        self.prompt_width = prompt_width
        self.completed = {}

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tok.batch_decode(
            input_ids[:, self.prompt_width :], skip_special_tokens=True
        )
        for index, text in enumerate(texts):
            if index in self.completed:
                continue
            match = _completed_mcp_prefix(text)
            if match is not None:
                response, suffix = match
                self.completed[index] = {
                    "response": response,
                    "reason": "valid_mcp_json",
                    "generated_tokens": input_ids.shape[1] - self.prompt_width,
                    "raw_text_at_stop": text,
                    "discarded_suffix": suffix,
                }
        return torch.tensor(
            [i in self.completed for i in range(input_ids.shape[0])],
            dtype=torch.bool,
            device=input_ids.device,
        )


def generate_batch(hf, tok, prompts, n, *, stop_at_mcp=False, generation_info=None):
    if not prompts:
        return []
    x = tok(prompts, return_tensors="pt", padding=True).to(hf.device)
    prompt_width = x["input_ids"].shape[1]
    stopper = None
    extra = {}
    if stop_at_mcp:
        if (getattr(hf.generation_config, "num_beams", 1) or 1) != 1:
            raise ValueError("MCP stopping requires greedy generation without beams")
        if (getattr(hf.generation_config, "num_return_sequences", 1) or 1) != 1:
            raise ValueError("MCP stopping requires one response per prompt")
        stopper = CompleteMCPStoppingCriteria(tok, prompt_width)
        extra["stopping_criteria"] = transformers.StoppingCriteriaList([stopper])
    with torch.no_grad():
        out = hf.generate(
            **x,
            max_new_tokens=n,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            **extra,
        )
    eos_ids = _eos_token_ids(hf, tok)
    results = []
    for index, row in enumerate(out):
        if stopper is not None and index in stopper.completed:
            completion = stopper.completed[index]
            results.append((completion["response"], False))
            if generation_info is not None:
                prompt_ids = x["input_ids"][index][
                    x["attention_mask"][index].bool()
                ].tolist()
                generated_ids = row[
                    prompt_width : prompt_width + completion["generated_tokens"]
                ].tolist()
                generation_info.append(
                    dict(
                        completion,
                        prompt_input_ids=prompt_ids,
                        generated_ids=generated_ids,
                    )
                )
            continue
        ids = row[prompt_width:].tolist()
        first_eos = next((i for i, t in enumerate(ids) if t in eos_ids), None)
        cut = len(ids) >= n and first_eos is None
        if first_eos is not None:
            ids = ids[:first_eos]
        text = tok.decode(ids, skip_special_tokens=True)
        results.append((text, cut))
        if generation_info is not None:
            generation_info.append(
                {
                    "reason": "max_new_tokens" if cut else "eos_or_model_stop",
                    "generated_tokens": len(ids),
                    "prompt_input_ids": x["input_ids"][index][
                        x["attention_mask"][index].bool()
                    ].tolist(),
                    "generated_ids": ids,
                }
            )
    return results


def batches(xs, size):
    for i in range(0, len(xs), size):
        yield (i, xs[i : i + size])


def last_summary(text):
    m = SUMMARY_RE.findall(text or "")
    return m[-1].strip() if m else None


def parse_pred(text, p):
    if not text:
        return None
    bad, good = (set(p["bad"]), set(p["good"]))
    rx = re.compile(
        "\\b(" + "|".join(sorted(bad | good, key=len, reverse=True)) + ")\\b",
        re.IGNORECASE,
    )
    for m in rx.finditer(text):
        w = m.group(1).lower()
        cls = "vulnerable" if w in bad else "safe"
        if NEG_RE.search(text[max(0, m.start() - 25) : m.start()]):
            cls = "safe" if cls == "vulnerable" else "vulnerable"
        return cls
    return None


def _tokenize_prompt_with_offsets(tok, prompt):
    candidates = []
    for add_special_tokens in (True, False):
        try:
            enc = tok(
                prompt,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
            )
        except (TypeError, NotImplementedError, ValueError):
            continue
        ids = enc.get("input_ids")
        offsets = enc.get("offset_mapping")
        if ids is None or offsets is None:
            continue
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if (
            offsets
            and isinstance(offsets[0], list)
            and offsets[0]
            and isinstance(offsets[0][0], (list, tuple))
        ):
            offsets = offsets[0]
        item = (
            [int(x) for x in ids],
            [(int(a), int(b)) for a, b in offsets],
            add_special_tokens,
        )
        if not any(item[0] == x[0] for x in candidates):
            candidates.append(item)
    return candidates


def _lcp_len(a, b):
    lim = min(len(a), len(b))
    i = 0
    while i < lim and int(a[i]) == int(b[i]):
        i += 1
    return i


def _resolve_char_anchors(
    tok, prompt, anchor_char_ends, full_token_ids, tokenization=None
):
    full_ids = [int(x) for x in full_token_ids]
    candidates = (
        [tokenization]
        if tokenization is not None
        else _tokenize_prompt_with_offsets(tok, prompt)
    )
    if tokenization is not None and tokenization[0] != full_ids:
        raise ValueError("Readout IDs differ from actual generation IDs")
    if not candidates:
        out = {}
        for name, char_end in anchor_char_ends.items():
            prefix = prompt[: int(char_end)]
            prefix_candidates = []
            for add_special_tokens in (True, False):
                try:
                    ids = tok.encode(prefix, add_special_tokens=add_special_tokens)
                except TypeError:
                    ids = tok.encode(prefix)
                ids = [int(x) for x in ids]
                prefix_candidates.append(ids)
            best = max(prefix_candidates, key=lambda ids: _lcp_len(ids, full_ids))
            lcp = _lcp_len(best, full_ids)
            out[name] = {
                "end": lcp,
                "boundary_exact": lcp == len(best),
                "requested_token_count": len(best),
                "char_end": int(char_end),
                "method": "prefix_lcp_fallback",
                "prompt_tokenization_lcp": lcp,
            }
        return out
    ids, offsets, add_special_tokens = max(
        candidates, key=lambda item: _lcp_len(item[0], full_ids)
    )
    lcp = _lcp_len(ids, full_ids)
    out = {}
    first_text_idx = next(
        (i for i, (a, b) in enumerate(offsets) if not a == b == 0), len(offsets)
    )
    for name, char_end_raw in anchor_char_ends.items():
        char_end = int(char_end_raw)
        requested_char_end = char_end
        if name in {"code_end", "code_content_end", "code_whitespace_end"}:
            for a, b in offsets[:lcp]:
                if a < char_end < b:
                    suffix = prompt[char_end:b]
                    if not suffix or not suffix.isspace():
                        raise ValueError(
                            f"{name} crosses a token containing subsequent non-whitespace text"
                        )
                    char_end = b
                    break
        if char_end >= len(prompt):
            end = min(lcp, len(full_ids))
            boundary_exact = lcp == len(ids)
            seen_text = True
        else:
            end = min(first_text_idx, lcp)
            boundary_exact = True
            seen_text = False
            for idx, (a, b) in enumerate(offsets[:lcp]):
                if a == b == 0:
                    if idx < first_text_idx:
                        end = idx + 1
                    continue
                if b <= char_end:
                    end = idx + 1
                    seen_text = True
                    continue
                if a < char_end < b:
                    boundary_exact = False
                break
            end = min(end, lcp, len(full_ids))
            if seen_text:
                last_end = 0
                for idx in range(min(end, len(offsets)) - 1, -1, -1):
                    a, b = offsets[idx]
                    if not a == b == 0:
                        last_end = b
                        break
                boundary_exact = boundary_exact and last_end == char_end
            else:
                boundary_exact = False
        out[name] = {
            "end": end,
            "boundary_exact": bool(boundary_exact),
            "requested_token_count": end,
            "char_end": char_end,
            "requested_char_end": requested_char_end,
            "boundary_adjustment": "right_whitespace"
            if char_end != requested_char_end
            else "none",
            "method": "actual_generation_offsets"
            if tokenization is not None
            else "full_prompt_offsets",
            "prompt_tokenization_lcp": lcp,
            "prompt_tokenization_length": len(ids),
            "add_special_tokens": bool(add_special_tokens),
        }
    return out


def semantic_token_mask(tok, size):
    allowed = []
    for begin in range(0, size, 4096):
        texts = tok.batch_decode(
            [[i] for i in range(begin, min(begin + 4096, size))],
            skip_special_tokens=False,
        )
        allowed.extend(any(char.isalnum() for char in text) for text in texts)
    return torch.tensor(allowed, dtype=torch.bool)


def ranked_indices(rows, k, mask=None):
    if mask is None:
        return rows.topk(min(k, rows.shape[-1]), dim=-1).indices
    mask = mask.to(rows.device)
    k = min(k, int(mask.sum()))
    if k < 1:
        raise ValueError("Readout filtering removed every token")
    return torch.cat(
        [
            part.masked_fill(~mask, -torch.inf).topk(k, dim=-1).indices
            for part in rows.split(32)
        ]
    )


def _aggregate_window_topk(rows, windows, k, end, ranking_mask=None):
    out = {}
    if end <= 0 or rows.numel() == 0:
        return out
    max_w = min(max(windows), end)
    begin = end - max_w
    focus_rows = rows[begin:end].float()
    focus_logp = focus_rows - focus_rows.logsumexp(dim=-1, keepdim=True)
    vocab_k = min(int(k), int(focus_logp.shape[-1]))
    for w in windows:
        used = min(int(w), end)
        rel_start = max_w - used
        agg = focus_logp[rel_start:].max(dim=0).values
        if ranking_mask is not None:
            agg = agg.masked_fill(~ranking_mask.to(agg.device), -torch.inf)
            vocab_k = min(vocab_k, int(ranking_mask.sum()))
        vals, ids = agg.topk(vocab_k)
        out[str(w)] = {
            "used_positions": used,
            "ids": [int(x) for x in ids.detach().cpu().tolist()],
            "logp": [float(x) for x in vals.detach().cpu().tolist()],
        }
    return out


def _aggregate_window_targets(rows, windows, term_ids, end):
    out = {}
    if end <= 0 or rows.numel() == 0 or (not term_ids):
        return out
    names = list(term_ids)
    ids = torch.tensor(
        [int(term_ids[name]) for name in names], dtype=torch.long, device=rows.device
    )
    max_w = min(max(windows), end)
    begin = end - max_w
    focus_rows = rows[begin:end].float()
    focus_logp = focus_rows - focus_rows.logsumexp(dim=-1, keepdim=True)
    selected = focus_logp[:, ids]
    for w in windows:
        used = min(int(w), end)
        rel_start = max_w - used
        vals = selected[rel_start:].max(dim=0).values
        out[str(w)] = {
            "used_positions": used,
            "logp": {name: float(vals[j]) for j, name in enumerate(names)},
        }
    return out


def _mcp_output_anchors(prefix, response):
    label, valid = parse_mcp_verdict(response)
    if not valid:
        return {}, {"available": False, "reason": "invalid_mcp_response"}
    name = re.search(r'"name"\s*:\s*"(record_result)"', response)
    verdict = re.search(r'"verdict"\s*:\s*"([AB])"', response)
    if name is None or verdict is None:
        return {}, {"available": False, "reason": "unsupported_json_spelling"}
    return {
        "mcp_record": len(prefix) + name.start(1) + len("record"),
        "mcp_pre_verdict": len(prefix) + verdict.start(1),
    }, {"available": True, "verdict": verdict.group(1), "prediction": label}


def _choice_ids(tok):
    result = {}
    for label in ("A", "B"):
        ids = tok.encode(label, add_special_tokens=False)
        if len(ids) != 1 or tok.decode(ids, skip_special_tokens=False) != label:
            raise ValueError(f"Choice {label} is not an exact single token")
        result[label] = int(ids[0])
    if result["A"] == result["B"]:
        raise ValueError("A and B must have distinct token IDs")
    return result


def _aggregate_choice_scores(rows, windows, end, choices):
    if not choices or end <= 0 or rows.numel() == 0:
        return {}
    if max(choices.values()) >= rows.shape[-1]:
        raise ValueError("Choice ID exceeds readout vocabulary")
    selected = rows[max(0, end - max(windows)) : end].float()
    pair = selected[:, [choices["A"], choices["B"]]]
    logp = pair - selected.logsumexp(-1, keepdim=True)
    conditional = torch.softmax(pair, dim=-1)
    entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(-1)
    out = {}
    for window in windows:
        used = min(window, end)
        probability = logp[-used:].exp().mean(0).detach().cpu().tolist()
        cond = conditional[-used:].mean(0).detach().cpu().tolist()
        out[str(window)] = {
            "used_positions": used,
            "aggregation": "mean_of_per_position_statistics",
            "token_ids": choices,
            "logits": dict(
                zip(
                    ("A", "B"),
                    pair[-used:].mean(0).detach().cpu().tolist(),
                    strict=True,
                )
            ),
            "logp": dict(
                zip(
                    ("A", "B"),
                    logp[-used:].mean(0).detach().cpu().tolist(),
                    strict=True,
                )
            ),
            "probability": dict(zip(("A", "B"), probability, strict=True)),
            "conditional_probability": dict(zip(("A", "B"), cond, strict=True)),
            "logit_margin_A_minus_B": float((pair[-used:, 0] - pair[-used:, 1]).mean()),
            "pair_probability": float(sum(probability)),
            "conditional_entropy_nats": float(entropy[-used:].mean()),
        }
    return out


def _decode_focus_group(tok, group):
    decoded = {}
    for w, entry in group.items():
        decoded[w] = {
            "used_positions": entry["used_positions"],
            "ids": entry["ids"],
            "tokens": tok.batch_decode([[int(t)] for t in entry["ids"]]),
            "logp": entry["logp"],
        }
    return decoded


def _decode_anchor_readout(tok, focuses, windows, top_k):
    if not focuses:
        return None
    assistant_end = None
    if "assistant_start" in focuses:
        assistant_end = focuses["assistant_start"]["anchor_end_exclusive"]
    out = {
        "aggregation": "max_logprob_over_last_n_positions_ending_at_anchor",
        "windows": list(windows),
        "top_k": int(top_k),
        "top_k_by_anchor": {
            name: int(focus.get("top_k", top_k)) for name, focus in focuses.items()
        },
        "anchors": {},
    }
    for name, focus in focuses.items():
        anchor = {
            "top_k": int(focus.get("top_k", top_k)),
            "anchor_end_exclusive": focus["anchor_end_exclusive"],
            "boundary_exact": focus["boundary_exact"],
            "prefix_token_count_requested": focus["prefix_token_count_requested"],
            "anchor_char_end": focus.get("anchor_char_end"),
            "requested_anchor_char_end": focus.get("requested_anchor_char_end"),
            "boundary_adjustment": focus.get("boundary_adjustment", "none"),
            "boundary_method": focus.get("boundary_method"),
            "personalized": focus.get("personalized", False),
            "prompt_tokenization_lcp": focus.get("prompt_tokenization_lcp"),
            "prompt_tokenization_length": focus.get("prompt_tokenization_length"),
            "tail_positions": focus["tail_positions"],
            "tail_token_ids": focus["tail_token_ids"],
            "tail_tokens": tok.batch_decode(
                [[int(t)] for t in focus["tail_token_ids"]]
            ),
            "model": _decode_focus_group(tok, focus.get("model", {})),
            "lens": {},
            "model_target_logp": focus.get("model_targets", {}),
            "lens_target_logp": {},
        }
        if focus.get("model_choice_scores"):
            anchor["choice_scores"] = {
                "version": "ab-full-logits-v1",
                "model": focus["model_choice_scores"],
                "lens": {
                    str(layer): values
                    for layer, values in focus["lens_choice_scores"].items()
                },
            }
        if focus.get("model_vocab_scores") is not None:
            anchor["vocab_scores"] = {
                "version": "sg-full-vocab-v1",
                "profile": "full",
                "model": focus["model_vocab_scores"],
                "lens": {
                    str(layer): values
                    for layer, values in focus.get("lens_vocab_scores", {}).items()
                },
            }
        if assistant_end is not None:
            anchor["distance_to_assistant_start_tokens"] = (
                assistant_end - focus["anchor_end_exclusive"]
            )
        for layer, group in focus.get("lens", {}).items():
            anchor["lens"][str(layer)] = _decode_focus_group(tok, group)
        for layer, group in focus.get("lens_targets", {}).items():
            anchor["lens_target_logp"][str(layer)] = group
        out["anchors"][name] = anchor
    return out


def _load_sg_vocab(path, tok, model_id):
    path = Path(path).expanduser()
    vocab_file = path / "token_groups.json" if path.is_dir() else path
    raw = vocab_file.read_bytes()
    compiled = json.loads(raw)
    meta = compiled.get("meta", {})
    if meta.get("tokenizer_id") != model_id:
        raise ValueError("SG vocabulary tokenizer_id does not match the model hf_id")
    catalog_file = vocab_file.parent / "concepts.json"
    if catalog_file.exists() and meta.get("catalog_sha256"):
        actual = hashlib.sha256(catalog_file.read_bytes()).hexdigest()
        if actual != meta["catalog_sha256"]:
            raise ValueError(
                "SG catalog changed after compilation; recompile the vocabulary"
            )
    groups = {}
    for name, values in compiled.get("groups", {}).items():
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in values):
            raise ValueError("SG group IDs must be nonnegative integers")
        groups[name] = sorted(set(values))
    if not groups:
        raise ValueError("SG vocabulary contains no groups")
    for token_id in sorted({i for values in groups.values() for i in values}):
        entry = compiled.get("tokens", {}).get(str(token_id))
        if not entry or tok.decode([token_id], skip_special_tokens=False) != entry.get(
            "text"
        ):
            raise ValueError(f"SG vocabulary ID/surface mismatch: {token_id}")
    if "union" in groups or "outside" in groups:
        raise ValueError("SG union and outside are reserved derived groups")
    groups["union"] = sorted({i for values in groups.values() for i in values})
    payload = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    signature = hashlib.sha256(payload).hexdigest()
    registry = {
        "groups": groups,
        "names": sorted(groups),
        "signature": signature,
        "cache": {},
    }
    info = {
        "version": "sg-full-vocab-v1",
        "profile": "full",
        "tokenizer_id": model_id,
        "catalog_version": meta.get("version"),
        "catalog_sha256": meta.get("catalog_sha256"),
        "token_groups_sha256": hashlib.sha256(raw).hexdigest(),
        "group_signature": signature,
        "group_counts": {g: len(ids) for g, ids in groups.items()},
        "derived_outside": "full_readout_vocabulary_complement_of_union",
        "sg_definition": "mean_group_logit_minus_mean_full_vocabulary_logit",
        "sg_z_definition": "per_position_sg_divided_by_full_vocabulary_population_logit_std",
        "probability_mass_definition": "exact_group_softmax_mass_before_topk",
        "window_aggregation": "mean_of_per_position_statistics",
        "empty_group": "sg_and_sg_z_null_probability_mass_zero",
    }
    return (registry, info)


def _sg_group_tensors(registry, rows):
    key = (str(rows.device), int(rows.shape[-1]))
    if key not in registry["cache"]:
        names = registry["names"]
        all_ids = sorted({i for ids in registry["groups"].values() for i in ids})
        if all_ids and all_ids[-1] >= rows.shape[-1]:
            raise ValueError(
                "SG vocabulary token ID exceeds the readout vocabulary size"
            )
        lookup = {token_id: j for j, token_id in enumerate(all_ids)}
        matrix = torch.zeros(
            (len(names), len(all_ids)), dtype=torch.float32, device=rows.device
        )
        counts = []
        for j, name in enumerate(names):
            values = registry["groups"][name]
            counts.append(len(values))
            if values:
                matrix[j, [lookup[i] for i in values]] = 1
        denominators = torch.tensor(
            [max(1, n) for n in counts], dtype=torch.float32, device=rows.device
        )
        registry["cache"][key] = (
            torch.tensor(all_ids, dtype=torch.long, device=rows.device),
            matrix.transpose(0, 1).contiguous(),
            (matrix / denominators[:, None]).transpose(0, 1).contiguous(),
            counts,
        )
    return registry["cache"][key]


@torch.no_grad()
def _aggregate_window_vocab_scores(rows, windows, end, registry):
    if registry is None:
        return None
    if end <= 0 or rows.numel() == 0:
        return {}
    end = min(int(end), int(rows.shape[0]))
    max_w = min(max(windows), end)
    focus = rows[end - max_w : end].float()
    if not torch.isfinite(focus).all().item():
        raise ValueError("SG is undefined for nonfinite full-vocabulary logits")
    ids, membership, mean_weights, counts = _sg_group_tensors(registry, focus)
    vocab_mean = focus.mean(dim=-1)
    vocab_std = focus.std(dim=-1, unbiased=False)
    logz = focus.logsumexp(dim=-1)
    selected = focus.index_select(-1, ids)
    group_means = selected @ mean_weights
    sg = group_means - vocab_mean[:, None]
    sg_z = (
        sg
        / torch.where(
            vocab_std > 0, vocab_std, torch.full_like(vocab_std, float("nan"))
        )[:, None]
    )
    group_logp = group_means - logz[:, None]
    group_mass = (selected - logz[:, None]).exp() @ membership
    values = (
        torch.stack([sg, sg_z, group_means, group_logp, group_mass], dim=-1)
        .detach()
        .cpu()
    )
    baselines = torch.stack([vocab_mean, vocab_std, logz], dim=-1).detach().cpu()
    result = {}
    for window in windows:
        used = min(int(window), end)
        start = max_w - used
        aggregate = values[start:].mean(dim=0).tolist()
        base = baselines[start:].mean(dim=0).tolist()
        groups = {}
        for j, name in enumerate(registry["names"]):
            item = {"n_ids": counts[j]}
            for k, field in enumerate(
                ["sg", "sg_z", "mean_logit", "mean_logp", "probability_mass"]
            ):
                value = aggregate[j][k]
                item[field] = float(value) if math.isfinite(value) else None
            if counts[j] == 0:
                item.update(
                    sg=None,
                    sg_z=None,
                    mean_logit=None,
                    mean_logp=None,
                    probability_mass=0.0,
                )
            groups[name] = item
        union = groups["union"]
        outside_count = int(focus.shape[-1]) - union["n_ids"]
        if outside_count:
            factor = -union["n_ids"] / outside_count
            outside_sg = factor * union["sg"] if union["sg"] is not None else 0.0
            outside_sg_z = factor * union["sg_z"] if union["sg_z"] is not None else None
            if union["n_ids"] == 0:
                outside_sg_z = 0.0 if (baselines[start:, 1] > 0).all().item() else None
            outside_mean = (
                focus.shape[-1] * base[0]
                - union["n_ids"] * (union["mean_logit"] or 0.0)
            ) / outside_count
            outside_logp = (
                focus.shape[-1] * (base[0] - base[2])
                - union["n_ids"] * (union["mean_logp"] or 0.0)
            ) / outside_count
            groups["outside"] = {
                "n_ids": outside_count,
                "sg": outside_sg,
                "sg_z": outside_sg_z,
                "mean_logit": outside_mean,
                "mean_logp": outside_logp,
                "probability_mass": max(0.0, min(1.0, 1.0 - union["probability_mass"])),
            }
        else:
            groups["outside"] = {
                "n_ids": 0,
                "sg": None,
                "sg_z": None,
                "mean_logit": None,
                "mean_logp": None,
                "probability_mass": 0.0,
            }
        result[str(window)] = {
            "used_positions": used,
            "aggregation": "mean_of_per_position_statistics",
            "vocab_size": int(focus.shape[-1]),
            "vocab_mean_logit": float(base[0]),
            "vocab_std_logit": float(base[1]),
            "log_partition_mean": float(base[2]),
            "groups": groups,
        }
    return result


def read_lens(
    model,
    lens,
    tok,
    prompt,
    band,
    k,
    max_len,
    start,
    chunk,
    term_ids,
    *,
    collect_topk=True,
    focus_anchors=None,
    focus_prompt=None,
    focus_windows=DEFAULT_LAST_TOKEN_WINDOWS,
    focus_k=50,
    focus_code_end_k=None,
    focus_target_ids=None,
    focus_vocab_registry=None,
    focus_choice_ids=None,
    focus_mcp_k=500,
    ranking_mask=None,
    focus_personalized_k=500,
    focus_personalized_names=(),
    actual_tokenization=None,
):
    band = [l for l in band if l in lens.source_layers]
    if actual_tokenization is not None:
        model = _ActualTokenModel(model, prompt, actual_tokenization[0])
    top_ids, logp, model_top_ids, toks = ({}, {}, None, None)
    tid = torch.tensor(list(term_ids.values()), dtype=torch.long) if term_ids else None
    names = list(term_ids)
    if focus_code_end_k is not None and focus_code_end_k < 1:
        raise ValueError("focus_code_end_k must be >= 1")
    focus_windows = tuple(sorted({int(w) for w in focus_windows if int(w) >= 1}))
    focuses = None
    for i in range(0, len(band), chunk):
        part = band[i : i + chunk]
        lg, ml, ids = lens.apply(model, prompt, layers=part, max_seq_len=max_len)
        n = lg[part[0]].shape[0]
        keep = slice(min(start, max(n - 1, 0)), n)
        if focus_anchors and focuses is None and focus_windows:
            full_ids = ids[0].detach().cpu().tolist()
            focuses = {}
            anchor_meta = _resolve_char_anchors(
                tok,
                focus_prompt or prompt,
                focus_anchors,
                full_ids,
                actual_tokenization,
            )
            for name, meta in anchor_meta.items():
                anchor_k = (
                    focus_personalized_k
                    if name in focus_personalized_names
                    else focus_code_end_k
                    if name in {"code_end", "code_content_end", "code_whitespace_end"}
                    and focus_code_end_k is not None
                    else focus_mcp_k
                    if name.startswith("mcp_")
                    else focus_k
                )
                focus_end = min(max(int(meta["end"]), 0), n)
                if (name.startswith("mcp_") or name in focus_personalized_names) and (
                    not meta["boundary_exact"] or focus_end == 0
                ):
                    raise ValueError(
                        f"MCP anchor {name} is not an exact token boundary; refusing shifted readout"
                    )
                if name in {"code_end", "code_content_end", "code_whitespace_end"} and (
                    not meta["boundary_exact"] or focus_end == 0
                ):
                    raise ValueError(
                        f"{name} non risolvibile su un confine token verificato"
                    )
                max_w = min(max(focus_windows), focus_end)
                tail_start = focus_end - max_w
                tail_ids = [int(x) for x in full_ids[tail_start:focus_end]]
                focuses[name] = {
                    "top_k": int(anchor_k),
                    "anchor_end_exclusive": focus_end,
                    "boundary_exact": meta["boundary_exact"],
                    "prefix_token_count_requested": meta["requested_token_count"],
                    "anchor_char_end": meta.get("char_end"),
                    "requested_anchor_char_end": meta.get(
                        "requested_char_end", meta.get("char_end")
                    ),
                    "boundary_adjustment": meta.get("boundary_adjustment", "none"),
                    "boundary_method": meta.get("method"),
                    "personalized": name in focus_personalized_names,
                    "prompt_tokenization_lcp": meta.get("prompt_tokenization_lcp"),
                    "prompt_tokenization_length": meta.get(
                        "prompt_tokenization_length"
                    ),
                    "tail_positions": list(range(tail_start, focus_end)),
                    "tail_token_ids": tail_ids,
                    "lens": {},
                    "lens_targets": {},
                    "model": _aggregate_window_topk(
                        ml, focus_windows, anchor_k, focus_end
                    ),
                    "model_targets": _aggregate_window_targets(
                        ml, focus_windows, focus_target_ids, focus_end
                    ),
                    "model_choice_scores": _aggregate_choice_scores(
                        ml, focus_windows, focus_end, focus_choice_ids
                    ),
                    "lens_choice_scores": {},
                    "model_vocab_scores": _aggregate_window_vocab_scores(
                        ml, focus_windows, focus_end, focus_vocab_registry
                    ),
                    "lens_vocab_scores": {},
                }
        if toks is None and collect_topk:
            token_ids = ids[0, keep].detach().cpu()
            toks = tok.batch_decode([[int(t)] for t in token_ids])
            model_top_ids = ml[keep].topk(k, dim=-1).indices.detach().cpu()
        for l in part:
            rows_all = lg[l]
            rows = rows_all[keep]
            if collect_topk:
                top_ids[l] = ranked_indices(rows, k, ranking_mask).detach().cpu()
            if tid is not None:
                tid_dev = tid.to(rows.device)
                lp = rows[:, tid_dev] - rows.logsumexp(dim=-1, keepdim=True)
                mx = lp.max(dim=0).values
                logp[l] = {nm: float(mx[j]) for j, nm in enumerate(names)}
                del lp
            if focuses is not None:
                for focus in focuses.values():
                    focus["lens_choice_scores"][l] = _aggregate_choice_scores(
                        rows_all,
                        focus_windows,
                        focus["anchor_end_exclusive"],
                        focus_choice_ids,
                    )
                    focus["lens"][l] = _aggregate_window_topk(
                        rows_all,
                        focus_windows,
                        focus["top_k"],
                        focus["anchor_end_exclusive"],
                        ranking_mask=ranking_mask,
                    )
                    focus["lens_targets"][l] = _aggregate_window_targets(
                        rows_all,
                        focus_windows,
                        focus_target_ids,
                        focus["anchor_end_exclusive"],
                    )
                    if focus_vocab_registry is not None:
                        focus["lens_vocab_scores"][l] = _aggregate_window_vocab_scores(
                            rows_all,
                            focus_windows,
                            focus["anchor_end_exclusive"],
                            focus_vocab_registry,
                        )
            del rows
        del lg, ml
    return (
        top_ids,
        logp,
        model_top_ids,
        toks,
        _decode_anchor_readout(tok, focuses, focus_windows, focus_k),
    )


def _pack_local_indices(values, dtype):
    code = "H" if dtype == "uint16" else "I"
    a = array(code, values)
    if sys.byteorder != "little":
        a.byteswap()
    compressed = zlib.compress(a.tobytes(), level=9)
    return base64.b64encode(compressed).decode("ascii")


def pack_topk(tok, lens_top_ids, model_top_ids):
    layers = sorted(lens_top_ids)
    if not layers or model_top_ids is None:
        return {}
    lens_stack = torch.stack([lens_top_ids[l].long() for l in layers], dim=0)
    model_stack = model_top_ids.long()
    flat = torch.cat([lens_stack.reshape(-1), model_stack.reshape(-1)])
    vocab_ids, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    vocab_n = int(vocab_ids.numel())
    dtype = "uint16" if vocab_n <= 65536 else "uint32"
    n_lens = lens_stack.numel()
    inv = inverse.tolist()
    lens_data = _pack_local_indices(inv[:n_lens], dtype)
    model_data = _pack_local_indices(inv[n_lens:], dtype)
    vocab_ids_list = [int(x) for x in vocab_ids.tolist()]
    vocab_text = tok.batch_decode([[x] for x in vocab_ids_list])
    return {
        "topk_vocab_ids": vocab_ids_list,
        "topk_vocab": vocab_text,
        "lens_topk_packed": {
            "version": 1,
            "layers": layers,
            "shape": list(lens_stack.shape),
            "dtype": dtype,
            "endian": "little",
            "codec": "zlib+base64",
            "data": lens_data,
        },
        "model_topk_packed": {
            "version": 1,
            "shape": list(model_stack.shape),
            "dtype": dtype,
            "endian": "little",
            "codec": "zlib+base64",
            "data": model_data,
        },
    }


def single_ids(tok, terms):
    out = {}
    for t in terms:
        i = tok.encode(" " + t, add_special_tokens=False)
        if len(i) == 1:
            out[t] = i[0]
    return out


def _parse_windows(value):
    if isinstance(value, (tuple, list)):
        vals = [int(x) for x in value]
    else:
        vals = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    vals = sorted(set(vals))
    if not vals or vals[0] < 1:
        raise argparse.ArgumentTypeError(
            "last-token windows must be a comma-separated list of integers >= 1"
        )
    return tuple(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="security")
    ap.add_argument(
        "--condition",
        choices=CONDITIONS,
        default="carrier",
        help="carrier: fixed unrelated output; cot: original explicit reasoning; silent: original direct assessment without reasoning.",
    )
    ap.add_argument(
        "--carrier",
        choices=tuple(CARRIERS),
        default="sentence",
        help="Visible carrier used by --condition carrier. sentence, mcp, and sequence are fixed; mcp-verdict emits a constrained MCP-like safe/vulnerable verdict. No real tool is invoked.",
    )
    ap.add_argument(
        "--carrier-task",
        choices=tuple(CARRIER_TASKS),
        default="neutral",
        help="neutral only asks for a private program model; behavioral asks for a concrete problematic behavior/mechanism without safe/vulnerable wording; security is the explicit-security positive control; problem-or-math keeps the original problem-vs-math control; problem-focus contains only the strong concrete-problem reasoning instruction, with no math branch.",
    )
    ap.add_argument(
        "--no-chat-template", dest="chat_template", action="store_false", default=True
    )
    ap.add_argument(
        "--silent", action="store_true", help="Legacy alias for --condition silent."
    )
    ap.add_argument("--skip-lens", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--lens-file")
    ap.add_argument("--lens-kind", choices=("jlens", "jpp"), default="jlens")
    ap.add_argument("--lens-revision")
    ap.add_argument("--refresh-lens", action="store_true")
    ap.add_argument("--download-lens-only", action="store_true")
    ap.add_argument("--cache-list", action="store_true")
    ap.add_argument("--cache-remove", choices=tuple(MODELS))
    ap.add_argument(
        "--layers", help="all oppure intervalli/indici, ad esempio 40-58 o 8,16,24"
    )
    ap.add_argument("--max-seq-len", type=int)
    ap.add_argument("--gpt-oss-direct-final", action="store_true")
    ap.add_argument(
        "--stop-at-mcp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop on the first complete valid MCP object; automatic for Gemma4 and model configurations with stop_at_mcp_default.",
    )
    ap.add_argument("--check-anchors", action="store_true")
    ap.add_argument(
        "--code-position-anchors",
        action="store_true",
        help="Also read code_content_end and code_whitespace_end using code-end top-k and vocabulary metrics.",
    )
    ap.add_argument("--max-reasoning", type=int)
    ap.add_argument(
        "--reasoning-effort",
        choices=REASONING_CHOICES,
        default=None,
        help="Override del modello: none = nessun livello esplicito (per Qwen in carrier disabilita il thinking). Senza flag: default del modello in cot/silent, nessuno in carrier.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        help="Override the model generation batch size (>=1). J-Lens remains per-sample.",
    )
    ap.add_argument(
        "--last-token-windows",
        type=_parse_windows,
        default=DEFAULT_LAST_TOKEN_WINDOWS,
        help="Suffix windows for each focused semantic anchor, e.g. 1,2,4,8,16. Window 1 is the exact token immediately before the anchor boundary.",
    )
    ap.add_argument(
        "--last-token-top-k",
        type=int,
        default=50,
        help="Top-k vocabulary items stored for each anchor/window/layer.",
    )
    ap.add_argument(
        "--code-end-top-k",
        type=int,
        default=None,
        help="Override Top-k only at the code_end anchor, for both J-Lens and ordinary-model readouts. Other anchors retain --last-token-top-k; full-sequence --top-k is unchanged.",
    )
    ap.add_argument(
        "--no-last-token-readout",
        action="store_true",
        help="Disable semantic-anchor readouts; full packed Top-K is still stored.",
    )
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument(
        "--dataset-kind",
        choices=("auto", "secllmholmes", "controlled"),
        default="auto",
        help="Dataset loader. auto detects the controlled benchmark from ground_truth.jsonl, otherwise falls back to SecLLMHolmes.",
    )
    ap.add_argument("--out")
    ap.add_argument(
        "--sg-vocab",
        help="Compiled full vocabulary folder or token_groups.json; saves exact SG per anchor/layer",
    )
    ap.add_argument("--mcp-top-k", type=int, default=500)
    ap.add_argument(
        "--personalized-anchors",
        action="store_true",
        help="Add model-specific readout positions; original anchors remain unchanged.",
    )
    ap.add_argument(
        "--personalized-top-k",
        type=int,
        default=500,
        help="Top-k at added positions; full-vocabulary masses and SG stay exact.",
    )
    a = ap.parse_args()
    if a.cache_list or a.cache_remove:
        cache_action(a.cache_remove)
        return
    if a.mcp_top_k < 1:
        ap.error("--mcp-top-k must be positive")
    if a.sg_vocab and (a.skip_lens or a.no_last_token_readout):
        ap.error("--sg-vocab requires lens and semantic-anchor readouts")
    if a.silent:
        a.condition = "silent"
    if a.list:
        for k, v in MODELS.items():
            print(f"{k}\n  " + "\n  ".join((f"{n}: {x}" for n, x in v.items())))
        return
    if a.model not in MODELS:
        raise SystemExit(f"modelli: {', '.join(MODELS)}")
    if a.last_token_top_k < 1:
        raise SystemExit("--last-token-top-k deve essere >= 1")
    if (
        a.code_end_top_k is None
        and a.condition == "carrier"
        and a.carrier == "mcp-verdict"
    ):
        a.code_end_top_k = 500
    if a.code_end_top_k is not None:
        if a.code_end_top_k < 1:
            raise SystemExit("--code-end-top-k deve essere >= 1")
        if a.condition != "carrier":
            raise SystemExit("--code-end-top-k richiede --condition carrier")
        if a.no_last_token_readout:
            raise SystemExit("--code-end-top-k richiede gli anchor readout attivi")
    spec = MODELS[a.model].copy()
    spec["model_name"] = a.model
    try:
        validate_model_request(spec, a.condition, a.reasoning_effort)
    except ValueError as ex:
        ap.error(str(ex))
    mcp_mode = a.condition == "carrier" and a.carrier == "mcp-verdict"
    if a.stop_at_mcp is True and not mcp_mode:
        ap.error("--stop-at-mcp requires --condition carrier --carrier mcp-verdict")
    stop_at_mcp = mcp_mode and (
        spec.get("stop_at_mcp_default", a.model == "gemma4-31b")
        if a.stop_at_mcp is None
        else a.stop_at_mcp
    )
    spec["code_position_anchors"] = a.code_position_anchors
    spec["personalized_anchors"] = a.personalized_anchors
    if a.personalized_top_k < 1:
        ap.error("--personalized-top-k must be positive")
    if a.personalized_anchors and (
        not mcp_mode or a.skip_lens or a.no_last_token_readout
    ):
        ap.error(
            "--personalized-anchors requires carrier mcp-verdict and lens anchor readouts"
        )
    if a.code_position_anchors and (
        a.condition != "carrier"
        or a.carrier != "mcp-verdict"
        or a.no_last_token_readout
        or a.skip_lens
    ):
        ap.error(
            "--code-position-anchors requires carrier mcp-verdict with lens anchor readouts"
        )
    if a.max_seq_len is not None:
        if a.max_seq_len < 1:
            ap.error("--max-seq-len deve essere positivo")
        spec["max_seq_len"] = a.max_seq_len
    if a.gpt_oss_direct_final:
        if a.model != "gpt-oss-20b" or a.condition != "carrier" or not a.chat_template:
            ap.error(
                "--gpt-oss-direct-final richiede GPT-OSS in carrier con chat template"
            )
        spec["gpt_oss_direct_final"] = True
    if spec.get("base_model") and a.chat_template:
        print(
            "Checkpoint base: uso del prompt testuale senza chat template", flush=True
        )
        a.chat_template = False
    if a.lens_kind == "jpp" and not a.lens_file and not spec.get("jpp_file"):
        ap.error(f"Nessuna J++ verificata per {spec['hf_id']}; usa --lens-kind jlens")
    spec["effort_explicit_none"] = a.reasoning_effort == "none"
    if a.reasoning_effort is None:
        spec["reasoning_effort"] = (
            None if a.condition == "carrier" else spec.get("reasoning_effort")
        )
    elif a.reasoning_effort == "none":
        spec["reasoning_effort"] = None
    else:
        if a.model not in SUPPORTED_REASONING:
            raise SystemExit(f"{a.model} non supporta --reasoning-effort")
        if a.reasoning_effort not in SUPPORTED_REASONING[a.model]:
            allowed = ", ".join(sorted(SUPPORTED_REASONING[a.model]))
            raise SystemExit(f"{a.model}: livelli consentiti: {allowed}, none")
        spec["reasoning_effort"] = a.reasoning_effort
    if not a.chat_template and a.reasoning_effort not in (None, "none"):
        raise SystemExit("--reasoning-effort richiede il chat template")
    p = PROMPTS[a.prompt]
    lens_file = a.lens_file or (
        spec["jpp_file"] if a.lens_kind == "jpp" else spec["lens_file"]
    )
    if a.download_lens_only:
        load_run_lens(spec, lens_file, a.lens_revision, a.refresh_lens, a.lens_kind)
        return
    band = []
    max_reasoning = (
        spec["max_reasoning"] if a.max_reasoning is None else a.max_reasoning
    )
    if max_reasoning < 1:
        raise SystemExit("--max-reasoning deve essere >= 1")
    if a.top_k < 1 or a.limit < 0:
        raise SystemExit("--top-k deve essere >= 1 e --limit >= 0")
    batch_size = spec["batch_size"] if a.batch_size is None else a.batch_size
    if batch_size < 1:
        raise SystemExit("--batch-size deve essere >= 1")
    carrier_text = CARRIERS[a.carrier].get("text")
    carrier_task = CARRIER_TASKS[a.carrier_task]
    out = Path(a.out or f"results/{a.model}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(str(Path(a.dataset).expanduser()), kind=a.dataset_kind)[
        : a.limit
    ]
    tok = transformers.AutoTokenizer.from_pretrained(spec["hf_id"])
    if getattr(tok, "bos_token_id", None) is not None and hasattr(tok, "add_bos_token"):
        tok.add_bos_token = True
    if a.check_anchors:
        if a.condition != "carrier":
            ap.error("--check-anchors richiede --condition carrier")
        check_anchor_boundaries(
            tok, spec, cases, a.chat_template, a.carrier, carrier_text, carrier_task
        )
        return
    choice_ids = (
        _choice_ids(tok)
        if a.condition == "carrier" and a.carrier == "mcp-verdict"
        else None
    )
    sg_registry, sg_info = (
        _load_sg_vocab(
            Path(__file__).parent / "vocab" / a.model
            if a.sg_vocab == "auto"
            else a.sg_vocab,
            tok,
            spec["hf_id"],
        )
        if a.sg_vocab
        else (None, None)
    )
    hf = load_model(spec)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise RuntimeError("Tokenizer senza pad_token_id e senza eos_token_id")
        tok.pad_token = tok.eos_token
    use_verdict_terms = a.condition in ("cot", "silent")
    terms = (
        {"bad": list(p["bad"]), "good": list(p["good"])}
        if use_verdict_terms
        else {"bad": [], "good": []}
    )
    model = lens = term_ids = null = None
    lens_info = None
    ranking_mask = None
    if not a.skip_lens:
        model = jlens.from_hf(hf, tok)
        lens, lens_info = load_run_lens(
            spec, lens_file, a.lens_revision, a.refresh_lens, a.lens_kind
        )
        if lens.d_model != model.d_model:
            raise ValueError("La dimensione della lens non coincide con il modello")
        band = selected_layers(a.layers, spec, lens)
        if not band or any(layer < 0 or layer >= model.n_layers for layer in band):
            raise ValueError("Nessun layer valido per il modello e la lens")
        if a.lens_kind == "jpp":
            ranking_mask = semantic_token_mask(tok, model._lm_head.weight.shape[0])
            if int(ranking_mask.sum()) < max(
                a.top_k, a.mcp_top_k, a.code_end_top_k or 0, a.last_token_top_k
            ):
                raise ValueError("Troppi pochi token dopo il filtro semantico")
    term_ids = single_ids(tok, p["bad"] + p["good"]) if use_verdict_terms else {}
    meta = {
        "_meta": {
            "model": a.model,
            "hf_id": spec["hf_id"],
            "prompt": a.prompt,
            "condition": a.condition,
            "chat_template": a.chat_template,
            "silent": a.condition == "silent",
            "carrier": a.carrier if a.condition == "carrier" else None,
            "carrier_text": carrier_text if a.condition == "carrier" else None,
            "carrier_description": CARRIERS[a.carrier]["description"]
            if a.condition == "carrier"
            else None,
            "carrier_task": a.carrier_task if a.condition == "carrier" else None,
            "carrier_task_text": carrier_task if a.condition == "carrier" else None,
            "carrier_prompt_version": "v5-controlled-target-scores"
            if a.condition == "carrier"
            else None,
            "band": [band[0], band[-1]] if band else None,
            "readout_layers": band,
            "lens_kind": a.lens_kind,
            "lens_checkpoint": lens_info,
            "gpt_oss_direct_final": a.gpt_oss_direct_final,
            "mcp_generation_stop": {
                "enabled": stop_at_mcp,
                "requested": a.stop_at_mcp,
                "policy": "first_complete_valid_mcp_prefix-v1"
                if stop_at_mcp
                else "model_eos_or_token_limit",
            },
            "base_model": bool(spec.get("base_model")),
            "no_explicit_reasoning": bool(spec.get("no_explicit_reasoning")),
            "top_k": a.top_k,
            "topk_storage": "local-vocab-uint-zlib-base64-v1",
            "anchor_storage": "local-vocab-uint-float-zlib-base64-v1",
            "personalized_anchors": a.personalized_anchors,
            "personalized_anchor_definitions": personalized_definitions(spec),
            "personalized_top_k": a.personalized_top_k,
            "readout_tokenization": "actual_prompt_ids_plus_actual_generated_ids_for_mcp",
            "readout_schema_version": "mcp-personalized-ab-v4"
            if a.personalized_anchors
            else "mcp-code-positions-ab-v3"
            if a.code_position_anchors
            else "mcp-three-anchors-ab-v2",
            "code_position_anchors": a.code_position_anchors,
            "code_position_definitions": {
                "code_content_end": "After the last non-whitespace character of the code",
                "code_end": "After the closing </code> marker",
                "code_whitespace_end": "After whitespace following </code>, before instructions",
            },
            "code_end_boundary_policy": "exact_or_token_end_with_whitespace_suffix_only",
            "anchor_readout": not a.no_last_token_readout,
            "anchor_names": (
                ["code_end"]
                + (
                    ["code_content_end", "code_whitespace_end"]
                    if a.code_position_anchors
                    else []
                )
                + ["mcp_record", "mcp_pre_verdict"]
                + list(personalized_definitions(spec))
                if a.condition == "carrier" and a.carrier == "mcp-verdict"
                else [
                    "code_end",
                    "instruction_end",
                    "assistant_prethink (when present)",
                    "assistant_start",
                ]
                if a.condition == "carrier"
                else ["assistant_start"]
            ),
            "last_token_windows": list(a.last_token_windows),
            "last_token_top_k": a.last_token_top_k,
            "code_end_top_k": a.code_end_top_k,
            "mcp_top_k": a.mcp_top_k,
            "choice_token_ids": choice_ids,
            "last_token_aggregation": "max_logprob_over_last_n_positions_ending_at_anchor",
            "quantize": spec["quantize"],
            "reasoning_effort_requested": a.reasoning_effort,
            "reasoning_effort_effective": spec.get("reasoning_effort"),
            "reasoning_note": (
                "GPT-OSS direct-final prefill; experimental generation condition"
                if a.gpt_oss_direct_final
                else "GPT-OSS cannot disable thinking; None uses model template default"
                if a.model == "gpt-oss-20b" and spec.get("reasoning_effort") is None
                else None
            ),
            "chat_template_kwargs": reasoning_template_kwargs(
                spec, condition=a.condition
            )
            if a.chat_template
            else {},
            "lens_file": None if a.skip_lens else lens_file,
            "max_reasoning": max_reasoning,
            "max_seq_len": spec["max_seq_len"],
            "batch_size": batch_size,
            "limit": a.limit,
            "dataset": a.dataset,
            "dataset_kind": a.dataset_kind,
            "terms": terms,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vocab_scores": sg_info,
        }
    }
    n_ok = n_scored = n_cut = n_carrier_ok = 0
    with out.open("w") as fh:
        fh.write(json.dumps(meta) + "\n")
        for batch_start, batch_cases in batches(cases, batch_size):
            prefixes = []
            parts_list = []
            focus_anchors_list = []
            for c in batch_cases:
                if a.condition == "carrier":
                    prefix, focus_anchors = build_carrier_prompt(
                        c.code,
                        a.chat_template,
                        tok,
                        spec,
                        a.carrier,
                        carrier_text,
                        carrier_task,
                    )
                    parts = None
                else:
                    prefix, parts = build_prompts(
                        p, c.code, a.chat_template, a.condition == "silent", tok, spec
                    )
                    focus_anchors = {"assistant_start": len(prefix)}
                prefixes.append(prefix)
                parts_list.append(parts)
                focus_anchors_list.append(focus_anchors)
            if a.condition == "cot":
                reasoning_out = generate_batch(hf, tok, prefixes, max_reasoning)
                reasonings = [text for text, _ in reasoning_out]
                reasoning_cuts = [cut for _, cut in reasoning_out]
                output_prompts = [
                    verdict_prompt(
                        p, prefix, parts, reasoning, a.chat_template, tok, spec
                    )
                    for prefix, parts, reasoning in zip(
                        prefixes, parts_list, reasonings, strict=False
                    )
                ]
            else:
                reasonings = [""] * len(batch_cases)
                reasoning_cuts = [False] * len(batch_cases)
                output_prompts = prefixes
            if a.condition != "carrier":
                focus_anchors_list = [
                    {"assistant_start": len(vp)} for vp in output_prompts
                ]
            if a.condition == "carrier":
                max_out = 96 if a.carrier in ("mcp", "mcp-verdict") else 48
            else:
                max_out = 8
            response_stop_info = []
            response_out = generate_batch(
                hf,
                tok,
                output_prompts,
                max_out,
                stop_at_mcp=stop_at_mcp,
                generation_info=response_stop_info,
            )
            responses = [text for text, _ in response_out]
            response_cuts = [cut for _, cut in response_out]
            for j, (
                c,
                _prefix,
                _parts,
                reasoning,
                reasoning_cut,
                vp,
                resp,
                response_cut,
                focus_anchors,
            ) in enumerate(
                zip(
                    batch_cases,
                    prefixes,
                    parts_list,
                    reasonings,
                    reasoning_cuts,
                    output_prompts,
                    responses,
                    response_cuts,
                    focus_anchors_list,
                    strict=False,
                )
            ):
                cut = bool(reasoning_cut or response_cut)
                n_cut += int(cut)
                if a.condition == "carrier":
                    sv = {}
                    ss = {}
                    prob = None
                    summary = None
                    if a.carrier == "mcp-verdict":
                        pred, ok_carrier = parse_mcp_verdict(resp)
                        correct = pred == c.label if pred is not None else None
                        if pred is not None:
                            n_scored += 1
                            n_ok += pred == c.label
                    else:
                        pred = None
                        ok_carrier = carrier_quality(resp, carrier_text)
                        correct = None
                    n_carrier_ok += int(ok_carrier)
                else:
                    all_scores = score_terms(hf, tok, vp, p["bad"] + p["good"])
                    sv = {t: all_scores[t] for t in p["bad"] if t in all_scores}
                    ss = {t: all_scores[t] for t in p["good"] if t in all_scores}
                    prob = predict_label(sv, ss)
                    summary = last_summary(reasoning)
                    pred = parse_pred(resp, p) or prob or parse_pred(summary, p)
                    ok_carrier = None
                    correct = pred == c.label if pred is not None else None
                    if pred is not None:
                        n_scored += 1
                        n_ok += pred == c.label
                rec = {
                    "case_id": c.case_id,
                    "cwe": c.cwe,
                    "pair_id": c.pair_id,
                    "quartet_id": c.meta.get("quartet_id"),
                    "cwes": c.meta.get("cwes", [c.cwe]),
                    "target_labels": c.meta.get("target_labels", {c.cwe: c.label}),
                    "source_code_sha256": hashlib.sha256(
                        c.code.encode("utf-8")
                    ).hexdigest(),
                    "ground_truth": c.label,
                    "gt_vuln": c.gt_vuln,
                    "gt_safe": c.gt_safe,
                    "condition": a.condition,
                    "carrier": a.carrier if a.condition == "carrier" else None,
                    "carrier_text": carrier_text if a.condition == "carrier" else None,
                    "carrier_task": a.carrier_task
                    if a.condition == "carrier"
                    else None,
                    "carrier_ok": ok_carrier,
                    "reasoning": reasoning,
                    "summary": summary,
                    "truncated": cut,
                    "response": resp,
                    "pred": pred,
                    "pred_prob": prob,
                    "correct": correct,
                    "scores_bad": sv,
                    "scores_good": ss,
                    "cwe_name": c.meta.get("cwe_name"),
                    "language": c.meta.get("language"),
                    "mechanism": c.meta.get("mechanism"),
                    "canonical_targets": c.meta.get("canonical_targets", []),
                }
                if stop_at_mcp:
                    rec["generation_stop"] = {
                        k: v
                        for k, v in response_stop_info[j].items()
                        if k not in {"response", "prompt_input_ids", "generated_ids"}
                    }
                if not a.skip_lens:
                    canonical_targets = list(c.meta.get("canonical_targets", []))
                    target_ids = single_ids(tok, canonical_targets)
                    rec["canonical_target_token_ids"] = target_ids
                    rec["canonical_targets_dropped_multitoken"] = [
                        t for t in canonical_targets if t not in target_ids
                    ]
                    full = vp + resp
                    actual_tokenization = None
                    if a.condition == "carrier" and a.carrier == "mcp-verdict":
                        info = response_stop_info[j]
                        full, actual_tokenization = actual_generation_tokenization(
                            tok, vp, info["generated_ids"], info["prompt_input_ids"]
                        )
                        raw_response = full[len(vp) :]
                        completed = _completed_mcp_prefix(raw_response)
                        anchor_response = (
                            completed[0] if completed is not None else raw_response
                        )
                        output_anchors, anchor_status = _mcp_output_anchors(
                            vp, anchor_response
                        )
                        output_anchors.update(
                            personalized_output_anchors(vp, anchor_response, spec)
                        )
                        rec["readout_response_text"] = raw_response
                        rec["input_token_ids"] = actual_tokenization[0]
                        rec["output_start_token"] = len(info["prompt_input_ids"])
                        focus_anchors = {
                            name: end
                            for name, end in focus_anchors.items()
                            if name
                            in {"code_end", "code_content_end", "code_whitespace_end"}
                            or name in personalized_definitions(spec)
                        } | output_anchors
                        rec["mcp_anchor_status"] = anchor_status
                        rec["output_start_char"] = len(vp)
                    n = (
                        len(actual_tokenization[0])
                        if actual_tokenization is not None
                        else len(tok.encode(full))
                    )
                    if n <= spec["max_seq_len"]:
                        start = 0
                        if term_ids and null is None:
                            _, null, _, _, _ = read_lens(
                                model,
                                lens,
                                tok,
                                p["tail"],
                                band,
                                a.top_k,
                                64,
                                0,
                                spec["layer_chunk"],
                                term_ids,
                                collect_topk=False,
                            )
                        tops, logp, mtop, toks, tail_readout = read_lens(
                            model,
                            lens,
                            tok,
                            full,
                            band,
                            a.top_k,
                            n,
                            start,
                            spec["layer_chunk"],
                            term_ids,
                            focus_anchors=None
                            if a.no_last_token_readout
                            else focus_anchors,
                            focus_prompt=full,
                            focus_windows=a.last_token_windows,
                            focus_k=a.last_token_top_k,
                            focus_code_end_k=a.code_end_top_k,
                            focus_target_ids=target_ids,
                            focus_vocab_registry=sg_registry,
                            focus_choice_ids=choice_ids,
                            focus_mcp_k=a.mcp_top_k,
                            ranking_mask=ranking_mask,
                            focus_personalized_k=a.personalized_top_k,
                            focus_personalized_names=tuple(
                                personalized_definitions(spec)
                            ),
                            actual_tokenization=actual_tokenization,
                        )
                        rec["tokens"] = toks
                        rec.update(pack_topk(tok, tops, mtop))
                        if tail_readout is not None:
                            rec["anchor_readout"] = tail_readout
                            assistant = tail_readout.get("anchors", {}).get(
                                "assistant_start"
                            )
                            if assistant is not None:
                                rec["prompt_tail_readout"] = {
                                    "aggregation": tail_readout["aggregation"],
                                    "prompt_end_exclusive": assistant[
                                        "anchor_end_exclusive"
                                    ],
                                    "prefix_boundary_exact": assistant[
                                        "boundary_exact"
                                    ],
                                    "prefix_token_count_requested": assistant[
                                        "prefix_token_count_requested"
                                    ],
                                    "windows": tail_readout["windows"],
                                    "top_k": tail_readout["top_k"],
                                    "tail_positions": assistant["tail_positions"],
                                    "tail_token_ids": assistant["tail_token_ids"],
                                    "tail_tokens": assistant["tail_tokens"],
                                    "lens": assistant["lens"],
                                    "model": assistant["model"],
                                }
                        if term_ids:
                            rec["lens_logp"] = logp
                            rec["lens_logp_net"] = {
                                l: {t: logp[l][t] - null[l][t] for t in logp[l]}
                                for l in logp
                            }
                    else:
                        rec["lens_skipped"] = n
                fh.write(json.dumps(pack_record(rec), separators=(",", ":")) + "\n")
                global_i = batch_start + j + 1
                if a.condition == "carrier":
                    if a.carrier == "mcp-verdict":
                        status = f"mcp_ok={ok_carrier} pred={pred} true={c.label}"
                    else:
                        status = f"carrier_ok={ok_carrier}"
                else:
                    status = f"pred={pred} true={c.label}"
                print(f"{global_i}/{len(cases)} {status} cut={cut} batch={batch_size}")
    if a.condition == "carrier":
        if a.carrier == "mcp-verdict":
            valid_rate = n_carrier_ok / len(cases) if cases else 0.0
            acc = n_ok / n_scored if n_scored else 0.0
            print(
                f"mcp valid {n_carrier_ok}/{len(cases)} = {valid_rate:.3f}  accuracy {n_ok}/{n_scored} = {acc:.3f}  troncati {n_cut}"
            )
        else:
            print(
                f"carrier exact {n_carrier_ok}/{len(cases)} = {n_carrier_ok / len(cases):.3f}  troncati {n_cut}"
                if cases
                else "carrier exact 0/0"
            )
    else:
        print(
            f"accuracy {n_ok}/{n_scored} = {n_ok / n_scored:.3f}  troncati {n_cut}"
            if n_scored
            else f"accuracy 0/0 = 0.000  troncati {n_cut}"
        )
    print(f"scritto a {out}")


if __name__ == "__main__":
    main()
