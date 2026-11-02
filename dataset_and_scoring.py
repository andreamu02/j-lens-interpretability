from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Case:
    source: str
    case_id: str
    cwe: str
    code: str
    label: str
    pair_id: str
    gt_vuln: str | None = None
    gt_safe: str | None = None
    meta: dict = field(default_factory=dict)


def load_secllmholmes(dataset_dir: str) -> list[Case]:
    cases = []
    root = Path(dataset_dir).expanduser()
    gt_root = root.parent / "ground-truth"

    for cwe_dir in sorted(root.glob("CWE-*")):
        cwe = cwe_dir.name
        for f in sorted(cwe_dir.iterdir()):
            if f.suffix not in (".c", ".py"):
                continue

            stem = f.stem
            num = stem[2:] if stem.startswith("p_") else stem
            gv = gt_root / cwe / f"{num}.txt"
            gs = gt_root / cwe / f"p_{num}.txt"

            cases.append(
                Case(
                    source="secllmholmes",
                    case_id=f"secllmholmes_{cwe}_{stem}",
                    cwe=cwe,
                    code=f.read_text(errors="ignore"),
                    label="safe" if stem.startswith("p_") else "vulnerable",
                    pair_id=f"secllmholmes_{cwe}_{num}",
                    gt_vuln=(
                        gv.read_text(errors="ignore").strip() if gv.is_file() else None
                    ),
                    gt_safe=(
                        gs.read_text(errors="ignore").strip() if gs.is_file() else None
                    ),
                    meta={"path": str(f)},
                )
            )

    return cases


def _controlled_layout(dataset_dir: str) -> tuple[Path, Path]:
    root = Path(dataset_dir).expanduser()

    if (root / "ground_truth.jsonl").is_file() and (root / "dataset").is_dir():
        return root, root / "dataset"

    if root.name == "dataset" and (root.parent / "ground_truth.jsonl").is_file():
        return root.parent, root

    raise FileNotFoundError(
        "Controlled dataset not recognized. Pass either the bundle directory "
        "containing ground_truth.jsonl + dataset/, or its dataset/ subdirectory."
    )


def load_controlled(dataset_dir: str) -> list[Case]:
    bundle_root, _ = _controlled_layout(dataset_dir)
    gt_file = bundle_root / "ground_truth.jsonl"

    rows = []
    with gt_file.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "file" not in row or "ground_truth" not in row:
                raise ValueError(
                    f"{gt_file}:{line_no}: missing required file/ground_truth field"
                )
            rows.append(row)

    # Pair metadata is useful in every record and lets us sanity-check the frozen GT.
    by_pair: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["ground_truth"]] = row
        quartets = {}
    for row in rows:
        if len(row.get("cwes", [])) == 2:
            key = row.get("quartet_id", row["pair_id"])
            quartets.setdefault(key, []).append(row)

    expected_states = {
        ("safe", "safe"),
        ("safe", "vulnerable"),
        ("vulnerable", "safe"),
        ("vulnerable", "vulnerable"),
    }

    for key, members in quartets.items():
        cwes = tuple(sorted(members[0]["cwes"]))
        if len(set(cwes)) != 2 or len(members) != 4:
            raise ValueError(f"{key}: expected two distinct CWEs and four cases")

        states = set()
        for row in members:
            labels = row.get("target_labels", {})
            if tuple(sorted(row["cwes"])) != cwes or set(labels) != set(cwes):
                raise ValueError(f"{key}: inconsistent CWE annotations")
            state = tuple(labels[cwe] for cwe in cwes)
            if any(label not in {"safe", "vulnerable"} for label in state):
                raise ValueError(f"{key}: invalid target labels")
            expected_label = "vulnerable" if "vulnerable" in state else "safe"
            if row["ground_truth"] != expected_label:
                raise ValueError(f"{key}: inconsistent global label")
            states.add(state)

        if states != expected_states:
            raise ValueError(f"{key}: incomplete or duplicate vulnerability states")

    cases = []
    seen_ids = set()

    for row in rows:
        case_id = row["case_id"]
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id in controlled ground truth: {case_id}")
        seen_ids.add(case_id)

        label = row["ground_truth"]
        if label not in ("vulnerable", "safe"):
            raise ValueError(f"{case_id}: unsupported ground_truth={label!r}")

        pair = by_pair.get(row["pair_id"], {})
        if set(pair) != {"vulnerable", "safe"}:
            raise ValueError(
                f"{case_id}: pair {row['pair_id']} must contain one vulnerable "
                "and one safe record"
            )

        source_path = bundle_root / row["file"]
        if not source_path.is_file():
            raise FileNotFoundError(f"{case_id}: source file not found: {source_path}")

        cwe_name = row.get("cwe_name")
        mechanism = row.get("mechanism")
        vuln_text = (
            f"{cwe_name}: {mechanism}"
            if cwe_name and mechanism
            else mechanism or cwe_name
        )
        safe_text = (
            f"Patched counterpart for {cwe_name}; target mechanism mitigated: {mechanism}"
            if cwe_name and mechanism
            else mechanism or cwe_name
        )

        meta = dict(row)
        meta["path"] = str(source_path)
        meta["bundle_root"] = str(bundle_root)

        cases.append(
            Case(
                source="jlens-controlled-100",
                case_id=case_id,
                cwe=row["cwe"],
                code=source_path.read_text(errors="ignore"),
                label=label,
                pair_id=row["pair_id"],
                gt_vuln=vuln_text,
                gt_safe=safe_text,
                meta=meta,
            )
        )

    return cases


def load_cases(dataset_dir: str, kind: str = "auto") -> list[Case]:
    kind = kind.lower()

    if kind == "secllmholmes":
        return load_secllmholmes(dataset_dir)

    if kind == "controlled":
        return load_controlled(dataset_dir)

    if kind != "auto":
        raise ValueError("kind must be one of: auto, secllmholmes, controlled")

    root = Path(dataset_dir).expanduser()
    controlled = (root / "ground_truth.jsonl").is_file() or (
        root.name == "dataset" and (root.parent / "ground_truth.jsonl").is_file()
    )
    return (
        load_controlled(dataset_dir) if controlled else load_secllmholmes(dataset_dir)
    )


def usable_terms(tok, terms: list[str]) -> tuple[list[str], list[str]]:
    keep, dropped = [], []
    for t in terms:
        ids = tok.encode(" " + t, add_special_tokens=False)
        (keep if len(ids) == 1 else dropped).append(t)
    return keep, dropped


def check_terms(tok, bad: list[str], good: list[str]) -> dict:
    kb, db = usable_terms(tok, bad)
    kg, dg = usable_terms(tok, good)
    n = min(len(kb), len(kg))
    return {
        "bad": kb[:n],
        "good": kg[:n],
        "dropped_bad": db,
        "dropped_good": dg,
        "trimmed_bad": kb[n:],
        "trimmed_good": kg[n:],
        "n": n,
    }


def score_terms(hf_model, tok, prompt: str, terms: list[str]) -> dict:
    inputs = tok(prompt, return_tensors="pt").to(hf_model.device)
    with torch.no_grad():
        logits = hf_model(**inputs).logits[0, -1].float()
    logprobs = torch.log_softmax(logits, dim=-1)

    out = {}
    for t in terms:
        ids = tok.encode(" " + t, add_special_tokens=False)
        if len(ids) == 1:
            out[t] = logprobs[ids[0]].item()
    return out


def predict_label(bad: dict, good: dict) -> str | None:
    if not bad or not good:
        return None
    return "vulnerable" if max(bad.values()) > max(good.values()) else "safe"
