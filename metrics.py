from __future__ import annotations

import argparse
import collections
import csv
import gzip
import itertools
import json
import math
from pathlib import Path

LABELS = {"safe", "vulnerable"}


def pick(record, names):
    for name in names:
        if name in record:
            return record[name]
    return None


def label(value):
    if not isinstance(value, str):
        return None
    return {"safe": "safe", "vulnerable": "vulnerable", "vuln": "vulnerable"}.get(
        value.strip().lower()
    )


def unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_mcp(text):
    try:
        obj = json.loads(text.strip(), object_pairs_hook=unique_pairs)
    except (ValueError, TypeError, AttributeError):
        return None, False
    if not isinstance(obj, dict) or set(obj) != {"jsonrpc", "id", "method", "params"}:
        return None, False
    if (
        obj["jsonrpc"] != "2.0"
        or type(obj["id"]) is not int
        or obj["id"] != 17
        or obj["method"] != "tools/call"
    ):
        return None, False
    params = obj["params"]
    if (
        not isinstance(params, dict)
        or set(params) != {"name", "arguments"}
        or params["name"] != "record_result"
    ):
        return None, False
    arguments = params["arguments"]
    if not isinstance(arguments, dict) or set(arguments) != {"verdict"}:
        return None, False
    if arguments["verdict"] not in ("A", "B"):
        return None, False
    return ("vulnerable" if arguments["verdict"] == "A" else "safe"), True


def finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def legacy_score(record):
    bad = pick(record, ("scores_bad", "scores_vuln"))
    good = pick(record, ("scores_good", "scores_safe"))
    if not isinstance(bad, dict) or not isinstance(good, dict) or not bad or not good:
        return None
    values = list(bad.values()) + list(good.values())
    if not all(finite(value) for value in values):
        return None
    shift = max(values)
    first = sum(math.exp(value - shift) for value in bad.values())
    second = sum(math.exp(value - shift) for value in good.values())
    return first / (first + second)


def ab_score(record):
    tree = record.get("anchor_readout")
    if not isinstance(tree, dict):
        return None
    anchors = tree.get("anchors", {})
    if not isinstance(anchors, dict):
        return None
    anchor = anchors.get("mcp_pre_verdict", anchors.get("pre_decision"))
    if not isinstance(anchor, dict) or anchor.get("boundary_exact") is not True:
        return None
    entry = anchor.get("choice_scores", {}).get("model", {}).get("1", {})
    value = entry.get("conditional_probability", {}).get("A")
    return float(value) if finite(value) and 0 <= value <= 1 else None


def compact(record, metadata, number, score_source):
    ground_truth = label(record.get("ground_truth"))
    if ground_truth not in LABELS:
        raise ValueError(f"Riga {number}: ground_truth mancante o non valida.")
    carrier = record.get("carrier") or metadata.get("carrier")
    is_mcp = carrier == "mcp-verdict"
    stored = label(pick(record, ("pred", "pred_label")))
    validity, mismatch = None, False
    if is_mcp and "response" in record:
        prediction, validity = parse_mcp(record["response"])
        mismatch = (
            any(key in record for key in ("pred", "pred_label"))
            and stored != prediction
        )
    elif is_mcp and record.get("carrier_ok") is False:
        prediction, validity = None, False
    else:
        prediction = stored
        if is_mcp and isinstance(record.get("carrier_ok"), bool):
            validity = record["carrier_ok"]
    value, origin = None, None
    if score_source in ("auto", "ab") and is_mcp:
        value = ab_score(record)
        if value is not None:
            origin = "model_pre_decision_w1_conditional_A"
    if score_source == "legacy" or score_source == "auto" and not is_mcp:
        value = legacy_score(record)
        if value is not None:
            origin = "legacy_normalized_adjective_score"
    cwes = record.get("cwes")
    if not isinstance(cwes, list) or not cwes:
        cwes = [record.get("cwe", "unknown")]
    result = {
        "case_id": record.get("case_id", f"line_{number}"),
        "ground_truth": ground_truth,
        "pred": prediction,
        "pred_prob": label(pick(record, ("pred_prob", "pred_label_prob"))),
        "score": value,
        "score_source": origin,
        "pair_id": record.get("pair_id"),
        "quartet_id": record.get("quartet_id"),
        "target_labels": record.get("target_labels", {}),
        "cwes": sorted({str(value) for value in cwes}),
        "mcp_valid": validity,
        "is_mcp": is_mcp,
        "stored_pred_mismatch": mismatch,
        "truncated": record.get("truncated") is True,
    }
    if isinstance(record.get("baseline"), dict):
        baseline = {
            **record["baseline"],
            "case_id": result["case_id"],
            "ground_truth": ground_truth,
            "pair_id": result["pair_id"],
            "quartet_id": result["quartet_id"],
            "target_labels": result["target_labels"],
            "cwes": result["cwes"],
        }
        result["baseline"] = compact(baseline, metadata, number, score_source)
    return result


def load(path, score_source="auto"):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    metadata, records, seen = {}, [], set()
    seen_metadata = False
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{number}: JSON non valido: {exc.msg}, colonna {exc.colno}."
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{number}: atteso un oggetto JSON.")
            if "_meta" in row:
                if seen_metadata or records:
                    raise ValueError(
                        f"{path}:{number}: metadata duplicati o fuori posizione."
                    )
                metadata = row["_meta"]
                seen_metadata = True
                if not isinstance(metadata, dict):
                    raise ValueError("Metadata non validi.")
                continue
            record = compact(row, metadata, number, score_source)
            identity = str(record["case_id"])
            if identity in seen:
                raise ValueError(f"{path}:{number}: case_id duplicato: {identity}")
            seen.add(identity)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: nessun esempio.")
    return metadata, records


def divide(first, second):
    return first / second if second else None


def classification(records):
    tp = sum(
        row["pred"] == "vulnerable" and row["ground_truth"] == "vulnerable"
        for row in records
    )
    fp = sum(
        row["pred"] == "vulnerable" and row["ground_truth"] == "safe" for row in records
    )
    tn = sum(row["pred"] == "safe" and row["ground_truth"] == "safe" for row in records)
    fn = sum(
        row["pred"] == "safe" and row["ground_truth"] == "vulnerable" for row in records
    )
    total = len(records)
    classified = tp + fp + tn + fn
    positive = sum(row["ground_truth"] == "vulnerable" for row in records)
    negative = total - positive
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    valid_mcp = sum(row["mcp_valid"] is True for row in records)
    known_mcp = sum(row["mcp_valid"] is not None for row in records)
    return {
        "total": total,
        "classified": classified,
        "unclassified": total - classified,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "coverage": divide(classified, total),
        "accuracy_all": divide(tp + tn, total),
        "accuracy_classified": divide(tp + tn, classified),
        "recall_classified": recall,
        "specificity_classified": specificity,
        "recall_all": divide(tp, positive),
        "specificity_all": divide(tn, negative),
        "precision": divide(tp, tp + fp),
        "f1_classified": divide(2 * tp, 2 * tp + fp + fn),
        "balanced_classified": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "mcc_classified": divide(tp * tn - fp * fn, denominator),
        "mcp_valid": valid_mcp,
        "mcp_invalid": known_mcp - valid_mcp,
        "mcp_validity_unknown": sum(
            row["is_mcp"] and row["mcp_valid"] is None for row in records
        ),
        "mcp_valid_rate_known": divide(valid_mcp, known_mcp),
        "stored_pred_mismatches": sum(row["stored_pred_mismatch"] for row in records),
        "truncated": sum(row["truncated"] for row in records),
    }


def roc(scores, labels):
    positive = sum(value == "vulnerable" for value in labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return []
    grouped = collections.defaultdict(lambda: [0, 0])
    for value, truth in zip(scores, labels, strict=True):
        grouped[value][truth == "safe"] += 1
    tp = fp = 0
    curve = [(0.0, 0.0, None)]
    for threshold in sorted(grouped, reverse=True):
        first, second = grouped[threshold]
        tp += first
        fp += second
        curve.append((fp / negative, tp / positive, threshold))
    return curve


def auc_from(scores, labels):
    curve = roc(scores, labels)
    if not curve:
        return None
    return sum(
        (x2 - x1) * (y1 + y2) / 2
        for (x1, y1, _), (x2, y2, _) in itertools.pairwise(curve)
    )


def score_reports(records):
    grouped = collections.defaultdict(list)
    for row in records:
        if row["score"] is not None:
            grouped[row["score_source"]].append(row)
    result = {}
    for source, selected in sorted(grouped.items()):
        values = [row["score"] for row in selected]
        truths = [row["ground_truth"] for row in selected]
        result[source] = {
            "n_scored": len(selected),
            "score_coverage": divide(len(selected), len(records)),
            "positive": truths.count("vulnerable"),
            "negative": truths.count("safe"),
            "auroc_gt_vulnerable": auc_from(values, truths),
        }
    return result


def pair_report(records):
    grouped = collections.defaultdict(list)
    for row in records:
        if row["pair_id"] is not None:
            grouped[str(row["pair_id"])].append(row)
    complete = [
        rows
        for rows in grouped.values()
        if len(rows) == 2
        and {row["ground_truth"] for row in rows} == LABELS
        and not any(
            row["quartet_id"] is not None or len(row["cwes"]) > 1 for row in rows
        )
    ]
    return {
        "complete_simple_pairs": len(complete),
        "other_pair_groups": len(grouped) - len(complete),
        "both_correct": sum(
            all(row["pred"] == row["ground_truth"] for row in rows) for rows in complete
        ),
        "same_classified_verdict": sum(
            rows[0]["pred"] is not None and rows[0]["pred"] == rows[1]["pred"]
            for rows in complete
        ),
        "pairs_with_unclassified": sum(
            any(row["pred"] is None for row in rows) for rows in complete
        ),
    }


def quartet_report(records):
    grouped = collections.defaultdict(list)
    for row in records:
        key = row["quartet_id"]
        if key is None and len(row["cwes"]) == 2:
            key = row["pair_id"]
        if key is not None:
            grouped[str(key)].append(row)
    complete, incomplete = [], 0
    for rows in grouped.values():
        cwes = rows[0]["cwes"]
        states = []
        valid = len(rows) == 4 and len(cwes) == 2
        for row in rows:
            targets = row["target_labels"]
            if (
                not isinstance(targets, dict)
                or row["cwes"] != cwes
                or set(targets) != set(cwes)
            ):
                valid = False
                continue
            state = tuple(label(targets[cwe]) for cwe in cwes)
            states.append(state)
            expected = "vulnerable" if "vulnerable" in state else "safe"
            if None in state or expected != row["ground_truth"]:
                valid = False
        if valid and set(states) == {
            ("safe", "safe"),
            ("safe", "vulnerable"),
            ("vulnerable", "safe"),
            ("vulnerable", "vulnerable"),
        }:
            complete.append(rows)
        else:
            incomplete += 1
    return {
        "complete_quartets": len(complete),
        "incomplete_or_inconsistent_quartets": incomplete,
        "all_four_correct": sum(
            all(row["pred"] == row["ground_truth"] for row in rows) for rows in complete
        ),
        "all_four_classified": sum(
            all(row["pred"] is not None for row in rows) for rows in complete
        ),
    }


def paired_intervention(records):
    selected = [row for row in records if "baseline" in row]
    corrected = worsened = changed = label_flips = gained = lost = 0
    transitions = collections.Counter()
    for row in selected:
        baseline = row["baseline"]
        first, second = baseline["pred"], row["pred"]
        before, after = first == row["ground_truth"], second == row["ground_truth"]
        corrected += not before and after
        worsened += before and not after
        changed += first != second
        label_flips += first is not None and second is not None and first != second
        gained += first is None and second is not None
        lost += first is not None and second is None
        transitions[f"{first or 'unclassified'} -> {second or 'unclassified'}"] += 1
    return {
        "paired_examples": len(selected),
        "examples_without_baseline": len(records) - len(selected),
        "corrected_including_unclassified_baselines": corrected,
        "worsened_including_unclassified_interventions": worsened,
        "prediction_changes": changed,
        "valid_label_flips": label_flips,
        "gained_classification": gained,
        "lost_classification": lost,
        "accuracy_all_delta_paired": divide(corrected - worsened, len(selected)),
        "transitions": dict(sorted(transitions.items())),
    }


def number(value):
    return "n/a" if value is None else f"{value:.4f}"


def summary(records, by_cwe):
    report = {
        "classification": classification(records),
        "scores": score_reports(records),
        "pairs": pair_report(records),
        "quartets": quartet_report(records),
    }
    if by_cwe:
        grouped = collections.defaultdict(list)
        for row in records:
            for cwe in row["cwes"]:
                grouped[cwe].append(row)
        report["by_cwe_global_verdict"] = {
            cwe: classification(rows) for cwe, rows in sorted(grouped.items())
        }
    return report


def print_report(title, report):
    values = report["classification"]
    print(f"\n== {title}")
    print(
        f"Esempi {values['total']} | classificati {values['classified']} | non classificati {values['unclassified']}"
    )
    print(f"TP={values['tp']} FP={values['fp']} TN={values['tn']} FN={values['fn']}")
    print(
        f"Coverage={number(values['coverage'])} | accuracy tutti={number(values['accuracy_all'])} | accuracy classificati={number(values['accuracy_classified'])}"
    )
    print(
        f"Sui classificati: recall={number(values['recall_classified'])} specificity={number(values['specificity_classified'])} precision={number(values['precision'])} F1={number(values['f1_classified'])} balanced={number(values['balanced_classified'])} MCC={number(values['mcc_classified'])}"
    )
    print(
        f"Su tutte le GT: recall={number(values['recall_all'])} specificity={number(values['specificity_all'])}"
    )
    if values["mcp_valid"] or values["mcp_invalid"] or values["mcp_validity_unknown"]:
        print(
            f"MCP validi={values['mcp_valid']} invalidi={values['mcp_invalid']} non verificabili={values['mcp_validity_unknown']}"
        )
    print(
        f"Troncati={values['truncated']} | pred salvate discordanti dal parsing MCP={values['stored_pred_mismatches']}"
    )
    for source, item in report["scores"].items():
        print(
            f"Score={source} | N={item['n_scored']} | copertura={number(item['score_coverage'])} | AUROC GT vulnerable={number(item['auroc_gt_vulnerable'])}"
        )
    if not report["scores"]:
        print("Nessuno score compatibile disponibile: AUROC non calcolata.")
    print("Coppie:", json.dumps(report["pairs"], ensure_ascii=False))
    print("Quartetti:", json.dumps(report["quartets"], ensure_ascii=False))
    for cwe, item in report.get("by_cwe_global_verdict", {}).items():
        print(
            f"{cwe}: N={item['total']} coverage={number(item['coverage'])} accuracy tutti={number(item['accuracy_all'])}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Metriche dei verdetti legacy, MCP e interventi appaiati. Nessun caricamento del modello."
    )
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--by-cwe", action="store_true")
    parser.add_argument("--roc", type=Path)
    parser.add_argument("--out", type=Path, help="Riepilogo JSON.")
    parser.add_argument(
        "--variant", choices=("auto", "baseline", "intervention"), default="auto"
    )
    parser.add_argument(
        "--score-source", choices=("auto", "legacy", "ab", "none"), default="auto"
    )
    args = parser.parse_args()
    report = {
        "schema": "verdict-metrics-v2",
        "positive_class": "vulnerable",
        "undefined_metrics": None,
        "auroc_target": "ground_truth_not_error_prediction",
        "runs": [],
    }
    roc_rows = []
    try:
        for path in args.jsonl:
            metadata, records = load(path, args.score_source)
            variants = {
                "intervention"
                if metadata.get("kind") == "intervention"
                or any("baseline" in row for row in records)
                else "original": records
            }
            paired = [row["baseline"] for row in records if "baseline" in row]
            if paired:
                variants["baseline_paired_subset"] = paired
                variants["intervention_paired_subset"] = [
                    row for row in records if "baseline" in row
                ]
            if args.variant == "baseline":
                if not paired:
                    raise ValueError(f"{path}: baseline appaiata assente.")
                variants = {"baseline_paired_subset": paired}
            elif args.variant == "intervention":
                if "intervention" not in variants:
                    raise ValueError(f"{path}: non è una run di intervento.")
                variants = {"intervention": records}
            elif len(paired) == len(records):
                variants.pop("intervention_paired_subset", None)
            run = {"path": str(path), "metadata": metadata, "variants": {}}
            print(f"\nRUN: {path}")
            for name, rows in variants.items():
                item = summary(rows, args.by_cwe)
                run["variants"][name] = item
                print_report(name, item)
                if args.roc:
                    sources = {
                        row["score_source"] for row in rows if row["score"] is not None
                    }
                    for source in sorted(sources):
                        selected = [
                            row for row in rows if row["score_source"] == source
                        ]
                        curve = roc(
                            [row["score"] for row in selected],
                            [row["ground_truth"] for row in selected],
                        )
                        for fpr, tpr, threshold in curve:
                            roc_rows.append(
                                {
                                    "run": str(path),
                                    "variant": name,
                                    "score_source": source,
                                    "fpr": fpr,
                                    "tpr": tpr,
                                    "threshold": threshold,
                                }
                            )
            if paired:
                run["paired_intervention"] = paired_intervention(records)
                print(
                    "\nConfronto appaiato:",
                    json.dumps(run["paired_intervention"], ensure_ascii=False),
                )
            alternate = [
                dict(row, pred=row["pred_prob"])
                for row in records
                if row["pred_prob"] is not None
            ]
            if alternate and args.variant != "baseline":
                run["legacy_adjective_prediction_subset"] = classification(alternate)
                print(
                    f"Predizioni alternative dagli aggettivi: {len(alternate)}/{len(records)}; accuracy={number(run['legacy_adjective_prediction_subset']['accuracy_all'])}"
                )
            report["runs"].append(run)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Errore: {exc}\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        print(f"JSON: {args.out}")
    if args.roc:
        args.roc.parent.mkdir(parents=True, exist_ok=True)
        with args.roc.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "run",
                    "variant",
                    "score_source",
                    "fpr",
                    "tpr",
                    "threshold",
                ],
            )
            writer.writeheader()
            writer.writerows(roc_rows)
        print(
            f"ROC: {args.roc} ({len(roc_rows)} punti; file con solo header se nessuna ROC definita)"
        )


if __name__ == "__main__":
    main()
