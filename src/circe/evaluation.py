from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .inference import write_json
from .labels import LABEL_VERSION, correctness
from .metrics import METRICS, PreparedMetrics, metric_dict
from .scoring import Weights, fuse

METHODS = ("CIRCE", "CIC", "GS", "negative_NLL")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def external_labels(path):
    if path is None:
        return None
    labels = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["model"]), str(row["dataset"]), str(row["id"]))
            value = row["correct"]
            if not isinstance(value, (bool, int)) or value not in (0, 1):
                raise ValueError("external correct labels must be binary booleans or integers")
            if key in labels:
                raise ValueError("duplicate external label")
            labels[key] = bool(value)
    return labels


def load_run(path, policy="final", labels=None):
    path = Path(path)
    run = read_json(path / "run.json")
    completion = read_json(path / "completion.json")
    if completion.get("status") != "COMPLETE":
        raise ValueError("inference did not complete; incomplete runs cannot enter evaluation")
    ids = run["ids"]
    expected = {f"{i:08d}.json" for i in range(len(ids))}
    if run["n"] != len(ids) or len(set(ids)) != len(ids) or completion["records"] != len(ids):
        raise ValueError("invalid run ID coverage")
    try:
        weights = Weights(**run["weights"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("the run must record a valid supplied fusion configuration") from error
    for kind in ("generations", "records"):
        if {p.name for p in (path / kind).glob("*.json")} != expected:
            raise ValueError(f"{kind} coverage differs from the declared inputs")
    rows, used = [], set()
    for index, qid in enumerate(ids):
        r = read_json(path / "records" / f"{index:08d}.json")
        g = read_json(path / "generations" / f"{index:08d}.json")
        if (r["id"], r["dataset"], r["model"]) != (qid, run["dataset"], run["model"]):
            raise ValueError("record identity differs from the run")
        for key in ("id", "dataset", "model", "input", "reference", "output_ids", "output_text",
                    "raw_output_text", "terminated", "generation_config"):
            if r[key] != g[key]:
                raise ValueError(f"original generation differs at {key}")
        if r["tokens"] != len(r["output_ids"]) or r["tokens"] != r["scores"]["num_decisions"]:
            raise ValueError("decision / original ID count differs")
        if r["scoring_mask"] != [True] * r["tokens"] or not r["tokens"]:
            raise ValueError("the main-method decision mask differs")
        values = r["scores"]
        expected_score = fuse(values["cic"], values["gs"], values["mean_nll"], weights=weights)
        if not np.isclose(values["score"], expected_score, atol=1e-12, rtol=1e-12):
            raise ValueError("stored score differs from its recorded fusion configuration")
        if not isinstance(r["terminated"], bool) or r["truncated"] != (not r["terminated"]):
            raise ValueError("invalid termination flags")
        confidence = np.array([values["score"], values["cic"], values["gs"], -values["mean_nll"]])
        if not np.isfinite(confidence).all():
            raise ValueError("all scores must be finite; do not drop invalid rows")
        key = (r["model"], r["dataset"], qid)
        if labels is not None:
            if key not in labels:
                raise ValueError("an external label is missing for a planned example")
            label = dict(correct=labels[key], strict_correct=None, extracted=None,
                         reason="external_fixed_label", has_answer_line=None, policy="external")
            used.add(key)
        else:
            label = correctness(r, policy)
        rows.append(dict(record=r, label=label, confidence=confidence))
    return run, rows, used


def _csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_runs(run_paths, output, *, policy="final", labels_path=None,
                  bootstrap=5000, seed=1729):
    if bootstrap < 0:
        raise ValueError("bootstrap draws must be nonnegative")
    labels = external_labels(labels_path)
    groups, used, systems, reference = [], set(), set(), {}
    shared_weights = None
    for path in run_paths:
        run, rows, label_keys = load_run(path, policy, labels)
        weights = Weights(**run["weights"])
        if shared_weights is None:
            shared_weights = weights
        elif weights != shared_weights:
            raise ValueError("all runs must use the same supplied fusion weights")
        key = (run["model"], run["dataset"])
        if key in systems:
            raise ValueError("duplicate model/dataset run")
        systems.add(key)
        used.update(label_keys)
        rows = sorted(rows, key=lambda r: r["record"]["id"])
        identity = [(r["record"]["id"], r["record"]["input"]["question"],
                     r["record"]["input"]["options"], r["record"]["input"]["passage"],
                     r["record"]["reference"], r["record"].get("cluster_id")) for r in rows]
        if run["dataset"] in reference and identity != reference[run["dataset"]]:
            raise ValueError("cross-model examples, references, or bootstrap clusters differ")
        reference[run["dataset"]] = identity
        y = np.array([r["label"]["correct"] for r in rows], dtype=int)
        stats = [PreparedMetrics(y, np.array([r["confidence"][i] for r in rows])) for i in range(len(METHODS))]
        groups.append({"run": run, "rows": rows, "stats": stats,
                       "point": np.stack([s.point() for s in stats]), "y": y})
    if not groups:
        raise ValueError("at least one completed inference run is required")
    if labels is not None and used != set(labels):
        raise ValueError("external labels contain unrequested or missing records")
    point = np.stack([g["point"] for g in groups])
    macro = point.mean(axis=0)
    draws = np.full((len(groups), bootstrap, len(METHODS), len(METRICS)), np.nan)
    rng = np.random.default_rng(seed)
    for dataset in sorted(reference):
        affected = [i for i, g in enumerate(groups) if g["run"]["dataset"] == dataset]
        first = groups[affected[0]]["rows"]
        cluster_ids = [r["record"].get("cluster_id") for r in first]
        if bootstrap and any(c is None for c in cluster_ids):
            raise ValueError("RACE passage cluster IDs are required for bootstrap; supply passage/cluster_id or --bootstrap 0")
        if not bootstrap:
            continue
        units = sorted(set(cluster_ids))
        unit_index = {value: i for i, value in enumerate(units)}
        index = np.array([unit_index[c] for c in cluster_ids])
        for start in range(0, bootstrap, 128):
            size = min(128, bootstrap - start)
            counts = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=size)
            weights = counts[:, index]
            for gi in affected:
                for mi, stat in enumerate(groups[gi]["stats"]):
                    draws[gi, start:start + size, mi] = stat.weighted(weights)
    macro_draws = draws.mean(axis=0)
    complete = np.isfinite(macro_draws).all(axis=(1, 2))
    ci_supported = bool(bootstrap and complete.sum() >= .9 * bootstrap and np.isfinite(macro).all())
    table, label_rows = [], []
    for gi, group in enumerate(groups):
        run, rows, y = group["run"], group["rows"], group["y"]
        for mi, method in enumerate(METHODS):
            table.append(dict(model=run["model"], dataset=run["dataset"], method=method,
                              n=len(y), correct=int(y.sum()), incorrect=int((1 - y).sum()),
                              accuracy=float(y.mean()), **metric_dict(point[gi, mi])))
        for row in rows:
            record, label = row["record"], row["label"]
            label_rows.append(dict(model=run["model"], dataset=run["dataset"], id=record["id"],
                                   terminated=record["terminated"], tokens=record["tokens"],
                                   score=record["scores"]["score"], **label))
    macro_rows = []
    for mi, method in enumerate(METHODS):
        row = dict(method=method, **metric_dict(macro[mi]))
        for ki, metric in enumerate(METRICS):
            bounds = np.quantile(macro_draws[complete, mi, ki], [.025, .975]) if ci_supported else (None, None)
            row[metric + "_lo95"], row[metric + "_hi95"] = bounds
        macro_rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _csv(output / "metrics.csv", table)
    _csv(output / "macro_metrics.csv", macro_rows)
    _csv(output / "labels.csv", label_rows)
    if bootstrap:
        with (output / "bootstrap.npz").open("xb") as handle:
            np.savez_compressed(handle, group_draws=draws, macro_draws=macro_draws, complete=complete,
                                per_system_point=point, macro_point=macro)
    summary = {
        "status": "EVALUATED", "groups": len(groups), "records": len(label_rows),
        "methods": list(METHODS), "metrics": list(METRICS), "label_policy": "external" if labels else policy,
        "fusion_weights": asdict(shared_weights),
        "label_version": LABEL_VERSION, "bootstrap_seed": seed, "bootstrap_draws": bootstrap,
        "complete_draws": int(complete.sum()), "macro_ci_supported": ci_supported,
        "macro": [{k: float(v) if isinstance(v, np.floating) else v for k, v in r.items()} for r in macro_rows],
        "conventions": {"positive_class": "correct", "aupr_definition": "average precision (AP)",
                        "fpr95": "first observed threshold attaining at least 95% TPR; no interpolation",
                        "macro": "equal weight for every supplied model/dataset group, including undefined groups",
                        "bootstrap": "shared question draws across models/methods; RACE passage clusters",
                        "undefined_metrics": "null for single-class groups; no imputation or group deletion",
                        "invalid_answers": "fixed external labels supplied by caller" if labels is not None else "retained as incorrect",
                        "truncated_answers": "fixed external labels supplied by caller" if labels is not None else "retained as incorrect"},
    }
    write_json(output / "summary.json", summary)
    return summary
