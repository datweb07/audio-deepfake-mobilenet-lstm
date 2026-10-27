"""Cluster-aware factorial contrasts: paired channels, unpaired attack/generator cohorts."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .analysis import loss
from .provenance import known


def generator_channel_interaction(rows: list[dict], *, channel: str, factor_definition: str,
                                  factor_kind: str = "attack", class_label: int | None = None,
                                  cluster_field: str = "speaker_id", risk: str = "error",
                                  replicates: int = 2000, seed: int = 804) -> dict:
    """I=(G1 channel-clean)-(G0 channel-clean), equal corpus and cluster weights.

    The caller supplies documented G0/G1 membership, not inferred model familiarity.
    Shared speaker clusters are resampled jointly. Different utterances are never paired.
    """
    if factor_kind not in {"attack", "generator"} or not factor_definition.strip() or class_label not in {None, 0, 1}:
        raise ValueError("Explicit factor definition/type and valid class stratum required")
    if replicates < 100 or cluster_field not in {"speaker_id", "source_speaker_id", "parent_recording_id"}:
        raise ValueError("Use explicit independent-unit clusters and >=100 bootstrap draws")
    paired = defaultdict(dict)
    for row in rows:
        if row["condition_id"] not in {"clean", channel}:
            continue
        if class_label is not None and int(row["label"]) != class_label:
            continue
        if row.get("status") != "OK" or row.get("factor_group") not in {"G0", "G1"}:
            raise ValueError("Invalid score status or undefined factorial membership")
        identity = row["sample_id"]
        if row["condition_id"] in paired[identity]:
            raise ValueError("Duplicate factorial score row")
        for field in (cluster_field, "dataset_id", factor_kind + "_id"):
            if not known(str(row.get(field, "UNKNOWN"))):
                raise ValueError(f"Cannot identify factor/cluster: {field}")
        score, threshold = float(row["raw_score"]), float(row["threshold"])
        if int(row["label"]) not in {0, 1}:
            raise ValueError("Invalid REAL/FAKE label")
        if not np.isfinite(score) or not 0 <= score <= 1 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid factorial score")
        if int(row["predicted_label"]) != int(score >= threshold):
            raise ValueError("Inconsistent fixed threshold")
        paired[identity][row["condition_id"]] = row
    if not paired:
        raise ValueError("No factorial observations")
    # Public factors must have common language/source support. Dataset IDs alone
    # cannot detect generators nested inside different underlying corpora.
    for field in ("language", "source_corpus"):
        if any(field in row for conditions in paired.values() for row in conditions.values()):
            support = {group: set() for group in ("G0", "G1")}
            for conditions in paired.values():
                for row in conditions.values():
                    value = str(row.get(field, "UNKNOWN"))
                    if not known(value):
                        raise ValueError(f"NOT_IDENTIFIABLE: unknown {field}")
                    support[row["factor_group"]].add(value)
                if len({str(row.get(field)) for row in conditions.values()}) != 1:
                    raise ValueError(f"Pairing mismatch: {field}")
            if support["G0"] != support["G1"]:
                raise ValueError(f"CONFOUNDED_FACTOR: unequal {field} support")
    groups = defaultdict(lambda: defaultdict(list))
    counts = {}
    model_strata = set()
    for identity, conditions in paired.items():
        if set(conditions) != {"clean", channel}:
            raise ValueError("Missing paired factorial channel cell")
        clean, changed = conditions["clean"], conditions[channel]
        for key in ("factor_group", "label", "dataset_id", cluster_field, factor_kind + "_id", "threshold", "model_id", "seed"):
            if str(clean[key]) != str(changed[key]):
                raise ValueError(f"Factorial pairing mismatch: {key}")
        model_strata.add((clean["model_id"], str(clean["threshold"]), str(clean["seed"])))
        corpus, factor, cluster = clean["dataset_id"], clean["factor_group"], clean[cluster_field]
        groups[(corpus, factor)][cluster].append((loss(clean, risk), loss(changed, risk)))
    if len(model_strata) != 1:
        raise ValueError("Analyze model/seed/threshold strata separately")
    factor_corpora = {factor: {corpus for corpus, member in groups if member == factor} for factor in ("G0", "G1")}
    if not factor_corpora["G0"] or factor_corpora["G0"] != factor_corpora["G1"]:
        raise ValueError("Generator/attack factor confounded with corpus or missing cell")
    corpora = sorted(factor_corpora["G0"])
    cluster_corpora = defaultdict(set)
    for (corpus, _), clusters in groups.items():
        for cluster in clusters:
            cluster_corpora[cluster].add(corpus)
    if any(len(member_corpora) > 1 for member_corpora in cluster_corpora.values()):
        raise ValueError("Cluster reused across corpora: current stratified bootstrap requires disjoint corpus clusters")
    means = {}
    for corpus in corpora:
        for factor in ("G0", "G1"):
            selected = [values["clean"] for values in paired.values()
                        if values["clean"]["dataset_id"] == corpus and values["clean"]["factor_group"] == factor]
            labels = {int(row["label"]) for row in selected}
            if class_label is None and labels != {0, 1}:
                raise ValueError("Single-class cell: request an explicit classwise estimand instead")
            clusters = groups[(corpus, factor)]
            if len(clusters) < 2:
                raise ValueError("At least two independent clusters per corpus/factor cell required")
            means[(corpus, factor)] = {cluster: np.mean(values, axis=0) for cluster, values in clusters.items()}
            counts[f"{corpus}/{factor}"] = {"recordings": len(selected), "clusters": len(clusters),
                                            "class_counts": {str(label): sum(int(row["label"]) == label for row in selected) for label in (0, 1)}}

    def contrasts(weights=None):
        differences = []
        for corpus in corpora:
            marginal = {}
            for factor in ("G0", "G1"):
                entries = means[(corpus, factor)]
                cluster_weights = np.array([weights[corpus].get(cluster, 0) if weights else 1 for cluster in entries])
                if cluster_weights.sum() == 0:
                    return None
                marginal[factor] = np.average(list(entries.values()), weights=cluster_weights, axis=0)
            shift = marginal["G1"] - marginal["G0"]
            differences.append([shift[1] - shift[0], shift[0], shift[1]])
        return np.mean(differences, axis=0)

    estimate = contrasts()
    generator = np.random.default_rng(seed)
    draws, rejected = [], 0
    for _ in range(replicates * 20):
        weights = {}
        for corpus in corpora:
            clusters = sorted(set(means[(corpus, "G0")]) | set(means[(corpus, "G1")]))
            multiplicities = generator.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)))
            weights[corpus] = dict(zip(clusters, multiplicities))
        value = contrasts(weights)
        if value is None:
            rejected += 1
        else:
            draws.append(value)
        if len(draws) == replicates:
            break
    if len(draws) < replicates:
        raise ValueError("Too few valid cluster bootstrap replicates")
    intervals = np.quantile(draws, [.025, .975], axis=0)
    names = ("generator_channel_interaction", "generator_shift_clean", "generator_shift_channel")
    return {"scope": "DESCRIPTIVE_DEVELOPMENT", "factor_kind": factor_kind, "factor_definition": factor_definition,
            "model_training_familiarity": "NOT_INFERRED", "class_label": class_label, "risk": risk,
            "estimand": "equal-corpus difference of equal-cluster channel risk changes; paired within recording only",
            "cells": counts, "cluster_field": cluster_field, "replicates": replicates, "seed": seed,
            "rejected_bootstrap_draws_missing_group": rejected,
            "contrasts": {name: {"estimate": float(estimate[index]), "ci95": intervals[:, index].tolist()}
                          for index, name in enumerate(names)},
            "limitations": "Observational contrast; no causal or calibrated-probability interpretation; no multiple-testing claim"}
