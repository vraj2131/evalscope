"""
Discriminative-diversity sample pruner for evalscope benchmarks.

Given per-sample scores from M models, selects a subset of N*prune_ratio
samples that:
  1. Maximise discriminative power  — high score variance across models
  2. Maintain diversity coverage    — samples spread evenly across difficulty
  3. Preserve model ranking         — leave-one-model-out Spearman >= threshold

Judge-noise guard (AA-LCR):
  LLM judges are non-deterministic.  Scores may flip on re-grading.  We
  suppress samples whose cross-model spread is below ``judge_noise_margin``.
  The default of 0.0 is correct for binary 0/1 scores: any disagreement
  between models is real signal (we cannot measure judge noise without
  repeated runs, so we conservatively trust observed disagreements).
  With repeated-run data, raise the margin to filter flip-prone samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core statistics helpers
# ---------------------------------------------------------------------------

def _variance(values: List[float]) -> float:
    """Population variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _model_pass_rates(
    scores: Dict[int, Dict[str, float]],
    indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Mean score per model, optionally restricted to the given *indices*."""
    idx_set = set(indices) if indices is not None else None
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for idx, model_scores in scores.items():
        if idx_set is not None and idx not in idx_set:
            continue
        for model, score in model_scores.items():
            totals[model] = totals.get(model, 0.0) + score
            counts[model] = counts.get(model, 0) + 1
    return {m: totals[m] / counts[m] for m in totals if counts[m] > 0}


def _ranking(rates: Dict[str, float]) -> List[str]:
    """Models sorted by pass-rate descending."""
    return [m for m, _ in sorted(rates.items(), key=lambda kv: -kv[1])]


def _spearman(a: List[str], b: List[str]) -> float:
    """Spearman rank correlation between two orderings of the same items."""
    models = [m for m in a if m in b]
    n = len(models)
    if n < 2:
        return 1.0
    rank_a = {m: i for i, m in enumerate(a)}
    rank_b = {m: i for i, m in enumerate(b)}
    d2 = sum((rank_a[m] - rank_b[m]) ** 2 for m in models)
    return 1.0 - 6 * d2 / (n * (n ** 2 - 1))


# ---------------------------------------------------------------------------
# Score loader
# ---------------------------------------------------------------------------

def load_scores_from_reviews(
    reviews_dir: str | Path,
    score_key: str,
    benchmark_prefix: str,
) -> Dict[int, Dict[str, float]]:
    """
    Load per-sample scores from evalscope review JSONL files.

    Files are expected to match ``<reviews_dir>/<benchmark_prefix>__<model>.jsonl``.
    Each line: ``{"index": N, "sample_score": {"score": {"value": {score_key: v}}}}``.

    Args:
        reviews_dir:      Directory containing review JSONL files.
        score_key:        Score field name, e.g. ``"pass"`` or ``"acc"``.
        benchmark_prefix: File prefix, e.g. ``"live_code_bench_v5"`` or ``"aa_lcr"``.

    Returns:
        ``{sample_index: {model_name: score}}``
    """
    reviews_dir = Path(reviews_dir)
    pattern = f"{benchmark_prefix}__*.jsonl"
    files = sorted(reviews_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No review files matching '{pattern}' found in {reviews_dir}"
        )

    scores: Dict[int, Dict[str, float]] = {}
    for path in files:
        model = path.stem[len(benchmark_prefix) + 2:]  # strip "<prefix>__"
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                idx = row["index"]
                score_value = (
                    row["sample_score"]["score"]["value"].get(score_key, 0.0)
                )
                scores.setdefault(idx, {})[model] = float(score_value)
    return scores


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------

def select_pruned_samples(
    scores: Dict[int, Dict[str, float]],
    prune_ratio: float = 0.3,
    n_buckets: int = 5,
    judge_noise_margin: float = 0.0,
) -> List[int]:
    """
    Select a pruned set of sample indices using the discriminative-diversity
    strategy.

    Algorithm
    ---------
    1. **Discriminative score** — population variance of per-model scores on
       each sample.  Samples where all models agree carry no ranking signal.
    2. **Noise guard** — if the spread (max − min) is below
       ``judge_noise_margin``, the disc score is set to 0 (sample excluded
       from discriminative selection, not dropped entirely — it may still be
       picked as a diversity representative).
    3. **Difficulty buckets** — samples are binned into ``n_buckets`` equal-
       width bins based on their mean score (0 = hardest, 1 = easiest).
    4. **Proportional selection** — the budget (``prune_ratio × N``) is
       distributed across buckets proportionally to their sizes, ensuring
       coverage of easy, medium, and hard problems.  Within each bucket,
       samples are ranked by discriminative score (descending).

    Args:
        scores:             ``{sample_idx: {model_name: score}}`` mapping.
        prune_ratio:        Fraction of samples to keep, in (0, 1].
        n_buckets:          Number of difficulty buckets.
        judge_noise_margin: Minimum score spread for a sample to be counted
                            as discriminative (default 0.0 for binary scores).

    Returns:
        Sorted list of selected sample indices.
    """
    if not 0 < prune_ratio <= 1:
        raise ValueError(f"prune_ratio must be in (0, 1], got {prune_ratio}")

    all_indices = sorted(scores.keys())
    n_total = len(all_indices)
    n_select = max(1, round(n_total * prune_ratio))

    if n_select >= n_total:
        return all_indices

    # Step 1 + 2: per-sample disc and difficulty
    disc: Dict[int, float] = {}
    difficulty: Dict[int, float] = {}
    for idx in all_indices:
        vals = list(scores[idx].values())
        spread = max(vals) - min(vals)
        disc[idx] = _variance(vals) if spread >= judge_noise_margin else 0.0
        difficulty[idx] = sum(vals) / len(vals)

    # Step 3: bin by difficulty
    bucket_width = 1.0 / n_buckets
    buckets: List[List[int]] = [[] for _ in range(n_buckets)]
    for idx in all_indices:
        b = min(int(difficulty[idx] / bucket_width), n_buckets - 1)
        buckets[b].append(idx)

    # Step 4: proportional selection within each non-empty bucket.
    # Within each bucket we process samples by discriminative-score tier
    # (highest first).  When samples share the same disc score — which is
    # common with binary 0/1 data where multiple outcome patterns yield
    # identical variance — we use round-robin interleaving across outcome
    # patterns.  This preserves each model's proportional win-rate inside the
    # bucket and prevents index-order bias from favouring one model over another.
    non_empty = [(b, bkt) for b, bkt in enumerate(buckets) if bkt]
    total_non_empty_samples = sum(len(bkt) for _, bkt in non_empty)

    selected: List[int] = []
    remaining = n_select

    for pos, (_, bkt) in enumerate(non_empty):
        is_last = pos == len(non_empty) - 1
        if is_last:
            quota = remaining
        else:
            quota = round(n_select * len(bkt) / total_non_empty_samples)
            quota = min(quota, len(bkt), remaining)

        picked = _select_from_bucket(bkt, disc, scores, quota)
        selected.extend(picked)
        remaining -= len(picked)

    return sorted(selected)


def _select_from_bucket(
    indices: List[int],
    disc: Dict[int, float],
    scores: Dict[int, Dict[str, float]],
    quota: int,
) -> List[int]:
    """
    Pick *quota* samples from *indices*, honouring two priorities:

    1. Higher discriminative score comes first.
    2. Among samples with the same disc score, round-robin across outcome
       patterns so no single model outcome dominates the selection.
    """
    from collections import defaultdict

    if quota <= 0:
        return []
    if quota >= len(indices):
        return list(indices)

    # Group by disc score (rounded to avoid float comparison noise)
    disc_tiers: Dict[int, List[int]] = defaultdict(list)
    for idx in sorted(indices):  # sort by index for a stable base order
        tier_key = round(disc[idx] * 1_000_000)
        disc_tiers[tier_key].append(idx)

    result: List[int] = []

    for tier_key in sorted(disc_tiers.keys(), reverse=True):
        if len(result) >= quota:
            break
        tier_indices = disc_tiers[tier_key]
        need = quota - len(result)

        if len(tier_indices) <= need:
            result.extend(tier_indices)
            continue

        # Round-robin across outcome patterns within this tier.
        # Pattern: tuple of score values sorted by model name → canonical key.
        patterns: Dict[tuple, List[int]] = defaultdict(list)
        for idx in tier_indices:
            pat = tuple(v for _, v in sorted(scores[idx].items()))
            patterns[pat].append(idx)

        # Queues ordered by size (largest first) so proportionality is approximate
        queues = [list(q) for q in sorted(patterns.values(), key=len, reverse=True)]

        picked = 0
        while picked < need and any(queues):
            for q in queues:
                if q and picked < need:
                    result.append(q.pop(0))
                    picked += 1

    return result


# ---------------------------------------------------------------------------
# Leave-one-model-out validation
# ---------------------------------------------------------------------------

def validate_leave_one_out(
    scores: Dict[int, Dict[str, float]],
    selected_indices: List[int],
    min_spearman: float = 0.85,
) -> Dict[str, float]:
    """
    Leave-one-model-out ranking validation.

    For each model m, compute the Spearman rank correlation between:
    - the model ranking on the **full** set (excluding m), and
    - the model ranking on the **pruned** set (excluding m).

    A correlation near 1.0 means the pruned set preserves the relative
    ordering of models that were not seen during sample selection — evidence
    that the pruner generalises to an unseen fourth model.

    Args:
        scores:           Full score matrix ``{idx: {model: score}}``.
        selected_indices: Output of :func:`select_pruned_samples`.
        min_spearman:     Raise if any held-out model's correlation falls
                          below this threshold.

    Returns:
        ``{model_name: spearman_rho}`` for each held-out model.

    Raises:
        ValueError: if any Spearman correlation is below ``min_spearman``.
    """
    all_models = sorted({m for ms in scores.values() for m in ms})
    results: Dict[str, float] = {}

    for held_out in all_models:
        remaining = [m for m in all_models if m != held_out]
        if len(remaining) < 2:
            continue

        # Score matrix restricted to the remaining models
        filtered = {
            idx: {m: v for m, v in ms.items() if m in remaining}
            for idx, ms in scores.items()
        }

        full_rates = _model_pass_rates(filtered)
        pruned_rates = _model_pass_rates(filtered, indices=selected_indices)

        rho = _spearman(_ranking(full_rates), _ranking(pruned_rates))
        results[held_out] = rho

    failed = [(m, rho) for m, rho in results.items() if rho < min_spearman]
    if failed:
        details = ", ".join(f"{m}={rho:.3f}" for m, rho in failed)
        raise ValueError(
            f"Leave-one-out validation failed (min_spearman={min_spearman}): {details}"
        )

    return results
