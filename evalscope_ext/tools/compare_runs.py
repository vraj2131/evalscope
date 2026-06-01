"""
Compare full-benchmark vs pruned-benchmark model rankings.

Usage
-----
::

    python -m evalscope_ext.tools.compare_runs \\
        --full   /path/to/Evals/Part\\ 1/reviews \\
        --pruned /path/to/Evals/Part\\ 1/reviews \\
        --benchmark lcb              # or: aa_lcr
        --prune-ratio 0.3

For the ``--pruned`` path you can point at the same reviews directory as
``--full``; the tool will re-compute the pruned subset on-the-fly (using
:func:`evalscope_ext.pruner.select_pruned_samples`) and compare against
the full-set ranking.  This mirrors the exact logic the adapter applies at
eval time, so the numbers here are what you would observe in production.

Output
------
Prints a table of per-model pass-rates (full vs pruned), a ranking-agreement
section (Spearman ρ and Kendall τ between the two orderings), and
leave-one-model-out Spearman scores.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure evalscope_ext is importable when invoked as ``python -m``
_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evalscope_ext.pruner import (
    _model_pass_rates,
    _ranking,
    _spearman,
    _variance,
    load_scores_from_reviews,
    select_pruned_samples,
    validate_leave_one_out,
)


# ---------------------------------------------------------------------------
# Kendall τ
# ---------------------------------------------------------------------------

def _kendall_tau(a: List[str], b: List[str]) -> float:
    """Kendall rank correlation coefficient between two orderings."""
    models = [m for m in a if m in b]
    n = len(models)
    if n < 2:
        return 1.0
    rank_b = {m: i for i, m in enumerate(b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = 1  # a is already sorted
            sign_b = 1 if rank_b[models[i]] < rank_b[models[j]] else -1
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1
    denom = n * (n - 1) // 2
    return (concordant - discordant) / denom if denom else 1.0


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt(value: float, width: int = 8) -> str:
    return f"{value:.4f}".rjust(width)


def _print_rate_table(
    full_rates: Dict[str, float],
    pruned_rates: Dict[str, float],
    full_n: int,
    pruned_n: int,
) -> None:
    models = sorted(full_rates.keys())
    col_w = max(len(m) for m in models) + 2

    header = (
        f"{'Model':<{col_w}} "
        f"{'Full (' + str(full_n) + ')':>14} "
        f"{'Pruned (' + str(pruned_n) + ')':>16} "
        f"{'Delta':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in models:
        full_r = full_rates.get(m, float("nan"))
        pruned_r = pruned_rates.get(m, float("nan"))
        delta = pruned_r - full_r if math.isfinite(pruned_r) and math.isfinite(full_r) else float("nan")
        delta_str = f"{delta:+.4f}" if math.isfinite(delta) else "   N/A"
        print(
            f"{m:<{col_w}} "
            f"{full_r:>14.4f} "
            f"{pruned_r:>16.4f} "
            f"{delta_str:>8}"
        )


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

_BENCHMARKS = {
    "lcb": {
        "prefix": "live_code_bench_v5",
        "score_key": "pass",
        "noise_margin": 0.0,
    },
    "aa_lcr": {
        "prefix": "aa_lcr",
        "score_key": "acc",
        "noise_margin": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare full vs pruned benchmark rankings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full",
        required=True,
        metavar="DIR",
        help="Directory containing full-run review JSONL files.",
    )
    parser.add_argument(
        "--pruned",
        required=True,
        metavar="DIR",
        help=(
            "Directory containing review JSONL files used to compute the "
            "pruned subset (may be the same as --full)."
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=list(_BENCHMARKS),
        default="lcb",
        help="Which benchmark to analyse (default: lcb).",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.3,
        metavar="R",
        help="Fraction of samples to keep (default: 0.3).",
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=5,
        metavar="K",
        help="Number of difficulty buckets (default: 5).",
    )
    parser.add_argument(
        "--judge-noise-margin",
        type=float,
        default=0.0,
        metavar="M",
        help=(
            "Minimum cross-model spread to count a sample as discriminative "
            "(default: 0.0, correct for binary 0/1 scores)."
        ),
    )
    parser.add_argument(
        "--min-spearman",
        type=float,
        default=0.85,
        metavar="RHO",
        help="LOO Spearman threshold (default: 0.85).",
    )
    args = parser.parse_args(argv)

    cfg = _BENCHMARKS[args.benchmark]

    # --- Load scores -------------------------------------------------------
    print(f"\n=== {args.benchmark.upper()} — Full vs Pruned Comparison ===\n")

    full_scores = load_scores_from_reviews(
        reviews_dir=args.full,
        score_key=cfg["score_key"],
        benchmark_prefix=cfg["prefix"],
    )
    pruning_scores = load_scores_from_reviews(
        reviews_dir=args.pruned,
        score_key=cfg["score_key"],
        benchmark_prefix=cfg["prefix"],
    )

    # --- Compute pruned subset --------------------------------------------
    selected = select_pruned_samples(
        scores=pruning_scores,
        prune_ratio=args.prune_ratio,
        n_buckets=args.n_buckets,
        judge_noise_margin=args.judge_noise_margin,
    )

    full_n = len(full_scores)
    pruned_n = len(selected)
    print(
        f"Samples:   {full_n} full  →  {pruned_n} pruned  "
        f"({100 * pruned_n / full_n:.1f}% kept)\n"
    )

    # --- Pass-rate table --------------------------------------------------
    full_rates = _model_pass_rates(full_scores)
    pruned_rates = _model_pass_rates(full_scores, indices=selected)

    print("Pass-rate by model:")
    _print_rate_table(full_rates, pruned_rates, full_n, pruned_n)

    # --- Ranking agreement ------------------------------------------------
    full_ranking = _ranking(full_rates)
    pruned_ranking = _ranking(pruned_rates)

    rho = _spearman(full_ranking, pruned_ranking)
    tau = _kendall_tau(full_ranking, pruned_ranking)

    print(f"\nFull ranking:   {' > '.join(full_ranking)}")
    print(f"Pruned ranking: {' > '.join(pruned_ranking)}")
    match = full_ranking == pruned_ranking
    print(f"Rankings match: {'YES ✓' if match else 'NO ✗'}")
    print(f"Spearman ρ:     {rho:.4f}")
    print(f"Kendall τ:      {tau:.4f}")

    # --- Leave-one-model-out ----------------------------------------------
    print("\nLeave-one-model-out Spearman (measures generalisation):")
    try:
        loo = validate_leave_one_out(
            scores=full_scores,
            selected_indices=selected,
            min_spearman=args.min_spearman,
        )
        for model, loo_rho in sorted(loo.items()):
            status = "✓" if loo_rho >= args.min_spearman else "✗"
            print(f"  {status} {model:<30} ρ = {loo_rho:.4f}")
        print(f"\nAll LOO Spearman ≥ {args.min_spearman}: PASS ✓")
    except ValueError as exc:
        print(f"  FAIL ✗  {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
