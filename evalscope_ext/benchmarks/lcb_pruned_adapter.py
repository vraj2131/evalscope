# flake8: noqa: E501
"""
LiveCodeBench — Discriminative-Diversity Pruned variant.

Registers ``live_code_bench_pruned`` into evalscope's BENCHMARK_REGISTRY.
Pruning logic lives in :class:`PrunedAdapterMixin`; this file contains only
the benchmark-specific configuration.

Usage with evalscope CLI (after ``import evalscope_ext``):

.. code-block:: bash

    evalscope eval \\
      --model my-new-model \\
      --datasets live_code_bench_pruned \\
      --dataset-args '{
        "live_code_bench_pruned": {
          "pruning_strategy": "discriminative_diversity",
          "prune_ratio": 0.3,
          "scores_dir": "/path/to/Evals/Part 1/reviews"
        }
      }'

``scores_dir`` must contain review JSONL files named
``live_code_bench_v5__<model>.jsonl``.  When omitted the adapter falls back
to the full benchmark (no pruning).
"""
from __future__ import annotations

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import (
    LiveCodeBenchAdapter,
)
from evalscope.constants import Tags

from evalscope_ext.benchmarks.base_pruned_adapter import PrunedAdapterMixin


@register_benchmark(
    BenchmarkMeta(
        name="live_code_bench_pruned",
        pretty_name="Live-Code-Bench (Pruned)",
        tags=[Tags.CODING],
        description=(
            "LiveCodeBench v5 pruned to a discriminative-diversity subset. "
            "Samples are selected to maximise cross-model score variance while "
            "preserving coverage of easy, medium, and hard problems. "
            "Ranking agreement is validated via leave-one-model-out Spearman "
            "correlation before the pruned set is returned."
        ),
        dataset_id="evalscope/livecodebench_code_generation_lite_parquet",
        subset_list=["release_v5"],
        default_subset="release_v5",
        metric_list=["acc"],
        aggregation="mean_and_pass_at_k",
        eval_split="test",
        prompt_template=(
            "### Question:\n{question_content}\n\n"
            "{format_prompt} ### Answer: (use the provided format with backticks)\n\n"
        ),
        review_timeout=6,
        extra_params={
            "pruning_strategy": {
                "type": "str",
                "description": "Pruning strategy. Currently only 'discriminative_diversity' is supported.",
                "value": "discriminative_diversity",
                "choices": ["discriminative_diversity"],
            },
            "prune_ratio": {
                "type": "float",
                "description": "Fraction of samples to keep (0, 1].",
                "value": 0.3,
            },
            "scores_dir": {
                "type": "str | null",
                "description": (
                    "Directory containing evalscope review JSONL files from "
                    "previous model runs.  When null the full benchmark is used."
                ),
                "value": None,
            },
            "n_buckets": {
                "type": "int",
                "description": "Number of difficulty buckets for diversity coverage.",
                "value": 5,
            },
            "min_spearman": {
                "type": "float",
                "description": "Minimum leave-one-out Spearman correlation to accept the pruned set.",
                "value": 0.85,
            },
            "start_date": {
                "type": "str | null",
                "description": "Filter problems starting from this date (YYYY-MM-DD). Null keeps all.",
                "value": None,
            },
            "end_date": {
                "type": "str | null",
                "description": "Filter problems up to this date (YYYY-MM-DD). Null keeps all.",
                "value": None,
            },
            "debug": {
                "type": "bool",
                "description": "Enable verbose debug logging.",
                "value": False,
            },
        },
        sandbox_config={
            "image": "python:3.11-slim",
            "tools_config": {
                "shell_executor": {},
                "python_executor": {},
            },
        },
    )
)
class LCBPrunedAdapter(PrunedAdapterMixin, LiveCodeBenchAdapter):
    """LiveCodeBench adapter with discriminative-diversity pruning."""

    _SCORE_KEY = "pass"
    _BENCHMARK_PREFIX = "live_code_bench_v5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_pruning_params()
