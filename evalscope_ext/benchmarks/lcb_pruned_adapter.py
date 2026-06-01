# flake8: noqa: E501
"""
LiveCodeBench — Discriminative-Diversity Pruned variant.

Registers the benchmark name ``live_code_bench_pruned`` into evalscope's
BENCHMARK_REGISTRY.  The adapter subclasses :class:`LiveCodeBenchAdapter`
and overrides :meth:`load_dataset` to apply the discriminative-diversity
pruner before returning samples to the evaluator.

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
``live_code_bench_v5__<model>.jsonl`` (standard evalscope output format).
When ``scores_dir`` is omitted the adapter falls back to the full benchmark
(no pruning), which is useful for generating the initial score files.
"""
from __future__ import annotations

import itertools
from typing import Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import (
    LiveCodeBenchAdapter,
)
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope_ext.pruner import (
    load_scores_from_reviews,
    select_pruned_samples,
    validate_leave_one_out,
)

logger = get_logger()

# Score key used by the LCB sandbox grader
_SCORE_KEY = "pass"
# Review file prefix (matches the shipped Evals file names)
_BENCHMARK_PREFIX = "live_code_bench_v5"


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
                "description": (
                    "Pruning strategy to use. Currently only "
                    "'discriminative_diversity' is supported."
                ),
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
class LCBPrunedAdapter(LiveCodeBenchAdapter):
    """LiveCodeBench adapter with discriminative-diversity pruning."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pruning_strategy: str = self.extra_params.get(
            "pruning_strategy", "discriminative_diversity"
        )
        self._prune_ratio: float = float(self.extra_params.get("prune_ratio", 0.3))
        self._scores_dir: Optional[str] = self.extra_params.get("scores_dir")
        self._n_buckets: int = int(self.extra_params.get("n_buckets", 5))
        self._min_spearman: float = float(self.extra_params.get("min_spearman", 0.85))

    def load_dataset(self) -> DatasetDict:
        full_dataset = super().load_dataset()

        if not self._scores_dir:
            logger.warning(
                "live_code_bench_pruned: no scores_dir provided — "
                "returning full benchmark (no pruning)."
            )
            return full_dataset

        logger.info(
            f"live_code_bench_pruned: loading historical scores from {self._scores_dir}"
        )
        scores = load_scores_from_reviews(
            reviews_dir=self._scores_dir,
            score_key=_SCORE_KEY,
            benchmark_prefix=_BENCHMARK_PREFIX,
        )

        selected = select_pruned_samples(
            scores=scores,
            prune_ratio=self._prune_ratio,
            n_buckets=self._n_buckets,
            judge_noise_margin=0.0,
        )

        # Leave-one-model-out validation
        try:
            rhos = validate_leave_one_out(
                scores=scores,
                selected_indices=selected,
                min_spearman=self._min_spearman,
            )
            for model, rho in sorted(rhos.items()):
                logger.info(f"  LOO Spearman ({model} held out): {rho:.3f}")
        except ValueError as exc:
            logger.warning(f"live_code_bench_pruned: {exc}")

        selected_set = set(selected)
        n_full = sum(len(list(ds)) for ds in full_dataset.values())
        logger.info(
            f"live_code_bench_pruned: selected {len(selected)}/{n_full} samples "
            f"(ratio={self._prune_ratio:.2f})"
        )

        return self._filter_dataset(full_dataset, selected_set)

    @staticmethod
    def _filter_dataset(dataset_dict: DatasetDict, keep: set) -> DatasetDict:
        """Keep only samples whose sequential position is in *keep*."""
        result = {}
        offset = 0
        for subset_name, dataset in dataset_dict.items():
            kept = []
            for local_idx, sample in enumerate(dataset):
                if offset + local_idx in keep:
                    kept.append(sample)
            result[subset_name] = MemoryDataset(
                samples=kept,
                name=dataset.name,
                location=dataset.location if hasattr(dataset, "location") else None,
            )
            offset += len(list(dataset))
        from evalscope.api.dataset import DatasetDict as DD
        dd = DD()
        dd.update(result)
        return dd
