# flake8: noqa: E501
"""
PrunedAdapterMixin — shared pruning logic for any evalscope benchmark.

Subclass this mixin alongside an evalscope adapter to get discriminative-
diversity pruning for free.  The subclass must:

  1. Declare class attributes ``_SCORE_KEY`` and ``_BENCHMARK_PREFIX``.
  2. Call ``self._init_pruning_params()`` in ``__init__``.

Example::

    class MyPrunedAdapter(PrunedAdapterMixin, MyBenchmarkAdapter):
        _SCORE_KEY = "acc"
        _BENCHMARK_PREFIX = "my_benchmark"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._init_pruning_params()

The mixin's ``load_dataset`` calls ``super().load_dataset()`` (i.e. the
benchmark adapter's loader via Python MRO) then applies pruning.
"""
from __future__ import annotations

from typing import Optional

from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.utils.logger import get_logger

from evalscope_ext.pruner import (
    load_scores_from_reviews,
    select_pruned_samples,
    validate_leave_one_out,
)

logger = get_logger()


class PrunedAdapterMixin:
    """
    Mixin that adds discriminative-diversity pruning to any evalscope adapter.

    Subclasses must set:
      _SCORE_KEY        — score field in review JSONL (e.g. "pass", "acc")
      _BENCHMARK_PREFIX — file prefix for review files (e.g. "live_code_bench_v5")
    """

    _SCORE_KEY: str = ""
    _BENCHMARK_PREFIX: str = ""

    def _init_pruning_params(self) -> None:
        """Read pruning hyper-params from ``self.extra_params``."""
        self._pruning_strategy: str = self.extra_params.get(
            "pruning_strategy", "discriminative_diversity"
        )
        self._prune_ratio: float = float(self.extra_params.get("prune_ratio", 0.3))
        self._scores_dir: Optional[str] = self.extra_params.get("scores_dir")
        self._n_buckets: int = int(self.extra_params.get("n_buckets", 5))
        self._min_spearman: float = float(self.extra_params.get("min_spearman", 0.85))
        self._judge_noise_margin: float = float(
            self.extra_params.get("judge_noise_margin", 0.0)
        )

    def load_dataset(self) -> DatasetDict:
        full_dataset = super().load_dataset()  # type: ignore[misc]

        if not self._scores_dir:
            logger.warning(
                f"{self._BENCHMARK_PREFIX}_pruned: no scores_dir provided — "
                "returning full benchmark (no pruning)."
            )
            return full_dataset

        logger.info(
            f"{self._BENCHMARK_PREFIX}_pruned: loading historical scores "
            f"from {self._scores_dir}"
        )
        scores = load_scores_from_reviews(
            reviews_dir=self._scores_dir,
            score_key=self._SCORE_KEY,
            benchmark_prefix=self._BENCHMARK_PREFIX,
        )

        selected = select_pruned_samples(
            scores=scores,
            prune_ratio=self._prune_ratio,
            n_buckets=self._n_buckets,
            judge_noise_margin=self._judge_noise_margin,
        )

        try:
            rhos = validate_leave_one_out(
                scores=scores,
                selected_indices=selected,
                min_spearman=self._min_spearman,
            )
            for model, rho in sorted(rhos.items()):
                logger.info(f"  LOO Spearman ({model} held out): {rho:.3f}")
        except ValueError as exc:
            logger.warning(f"{self._BENCHMARK_PREFIX}_pruned: {exc}")

        n_full = sum(len(list(ds)) for ds in full_dataset.values())
        logger.info(
            f"{self._BENCHMARK_PREFIX}_pruned: selected {len(selected)}/{n_full} "
            f"samples (ratio={self._prune_ratio:.2f})"
        )

        return self._filter_dataset(full_dataset, set(selected))

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
