# flake8: noqa: E501
"""
AA-LCR — Discriminative-Diversity Pruned variant.

Registers the benchmark name ``aa_lcr_pruned`` into evalscope's
BENCHMARK_REGISTRY.  Extends :class:`AALCRAdapter` with the same
discriminative-diversity pruner as the LCB variant, with one additional
parameter: ``judge_noise_margin``.

AA-LCR is graded by an LLM judge, which is non-deterministic.  A score of
0 or 1 may reflect judge variance rather than true model capability.  We
handle this as follows:

* **Observable** variance: cross-model score spread on a single sample.
  If model A scores 1 and model B scores 0, that spread is real signal
  (the models genuinely differ) — we want such samples.
* **Latent** judge noise: the same model on the same sample might score 0
  on one judge run and 1 on another.  Without repeated runs we cannot
  measure this directly.
* **Guard**: ``judge_noise_margin`` (default 0.0) suppresses samples where
  the cross-model spread is below the margin.  With binary 0/1 scores, any
  spread ≥ 0.0 passes (spread is either 0 or 1).  Raising the margin (e.g.
  to 0.5) would exclude all samples where models fully agree — effectively
  requiring *unanimous disagreement* as evidence the sample is truly
  discriminative rather than just caught a judge flip.  The default of 0.0
  is conservative and correct for binary data; it avoids discarding valid
  discriminative samples on the assumption of noise.

Usage with evalscope CLI (after ``import evalscope_ext``):

.. code-block:: bash

    evalscope eval \\
      --model my-new-model \\
      --datasets aa_lcr_pruned \\
      --dataset-args '{
        "aa_lcr_pruned": {
          "pruning_strategy": "discriminative_diversity",
          "prune_ratio": 0.3,
          "scores_dir": "/path/to/Evals/Part 1/reviews",
          "judge_noise_margin": 0.0
        }
      }'
"""
from __future__ import annotations

from typing import Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import (
    AALCRAdapter,
    JUDGE_PROMPT,
    PROMPT_TEMPLATE,
)
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope_ext.pruner import (
    load_scores_from_reviews,
    select_pruned_samples,
    validate_leave_one_out,
)

logger = get_logger()

# Score key used by the LLM judge
_SCORE_KEY = "acc"
# Review file prefix (matches the shipped Evals file names)
_BENCHMARK_PREFIX = "aa_lcr"


@register_benchmark(
    BenchmarkMeta(
        name="aa_lcr_pruned",
        pretty_name="AA-LCR (Pruned)",
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description=(
            "AA-LCR pruned to a discriminative-diversity subset. "
            "Samples are selected to maximise cross-model score variance while "
            "ensuring coverage across context lengths and difficulty levels. "
            "A judge_noise_margin parameter suppresses samples whose score "
            "spread may reflect LLM-judge non-determinism rather than true "
            "capability differences."
        ),
        dataset_id="evalscope/AA-LCR",
        metric_list=["acc"],
        few_shot_num=0,
        train_split=None,
        eval_split="test",
        prompt_template=PROMPT_TEMPLATE,
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
                    "Directory containing evalscope review JSONL files. "
                    "When null, the full benchmark is used."
                ),
                "value": None,
            },
            "n_buckets": {
                "type": "int",
                "description": "Number of difficulty buckets for diversity coverage.",
                "value": 5,
            },
            "judge_noise_margin": {
                "type": "float",
                "description": (
                    "Minimum cross-model score spread for a sample to be "
                    "treated as discriminative.  0.0 = any disagreement counts "
                    "(correct for binary 0/1 scoring).  Raise to suppress "
                    "samples that may be noise-driven when repeated-run data "
                    "suggests they are unstable."
                ),
                "value": 0.0,
            },
            "min_spearman": {
                "type": "float",
                "description": "Minimum leave-one-out Spearman correlation to accept the pruned set.",
                "value": 0.85,
            },
            "text_dir": {
                "type": "str | null",
                "description": "Local directory containing extracted AA-LCR text files; if null will auto-download.",
                "value": None,
            },
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """AA-LCR adapter with discriminative-diversity pruning and judge-noise guard."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pruning_strategy: str = self.extra_params.get(
            "pruning_strategy", "discriminative_diversity"
        )
        self._prune_ratio: float = float(self.extra_params.get("prune_ratio", 0.3))
        self._scores_dir: Optional[str] = self.extra_params.get("scores_dir")
        self._n_buckets: int = int(self.extra_params.get("n_buckets", 5))
        self._judge_noise_margin: float = float(
            self.extra_params.get("judge_noise_margin", 0.0)
        )
        self._min_spearman: float = float(self.extra_params.get("min_spearman", 0.85))

    def load_dataset(self) -> DatasetDict:
        full_dataset = super().load_dataset()

        if not self._scores_dir:
            logger.warning(
                "aa_lcr_pruned: no scores_dir provided — "
                "returning full benchmark (no pruning)."
            )
            return full_dataset

        logger.info(
            f"aa_lcr_pruned: loading historical scores from {self._scores_dir} "
            f"(judge_noise_margin={self._judge_noise_margin})"
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
            judge_noise_margin=self._judge_noise_margin,
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
            logger.warning(f"aa_lcr_pruned: {exc}")

        selected_set = set(selected)
        n_full = sum(len(list(ds)) for ds in full_dataset.values())
        logger.info(
            f"aa_lcr_pruned: selected {len(selected)}/{n_full} samples "
            f"(ratio={self._prune_ratio:.2f}, noise_margin={self._judge_noise_margin})"
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
