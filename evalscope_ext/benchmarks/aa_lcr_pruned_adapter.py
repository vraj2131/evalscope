# flake8: noqa: E501
"""
AA-LCR — Discriminative-Diversity Pruned variant.

Registers ``aa_lcr_pruned`` into evalscope's BENCHMARK_REGISTRY.
Pruning logic lives in :class:`PrunedAdapterMixin`; this file contains only
the benchmark-specific configuration plus the ``judge_noise_margin`` guard
that is unique to LLM-judged benchmarks.

AA-LCR is graded by an LLM judge, which is non-deterministic.  A score of
0 or 1 may reflect judge variance rather than true model capability.  We
handle this with ``judge_noise_margin``:

* **Observable** variance: cross-model score spread on a single sample.
  If model A scores 1 and model B scores 0, that spread is real signal — we
  want such samples.
* **Guard**: ``judge_noise_margin`` (default 0.0) suppresses samples whose
  cross-model spread is below the margin.  With binary 0/1 scores any spread
  ≥ 0.0 passes; raising the margin to 0.5 would require unanimous disagreement
  as evidence the sample is truly discriminative.  The default is correct for
  binary data and avoids discarding valid discriminative samples.

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

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import (
    AALCRAdapter,
    PROMPT_TEMPLATE,
)
from evalscope.constants import Tags

from evalscope_ext.benchmarks.base_pruned_adapter import PrunedAdapterMixin


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
class AALCRPrunedAdapter(PrunedAdapterMixin, AALCRAdapter):
    """AA-LCR adapter with discriminative-diversity pruning and judge-noise guard."""

    _SCORE_KEY = "acc"
    _BENCHMARK_PREFIX = "aa_lcr"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_pruning_params()
