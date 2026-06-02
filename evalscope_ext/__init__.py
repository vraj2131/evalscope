"""
evalscope_ext — discriminative-diversity pruning extension for evalscope.

Importing this package registers three new benchmarks into evalscope's
BENCHMARK_REGISTRY:

  * ``live_code_bench_pruned``   — LCB v5 pruned to discriminative-diversity subset
  * ``aa_lcr_pruned``            — AA-LCR pruned to discriminative-diversity subset
  * ``mmmu_encoder_stress``      — MMMU filtered to encoder-stress subjects

Simply import before calling ``evalscope eval``::

    import evalscope_ext
    # benchmark names are now available in the registry

Or from the CLI, prefix the command::

    python -c "import evalscope_ext" && evalscope eval ...
"""

from evalscope_ext.benchmarks import lcb_pruned_adapter, aa_lcr_pruned_adapter, mmmu_encoder_stress_adapter  # noqa: F401
