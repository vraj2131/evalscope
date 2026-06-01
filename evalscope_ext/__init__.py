"""
evalscope_ext — discriminative-diversity pruning extension for evalscope.

Importing this package registers two new benchmarks into evalscope's
BENCHMARK_REGISTRY:

  * ``live_code_bench_pruned``
  * ``aa_lcr_pruned``

Simply import before calling ``evalscope eval``::

    import evalscope_ext
    # benchmark names are now available in the registry

Or from the CLI, prefix the command::

    python -c "import evalscope_ext" && evalscope eval ...
"""

from evalscope_ext.benchmarks import lcb_pruned_adapter, aa_lcr_pruned_adapter  # noqa: F401
