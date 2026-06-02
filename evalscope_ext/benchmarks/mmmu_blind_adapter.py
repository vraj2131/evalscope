# flake8: noqa: E501
"""
MMMU Blind — image-stripped baseline for encoder-stress measurement.

Registers ``mmmu_blind`` into evalscope's BENCHMARK_REGISTRY. Identical to
the standard MMMU adapter except every image is replaced with the text
placeholder ``[image omitted]`` before the prompt is sent to the model via
the standard OpenAI interface.

**Why this matters.**  Running ``mmmu_blind`` and ``mmmu`` (sighted) on the
same model gives a per-subject blind delta (sighted − blind accuracy).
Subjects with large delta have image-load-bearing questions — the encoder is
doing real work.  Subjects with small delta are dominated by language priors
— a degraded encoder would be invisible there.

The delta is the empirical ground truth that ``mmmu_encoder_stress`` uses to
select its probe set.  The blind pass costs one additional inference run and
needs no tooling beyond the standard OpenAI-compatible API.

Pipeline::

    # 1. Sighted run (standard MMMU)
    evalscope eval --model my-vl-model --datasets mmmu ...
    # → review files: outputs/my-vl-model/reviews/mmmu_<Subject>.jsonl

    # 2. Blind run (this adapter)
    evalscope eval --model my-vl-model --datasets mmmu_blind ...
    # → review files: outputs/my-vl-model/reviews/mmmu_blind_<Subject>.jsonl

    # 3. Compare to find encoder-stress subjects
    python evalscope_ext/tools/mmmu_compare_runs.py \\
        --sighted-dir outputs/my-vl-model/reviews \\
        --blind-dir   outputs/my-vl-model/reviews

    # 4. Run encoder-stress probe on the candidate model
    evalscope eval --model candidate --datasets mmmu_encoder_stress \\
      --dataset-args '{
        "mmmu_encoder_stress": {
          "sighted_reviews_dir": "outputs/my-vl-model/reviews",
          "blind_reviews_dir":   "outputs/my-vl-model/reviews"
        }
      }'
"""
from __future__ import annotations

import ast
import re as _re

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter, OPEN_PROMPT, SUBSET_LIST

_IMG_TAG_RE = _re.compile(r'<image\s+\d+>', _re.IGNORECASE)
_PLACEHOLDER = '[image omitted]'


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_blind',
        pretty_name='MMMU Blind Baseline',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=(
            'MMMU with all images replaced by "[image omitted]". '
            'Run alongside standard MMMU to compute per-subject blind delta '
            '(sighted − blind accuracy). Subjects with large delta are '
            'encoder-load-bearing; subjects with small delta are dominated by '
            'language priors. The blind delta drives sample selection in '
            'mmmu_encoder_stress.'
        ),
        dataset_id='AI-ModelScope/MMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=OPEN_PROMPT,
    )
)
class MMMUBlindAdapter(MMMUAdapter):
    """
    MMMU adapter that strips all images before sending to the model.

    Overrides ``create_content_and_answers_list`` to:
    1. Null out all image fields so the parent builds no image Content objects.
    2. Replace ``<image N>`` markers in question/options text with
       ``[image omitted]`` so the model sees where an image was but gets no
       visual signal.
    """

    def create_content_and_answers_list(self, record):
        blind = dict(record)

        # Null out every image field → parent builds empty image_map
        for i in range(1, MMMUAdapter.MAX_IMAGES + 1):
            blind[f'image_{i}'] = None

        # Replace <image N> markers in question text
        blind['question'] = _IMG_TAG_RE.sub(_PLACEHOLDER, record.get('question', ''))

        # Replace in options (stored as a Python literal string)
        raw_opts = record.get('options', '[]')
        try:
            opts = ast.literal_eval(raw_opts) if isinstance(raw_opts, str) else list(raw_opts)
            blind['options'] = str([_IMG_TAG_RE.sub(_PLACEHOLDER, str(o)) for o in opts])
        except Exception:
            blind['options'] = _IMG_TAG_RE.sub(_PLACEHOLDER, str(raw_opts))

        return super().create_content_and_answers_list(blind)
