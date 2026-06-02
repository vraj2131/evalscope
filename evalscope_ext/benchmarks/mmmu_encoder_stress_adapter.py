# flake8: noqa: E501
"""
MMMU Encoder-Stress Probe.

Registers ``mmmu_encoder_stress`` into evalscope's BENCHMARK_REGISTRY.

**What we are detecting.**  An encoder can be degraded in ways invisible to
generic VQA — it may fail on fine spatial detail, small text, or visually
similar categories while language priors mask the failure on easy questions.
We want samples where the image is *provably load-bearing*, meaning removing
it collapses accuracy.

**Approach.**  Filter MMMU to subjects where image content is
non-substitutable by text (radiology, circuit schematics, spectroscopy, etc.)
then rank within each subject by an encoder-stress score that prefers:

  1. High-stress image types (diagrams, schematics, medical images, plots)
  2. Questions containing explicit visual references (``figure``, ``chart``,
     ``shown``, ``labeled``, etc.)

The top ``samples_per_subject`` samples are kept per subject, giving a probe
of ≤ ``len(ENCODER_STRESS_SUBJECTS) × samples_per_subject`` items.

**Blind-baseline extension (future).** Send each question with the image
replaced by a placeholder through the OpenAI API.  Subjects where
blind ≈ sighted accuracy have decorative images (language priors dominate);
subjects where blind << sighted have load-bearing images.  The static subject
filter below is a well-motivated approximation of this test that requires no
additional inference budget.

Usage with evalscope CLI (after ``import evalscope_ext``):

.. code-block:: bash

    evalscope eval \\
      --model my-vl-model \\
      --datasets mmmu_encoder_stress \\
      --dataset-args '{
        "mmmu_encoder_stress": {
          "samples_per_subject": 30
        }
      }'
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter, OPEN_PROMPT

logger = get_logger()

# Subjects where the image is non-substitutable by language priors.
# Selected based on domain: radiology/histology, circuit schematics, P&IDs,
# spectroscopy, crystallography, technical drawings.
ENCODER_STRESS_SUBJECTS = [
    'Architecture_and_Engineering',
    'Basic_Medical_Science',
    'Chemistry',
    'Clinical_Medicine',
    'Diagnostics_and_Laboratory_Medicine',
    'Electronics',
    'Energy_and_Power',
    'Materials',
    'Mechanical_Engineering',
]

# Image types that require fine spatial reasoning (encoder load-bearing).
_HIGH_STRESS_IMG_TYPES = {
    'figure', 'chart', 'diagram', 'schematic', 'plot',
    'medical image', 'formula', 'table', 'graph', 'map',
}

# Keywords that signal the answer must be read from the image.
_VISUAL_REF_RE = re.compile(
    r'\b(figure|fig|diagram|chart|table|graph|shown|depicted|illustrated|'
    r'labeled|marked|indicated|plotted|image|schematic|curve|spectrum)\b',
    re.IGNORECASE,
)


def _encoder_stress_score(sample: Any) -> int:
    """
    Simple ordinal score prioritising spatially demanding samples.

    +2  image type is in the high-stress set
    +1  question explicitly references a visual element
    """
    score = 0
    img_type = (sample.metadata.get('img_type') or '').lower()
    if any(t in img_type for t in _HIGH_STRESS_IMG_TYPES):
        score += 2

    # Extract text from the first user message content list
    question_text = ''
    if sample.input:
        msg = sample.input[0]
        content = msg.content if hasattr(msg, 'content') else []
        if isinstance(content, list):
            for part in content:
                if hasattr(part, 'text'):
                    question_text += ' ' + part.text
        elif isinstance(content, str):
            question_text = content

    if _VISUAL_REF_RE.search(question_text):
        score += 1

    return score


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_encoder_stress',
        pretty_name='MMMU Encoder-Stress Probe',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=(
            'MMMU filtered to encoder-stress subjects and image-load-bearing '
            'samples. Covers radiology, circuit schematics, spectroscopy plots, '
            'P&IDs, and technical drawings — domains where language priors cannot '
            'substitute for image content.  Samples are ranked by image-type '
            'stress and explicit visual references; top samples_per_subject kept '
            'per subject.'
        ),
        dataset_id='AI-ModelScope/MMMU',
        subset_list=ENCODER_STRESS_SUBJECTS,
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=OPEN_PROMPT,
        extra_params={
            'samples_per_subject': {
                'type': 'int',
                'description': (
                    'Maximum number of samples to keep per encoder-stress subject. '
                    'Default 30 keeps the full validation split per subject.'
                ),
                'value': 30,
            },
        },
    )
)
class MMMUEncoderStressAdapter(MMMUAdapter):
    """MMMU adapter restricted to encoder-stress subjects and image-load-bearing samples."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._samples_per_subject: int = int(
            self.extra_params.get('samples_per_subject', 30)
        )

    def load_dataset(self) -> DatasetDict:
        full_dataset = super().load_dataset()

        result: Dict[str, MemoryDataset] = {}
        total_kept = 0

        for subject, dataset in full_dataset.items():
            samples = list(dataset)
            scored: List[tuple[int, Any]] = [
                (_encoder_stress_score(s), s) for s in samples
            ]
            scored.sort(key=lambda x: -x[0])
            kept = [s for _, s in scored[: self._samples_per_subject]]
            result[subject] = MemoryDataset(
                samples=kept,
                name=dataset.name,
                location=dataset.location if hasattr(dataset, 'location') else None,
            )
            total_kept += len(kept)
            logger.info(
                f'mmmu_encoder_stress: {subject} — kept {len(kept)}/{len(samples)} samples'
            )

        logger.info(
            f'mmmu_encoder_stress: total {total_kept} samples across '
            f'{len(result)} encoder-stress subjects'
        )

        from evalscope.api.dataset import DatasetDict as DD
        dd = DD()
        dd.update(result)
        return dd
