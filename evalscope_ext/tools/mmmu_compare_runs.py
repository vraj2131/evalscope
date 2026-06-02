#!/usr/bin/env python3
# flake8: noqa: E501
"""
Compare sighted vs blind MMMU runs to identify encoder-stress subjects.

Reads per-sample review JSONL files from a prior ``mmmu`` (sighted) run and
a prior ``mmmu_blind`` run on the same model, then reports per-subject blind
delta (sighted − blind accuracy).  Subjects with large delta have
image-load-bearing questions; subjects with small delta are dominated by
language priors.

Usage::

    python evalscope_ext/tools/mmmu_compare_runs.py \\
        --sighted-dir /path/to/reviews \\
        --blind-dir   /path/to/reviews \\
        [--min-delta  0.1]

The review directories may be the same path if both benchmark runs wrote
their outputs there (evalscope names files by benchmark: ``mmmu_<Subject>.jsonl``
vs ``mmmu_blind_<Subject>.jsonl``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def _load_subject_accuracy(reviews_dir: Path, prefix: str, subject: str) -> Optional[Tuple[float, int]]:
    """Return (mean_accuracy, n_samples) for one subject file, or None if missing."""
    path = reviews_dir / f'{prefix}_{subject}.jsonl'
    if not path.exists():
        return None
    correct = 0.0
    total = 0
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            acc = (
                row.get('sample_score', {})
                .get('score', {})
                .get('value', {})
                .get('acc', 0.0)
            )
            correct += float(acc)
            total += 1
    if total == 0:
        return None
    return correct / total, total


def _discover_subjects(reviews_dir: Path, prefix: str) -> list[str]:
    """Find all subjects that have a review file with the given prefix."""
    subjects = []
    for path in sorted(reviews_dir.glob(f'{prefix}_*.jsonl')):
        subject = path.stem[len(prefix) + 1:]
        subjects.append(subject)
    return subjects


def report(sighted_dir: Path, blind_dir: Path, min_delta: float) -> None:
    # Discover subjects from sighted run
    subjects = _discover_subjects(sighted_dir, 'mmmu')
    if not subjects:
        print(f'No mmmu_<Subject>.jsonl files found in {sighted_dir}')
        return

    rows = []
    for subject in subjects:
        sighted = _load_subject_accuracy(sighted_dir, 'mmmu', subject)
        blind = _load_subject_accuracy(blind_dir, 'mmmu_blind', subject)
        if sighted is None or blind is None:
            continue
        sighted_acc, n = sighted
        blind_acc, _ = blind
        delta = sighted_acc - blind_acc
        rows.append((subject, sighted_acc, blind_acc, delta, n))

    if not rows:
        print('No matching sighted + blind review files found.')
        return

    # Sort by delta descending (most encoder-load-bearing first)
    rows.sort(key=lambda r: -r[3])

    col = 42
    print(f'\n=== MMMU Sighted vs Blind — Blind Delta Report ===\n')
    print(f'Sighted dir : {sighted_dir}')
    print(f'Blind dir   : {blind_dir}')
    print(f'Min delta   : {min_delta}\n')
    print(f'{"Subject":<{col}} {"Sighted":>8} {"Blind":>8} {"Delta":>8} {"N":>5}  Signal')
    print('-' * (col + 38))

    encoder_stress = []
    language_prior = []

    for subject, sighted_acc, blind_acc, delta, n in rows:
        if delta >= min_delta:
            signal = 'encoder-load-bearing ✓'
            encoder_stress.append(subject)
        else:
            signal = 'language-prior dominated'
            language_prior.append(subject)
        print(
            f'{subject:<{col}} {sighted_acc:>8.1%} {blind_acc:>8.1%} '
            f'{delta:>+8.3f} {n:>5}  {signal}'
        )

    print('-' * (col + 38))
    print(f'\nEncoder-stress subjects (delta ≥ {min_delta}): {len(encoder_stress)}')
    for s in encoder_stress:
        print(f'  {s}')

    print(f'\nLanguage-prior dominated (delta < {min_delta}): {len(language_prior)}')
    for s in language_prior:
        print(f'  {s}')

    if encoder_stress:
        print(
            f'\nRun the encoder-stress probe on a candidate model:\n'
            f'  evalscope eval --model <candidate> --datasets mmmu \\\n'
            f'    --dataset-args \'{{"mmmu": {{"subset_list": {encoder_stress}}}}}\''
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare sighted vs blind MMMU runs to find encoder-stress subjects.'
    )
    parser.add_argument(
        '--sighted-dir', required=True, metavar='DIR',
        help='Directory containing mmmu_<Subject>.jsonl review files.',
    )
    parser.add_argument(
        '--blind-dir', required=True, metavar='DIR',
        help='Directory containing mmmu_blind_<Subject>.jsonl review files.',
    )
    parser.add_argument(
        '--min-delta', type=float, default=0.1, metavar='D',
        help='Minimum blind delta to classify a subject as encoder-load-bearing (default: 0.1).',
    )
    args = parser.parse_args()
    report(Path(args.sighted_dir), Path(args.blind_dir), args.min_delta)


if __name__ == '__main__':
    main()
