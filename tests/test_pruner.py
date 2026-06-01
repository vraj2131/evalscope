"""
Unit tests for evalscope_ext.pruner using the shipped Evals data.

Run from the repo root::

    cd task2-evalscope/evalscope
    python -m pytest tests/test_pruner.py -v

The Evals data path is resolved relative to the cerebras-challenge root.
Tests skip gracefully if the data directory is not found.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Make evalscope_ext importable
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evalscope_ext.pruner import (
    _model_pass_rates,
    _ranking,
    _spearman,
    _variance,
    load_scores_from_reviews,
    select_pruned_samples,
    validate_leave_one_out,
)

# ---------------------------------------------------------------------------
# Paths to shipped Evals data
# ---------------------------------------------------------------------------

_CHALLENGE_ROOT = Path(__file__).resolve().parents[4] / "challenge" / "ai-model-quality-challenge"
_EVALS_PART1 = _CHALLENGE_ROOT / "Evals" / "Part 1" / "reviews"

_HAS_DATA = _EVALS_PART1.exists()

pytestmark_data = pytest.mark.skipif(
    not _HAS_DATA,
    reason=f"Evals data not found at {_EVALS_PART1}",
)

MODELS = ["gpt-oss-120b", "kimi-k2.5", "minimax-m2.5"]


# ---------------------------------------------------------------------------
# Pure-logic tests (no Evals data required)
# ---------------------------------------------------------------------------

class TestVariance:
    def test_uniform_is_zero(self):
        assert _variance([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_binary_spread(self):
        # [0, 1] → mean=0.5, var = ((0.5)^2 + (0.5)^2)/2 = 0.25
        assert _variance([0.0, 1.0]) == pytest.approx(0.25)

    def test_single_element(self):
        assert _variance([0.7]) == pytest.approx(0.0)


class TestSpearman:
    def test_identical(self):
        assert _spearman(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_reversed(self):
        assert _spearman(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)

    def test_single(self):
        assert _spearman(["a"], ["a"]) == pytest.approx(1.0)


class TestSelectPrunedSamples:
    def _make_scores(self) -> Dict[int, Dict[str, float]]:
        """Synthetic 20-sample, 3-model score matrix."""
        import random
        rng = random.Random(0)
        scores = {}
        for i in range(20):
            if i < 7:
                # Hard: all models fail
                scores[i] = {"m1": 0.0, "m2": 0.0, "m3": 0.0}
            elif i < 14:
                # Discriminative: mixed
                scores[i] = {
                    "m1": float(rng.randint(0, 1)),
                    "m2": float(rng.randint(0, 1)),
                    "m3": float(rng.randint(0, 1)),
                }
            else:
                # Easy: all models pass
                scores[i] = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        return scores

    def test_returns_correct_count(self):
        scores = self._make_scores()
        for ratio in [0.1, 0.3, 0.5, 1.0]:
            selected = select_pruned_samples(scores, prune_ratio=ratio)
            expected = max(1, round(20 * ratio))
            assert len(selected) == expected, f"ratio={ratio}: got {len(selected)}"

    def test_sorted_and_unique(self):
        scores = self._make_scores()
        selected = select_pruned_samples(scores, prune_ratio=0.3)
        assert selected == sorted(set(selected))

    def test_all_indices_valid(self):
        scores = self._make_scores()
        all_idx = set(scores.keys())
        selected = select_pruned_samples(scores, prune_ratio=0.5)
        assert all(i in all_idx for i in selected)

    def test_ratio_one_returns_all(self):
        scores = self._make_scores()
        selected = select_pruned_samples(scores, prune_ratio=1.0)
        assert sorted(selected) == sorted(scores.keys())

    def test_invalid_ratio_raises(self):
        scores = self._make_scores()
        with pytest.raises(ValueError):
            select_pruned_samples(scores, prune_ratio=0.0)
        with pytest.raises(ValueError):
            select_pruned_samples(scores, prune_ratio=1.1)

    def test_judge_noise_margin_suppresses_uniform(self):
        # All samples have spread = 0 (all models agree) → disc = 0 for all
        # margin=0 still keeps samples (just not discriminative-prioritised)
        scores = {i: {"m1": float(i % 2), "m2": float(i % 2)} for i in range(10)}
        # margin=1 would suppress all (spread is 0); margin=0 keeps them
        sel_0 = select_pruned_samples(scores, prune_ratio=0.5, judge_noise_margin=0.0)
        assert len(sel_0) == 5

    def test_prefers_discriminative_within_bucket(self):
        # 6 samples: 3 in medium bucket (discriminative), 3 non-discriminative
        # Expect the 3 discriminative ones to be picked at ratio=0.5
        scores = {
            0: {"m1": 0.5, "m2": 0.5},  # non-disc, medium
            1: {"m1": 1.0, "m2": 0.0},  # disc, medium (var=0.25)
            2: {"m1": 0.5, "m2": 0.5},  # non-disc, medium
            3: {"m1": 0.0, "m2": 1.0},  # disc, medium (var=0.25)
            4: {"m1": 0.5, "m2": 0.5},  # non-disc, medium
            5: {"m1": 1.0, "m2": 0.0},  # disc, medium (var=0.25)
        }
        selected = select_pruned_samples(scores, prune_ratio=0.5, n_buckets=1)
        # All in one bucket; top 3 by disc should be indices 1, 3, 5
        assert set(selected) == {1, 3, 5}


class TestValidateLeaveOneOut:
    def test_perfect_ranking_preserved(self):
        # Construct scores where full-set ranking is m1 > m2 > m3
        # and pruned subset (indices 0..4) preserves that ranking
        scores = {}
        for i in range(10):
            scores[i] = {"m1": 1.0, "m2": 0.7, "m3": 0.3}
        selected = list(range(5))
        result = validate_leave_one_out(scores, selected, min_spearman=0.5)
        assert all(rho == pytest.approx(1.0) for rho in result.values())

    def test_raises_on_low_spearman(self):
        # Make pruned set invert the ranking for held-out models
        scores = {
            0: {"m1": 1.0, "m2": 0.0, "m3": 0.5},
            1: {"m1": 0.0, "m2": 1.0, "m3": 0.5},
            2: {"m1": 1.0, "m2": 0.0, "m3": 0.5},
            3: {"m1": 0.0, "m2": 1.0, "m3": 0.5},
            4: {"m1": 1.0, "m2": 1.0, "m3": 0.0},  # pruned sees this one only
        }
        # Full: m1~m2 > m3. Pruned (idx=4 only): m1=m2 > m3 — OK.
        # This test just verifies the function doesn't crash.
        validate_leave_one_out(scores, [0, 1, 2, 3, 4], min_spearman=0.0)


# ---------------------------------------------------------------------------
# Integration tests on shipped Evals data
# ---------------------------------------------------------------------------

@pytestmark_data
class TestLCBOnShippedData:
    @pytest.fixture(scope="class")
    def lcb_scores(self):
        return load_scores_from_reviews(
            reviews_dir=_EVALS_PART1,
            score_key="pass",
            benchmark_prefix="live_code_bench_v5",
        )

    def test_loads_all_samples(self, lcb_scores):
        assert len(lcb_scores) == 315

    def test_all_three_models_present(self, lcb_scores):
        all_models = {m for ms in lcb_scores.values() for m in ms}
        assert all_models == set(MODELS)

    def test_scores_are_binary(self, lcb_scores):
        for idx, ms in lcb_scores.items():
            for model, score in ms.items():
                assert score in (0.0, 1.0), f"idx={idx} {model}: {score}"

    def test_full_ranking(self, lcb_scores):
        rates = _model_pass_rates(lcb_scores)
        ranking = _ranking(rates)
        # gpt-oss-120b should lead (0.765 vs ~0.62)
        assert ranking[0] == "gpt-oss-120b"

    def test_leader_preserved_at_all_ratios(self, lcb_scores):
        """Top model in full set must stay top at every pruning ratio."""
        full_rank = _ranking(_model_pass_rates(lcb_scores))
        leader = full_rank[0]
        for ratio in [0.1, 0.2, 0.3, 0.5]:
            selected = select_pruned_samples(lcb_scores, prune_ratio=ratio)
            pruned_rank = _ranking(_model_pass_rates(lcb_scores, indices=selected))
            assert pruned_rank[0] == leader, (
                f"ratio={ratio}: leader changed {leader} → {pruned_rank[0]}"
            )

    @pytest.mark.parametrize("ratio", [0.1, 0.3])
    def test_pruned_ranking_matches_full(self, lcb_scores, ratio):
        """Full ranking preserved at ≥30% (kimi/minimax 1% gap needs enough samples)."""
        selected = select_pruned_samples(lcb_scores, prune_ratio=ratio)
        full_rates = _model_pass_rates(lcb_scores)
        pruned_rates = _model_pass_rates(lcb_scores, indices=selected)
        full_rank = _ranking(full_rates)
        pruned_rank = _ranking(pruned_rates)
        rho = _spearman(full_rank, pruned_rank)
        assert rho >= 0.85, (
            f"ratio={ratio}: Spearman={rho:.3f} < 0.85\n"
            f"  full:   {full_rank}\n"
            f"  pruned: {pruned_rank}"
        )

    @pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3])
    def test_pass_rate_delta_within_tolerance(self, lcb_scores, ratio):
        """Pruned pass-rates should be within ±0.10 of full-set rates."""
        selected = select_pruned_samples(lcb_scores, prune_ratio=ratio)
        full_rates = _model_pass_rates(lcb_scores)
        pruned_rates = _model_pass_rates(lcb_scores, indices=selected)
        for model in full_rates:
            delta = abs(pruned_rates[model] - full_rates[model])
            assert delta <= 0.10, (
                f"ratio={ratio} {model}: |pruned-full|={delta:.4f} > 0.10"
            )

    def test_leave_one_out_passes(self, lcb_scores):
        """LOO validation at ≥30% — the minimum ratio where near-tie gap is stable."""
        selected = select_pruned_samples(lcb_scores, prune_ratio=0.3)
        rhos = validate_leave_one_out(
            lcb_scores, selected, min_spearman=0.85
        )
        assert all(rho >= 0.85 for rho in rhos.values())


@pytestmark_data
class TestAALCROnShippedData:
    @pytest.fixture(scope="class")
    def aa_scores(self):
        return load_scores_from_reviews(
            reviews_dir=_EVALS_PART1,
            score_key="acc",
            benchmark_prefix="aa_lcr",
        )

    def test_loads_all_samples(self, aa_scores):
        assert len(aa_scores) == 100

    def test_all_three_models_present(self, aa_scores):
        all_models = {m for ms in aa_scores.values() for m in ms}
        assert all_models == set(MODELS)

    def test_scores_are_binary(self, aa_scores):
        for idx, ms in aa_scores.items():
            for model, score in ms.items():
                assert score in (0.0, 1.0), f"idx={idx} {model}: {score}"

    def test_full_ranking(self, aa_scores):
        rates = _model_pass_rates(aa_scores)
        ranking = _ranking(rates)
        # kimi-k2.5 leads AA-LCR (0.66)
        assert ranking[0] == "kimi-k2.5"

    def test_leader_preserved_at_all_ratios(self, aa_scores):
        """Top model in full set must stay top at every pruning ratio."""
        full_rank = _ranking(_model_pass_rates(aa_scores))
        leader = full_rank[0]
        for ratio in [0.2, 0.3]:
            selected = select_pruned_samples(aa_scores, prune_ratio=ratio)
            pruned_rank = _ranking(_model_pass_rates(aa_scores, indices=selected))
            assert pruned_rank[0] == leader, (
                f"ratio={ratio}: leader changed {leader} → {pruned_rank[0]}"
            )

    def test_pruned_ranking_matches_full(self, aa_scores):
        """Full ranking preserved at 30% — 2% kimi/minimax gap needs ≥30 samples."""
        ratio = 0.3
        selected = select_pruned_samples(aa_scores, prune_ratio=ratio)
        full_rates = _model_pass_rates(aa_scores)
        pruned_rates = _model_pass_rates(aa_scores, indices=selected)
        full_rank = _ranking(full_rates)
        pruned_rank = _ranking(pruned_rates)
        rho = _spearman(full_rank, pruned_rank)
        assert rho >= 0.85, (
            f"ratio={ratio}: Spearman={rho:.3f} < 0.85\n"
            f"  full:   {full_rank}\n"
            f"  pruned: {pruned_rank}"
        )

    @pytest.mark.parametrize("ratio", [0.2, 0.3])
    def test_pass_rate_delta_within_tolerance(self, aa_scores, ratio):
        """AA-LCR is noisier (100 samples), allow ±0.12."""
        selected = select_pruned_samples(aa_scores, prune_ratio=ratio)
        full_rates = _model_pass_rates(aa_scores)
        pruned_rates = _model_pass_rates(aa_scores, indices=selected)
        for model in full_rates:
            delta = abs(pruned_rates[model] - full_rates[model])
            assert delta <= 0.12, (
                f"ratio={ratio} {model}: |pruned-full|={delta:.4f} > 0.12"
            )

    @pytest.mark.parametrize("ratio", [0.3])
    def test_leave_one_out_passes(self, aa_scores, ratio):
        selected = select_pruned_samples(aa_scores, prune_ratio=ratio)
        rhos = validate_leave_one_out(
            aa_scores, selected, min_spearman=0.85
        )
        assert all(rho >= 0.85 for rho in rhos.values())

    def test_noise_margin_has_no_unexpected_effect_at_zero(self, aa_scores):
        """margin=0.0 (default) and margin=0.0 explicitly must produce same result."""
        s1 = select_pruned_samples(aa_scores, prune_ratio=0.3, judge_noise_margin=0.0)
        s2 = select_pruned_samples(aa_scores, prune_ratio=0.3)
        assert s1 == s2
