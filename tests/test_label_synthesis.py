"""Unit tests for continuous label synthesis and target mapping."""

import numpy as np
import pytest

from src.banana_mlops.data.make_dataset import (
    CLASS_BOUNDS,
    synthesize_continuous_target,
)


def test_continuous_bounds_unripe():
    rng = np.random.default_rng(42)
    for _ in range(100):
        score = synthesize_continuous_target("unripe", rng=rng)
        low, high = CLASS_BOUNDS["unripe"]
        assert low <= score <= high, f"Score {score} out of bounds [{low}, {high}]"


def test_continuous_bounds_ripe():
    rng = np.random.default_rng(42)
    for _ in range(100):
        score = synthesize_continuous_target("ripe", rng=rng)
        low, high = CLASS_BOUNDS["ripe"]
        assert low <= score <= high, f"Score {score} out of bounds [{low}, {high}]"


def test_continuous_bounds_overripe():
    rng = np.random.default_rng(42)
    for _ in range(100):
        score = synthesize_continuous_target("overripe", rng=rng)
        low, high = CLASS_BOUNDS["overripe"]
        assert low <= score <= high, f"Score {score} out of bounds [{low}, {high}]"


def test_continuous_bounds_rotten():
    rng = np.random.default_rng(42)
    for _ in range(100):
        score = synthesize_continuous_target("rotten", rng=rng)
        low, high = CLASS_BOUNDS["rotten"]
        assert low <= score <= high, f"Score {score} out of bounds [{low}, {high}]"


def test_unknown_class_raises_value_error():
    with pytest.raises(ValueError):
        synthesize_continuous_target("invalid_banana_class")
