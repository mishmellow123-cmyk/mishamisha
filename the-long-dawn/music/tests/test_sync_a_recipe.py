"""A's first cut must not inherit the flint recording's earlier scrape.

Recipe-data checks only; no recording is loaded and no sound is rendered.
"""
import copy

import numpy as np
import pytest

import sound_flint_v3 as shared
import sound_recipes_A as recipes


def assert_cut_safe(recipe):
    points = np.asarray(recipe.get('env_after', [(-.09, 0), (.55, 0)]))
    before_cut = np.arange(-.09, -.5 / 24, 1 / 48000)
    assert np.max(np.interp(before_cut, points[:, 0], points[:, 1])) <= -120
    # The authored click and all of its following tail keep unity envelope gain.
    click_and_tail = np.arange(0, recipe['post'], 1 / 48000)
    assert np.all(np.interp(click_and_tail, points[:, 0], points[:, 1]) == 0)


def test_first_strike_mutes_only_early_scrape_without_changing_level_contract():
    recipe = copy.deepcopy(recipes.RECIPES['A.strike1'])
    assert_cut_safe(recipe)
    recipe.pop('env_after')
    assert recipe == shared.strike(1)
    assert recipes.RECIPES['A.strike2'] == shared.strike(2)
    assert recipes.RECIPES['A.strike3'] == shared.strike(3)


def test_missing_scrape_gate_fails_the_cut_check():
    with pytest.raises(AssertionError):
        assert_cut_safe(shared.strike(1))
