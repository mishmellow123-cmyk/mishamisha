"""D's provisional audio inventory must not claim credits from unrelated recipes.

CUTD_CREDITS_UNDER_TEST selects a deliberately broken source copy for negative controls.
"""
import importlib.util
import builtins
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(os.environ.get('CUTD_CREDITS_UNDER_TEST') or
              Path(__file__).resolve().parents[1] / 'src' / 'sound_credits_v3.py')
SPEC = importlib.util.spec_from_file_location('cutd_credits', SOURCE)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_missing_d_recipe_is_empty_without_importing_ir(monkeypatch):
    calls = []

    def missing(name):
        calls.append(name)
        raise ModuleNotFoundError(f'No module named {name!r}', name=name)

    monkeypatch.setattr(C.importlib, 'import_module', missing)
    assert C.used_refs('D') == {'D': set()}
    assert calls == ['sound_recipes_D']


def test_installed_d_recipe_is_walked_and_broken_dependency_is_not_silenced(monkeypatch):
    monkeypatch.setattr(C.importlib, 'import_module', lambda name: SimpleNamespace(
        RECIPES={'fixture': [('fs:123', {}), ('el:synthetic_01', {})]}))
    refs = C.used_refs('D')
    assert refs['D'] == {'fs:123', 'el:synthetic_01'}

    def broken(name):
        raise ModuleNotFoundError('missing dependency', name='missing_recipe_dependency')

    monkeypatch.setattr(C.importlib, 'import_module', broken)
    with pytest.raises(ModuleNotFoundError, match='missing dependency'):
        C.used_refs('D')


def test_default_recipe_inventory_remains_abc(monkeypatch):
    calls = []
    monkeypatch.setattr(C.importlib, 'import_module', lambda name: calls.append(name) or
                        SimpleNamespace(RECIPES={}))
    assert set(C.used_refs()) == {'A', 'B', 'C', 'IR'}
    assert calls == ['sound_recipes_A', 'sound_recipes_B', 'sound_recipes_C']


def test_d_credits_without_recipes_need_no_source_lookup(monkeypatch, tmp_path):
    def missing(name):
        raise ModuleNotFoundError(f'No module named {name!r}', name=name)

    monkeypatch.setattr(C.importlib, 'import_module', missing)
    monkeypatch.setattr(C.F, 'info', lambda sid: pytest.fail('stand-in credits attempted a recording lookup'))
    # These logs deliberately do not exist: they describe other films, not this stand-in.
    monkeypatch.setattr(C, 'MUSIC', str(tmp_path / 'no-other-film-logs'))
    out = tmp_path / 'D_credits.md'
    C.main('D', str(out))
    text = out.read_text()
    assert '**STAND-IN**' in text
    assert '115-bar consolidated animatic (6:23.3)' in text and 'score is pending' in text
    assert 'scaffold' not in text.lower()
    assert 'D recipe references: 0.' in text
    assert 'No D sound recipes are installed.' in text
    assert 'requires its own source credits' in text
    assert '## Freesound recordings' not in text
    # Measured from the committed cut D v2 generator (cbcc718) before D's score-only opt-in was merged; the merged
    # generator emits the same bytes (owner integration check, 30 Sep).
    assert hashlib.sha256(out.read_bytes()).hexdigest() == '9348fcb4201ee30603aa06a19214cfcd03e9feaa45ace61a67b80deabe792959'


def test_d_credits_cli_recognizes_installed_score_only_recipe(tmp_path):
    out = tmp_path / 'D_credits.md'
    result = subprocess.run([sys.executable, str(SOURCE), '--cuts', 'D', '--out', str(out)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'installed **score-only** draft recipe' in out.read_text()
    assert 'No D sound recipes are installed' not in out.read_text()
    assert '## Freesound recordings' not in out.read_text()


def test_score_only_d_neither_imports_ir_nor_looks_up_recordings(monkeypatch, tmp_path):
    recipe = SimpleNamespace(RECIPES={}, SCORE_ONLY=True)
    monkeypatch.setattr(C.importlib, 'import_module', lambda name: recipe)
    original_import = builtins.__import__

    def no_ir(name, *args, **kwargs):
        if name == 'ir_v3':
            pytest.fail('IR imported for the score-only recipe')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_ir)
    monkeypatch.setattr(C.F, 'info', lambda sid: pytest.fail('score-only inventory looked up a recording'))
    assert C.used_refs('D') == {'D': set()}
    out = tmp_path / 'score_only.md'
    C.main('D', str(out))
    assert 'D recipe references: 0.' in out.read_text()
    # Flag-off control: an installed effects recipe retains the original IR inventory.
    recipe.SCORE_ONLY = False
    with pytest.raises(pytest.fail.Exception, match='IR imported'):
        C.used_refs('D')


@pytest.mark.parametrize('field,value', [
    ('RECIPES', {'effect': [('fs:123', {})]}),
    ('EXTRA_EVENTS', [('effect', 100)]),
    ('EXTRA_BEDS', [('effect', 0, 100)]),
    ('SPACE', {'forest': 'recorded'}),
])
def test_score_only_flag_cannot_hide_effects(monkeypatch, field, value):
    recipe = SimpleNamespace(RECIPES={}, SCORE_ONLY=True)
    setattr(recipe, field, value)
    monkeypatch.setattr(C.importlib, 'import_module', lambda name: recipe)
    with pytest.raises(ValueError, match='SCORE_ONLY recipe declares effects'):
        C.used_refs('D')


@pytest.mark.parametrize('cuts,flag,expected', [
    ('ABC', False, 'e0d0743bafc7eae23759fe7e69cd6c2b2213c3e04f30b84bd1ea2eb5fc83f887'),
    ('ABC', True, 'e0d0743bafc7eae23759fe7e69cd6c2b2213c3e04f30b84bd1ea2eb5fc83f887'),
    ('D', False, 'd2c5d8c25a62e57295af573c720227a99c904d3e3b8fab138b71d7f8108f9d2f'),  # cbcc718's generator, same fixtures
])
def test_nonopted_inventory_and_output_bytes_are_unchanged(monkeypatch, tmp_path, cuts, flag, expected):
    """Hashes measured from the pre-change generator using these same fixtures."""
    monkeypatch.setattr(C.importlib, 'import_module', lambda name: SimpleNamespace(
        RECIPES={'fixture': [('fs:101', {}), ('el:fixture_01', {})]}, SCORE_ONLY=flag))
    monkeypatch.setattr(C, 'MUSIC', str(tmp_path))
    monkeypatch.setattr(C.F, 'info', lambda sid: dict(
        license='https://creativecommons.org/licenses/by/4.0/', username='fixture author', name=f'fixture {sid}'))
    monkeypatch.setitem(sys.modules, 'ir_v3', SimpleNamespace(SPACES={'fixture': {'src': '202'}}))
    monkeypatch.setitem(sys.modules, 'sound_v3', SimpleNamespace(src_path=lambda r: '/fixture/source_orig.wav'))
    expected_inventory = {cut: {'fs:101', 'el:fixture_01'} for cut in cuts}
    expected_inventory['IR'] = {'fs:202'}
    assert C.used_refs(cuts) == expected_inventory
    out = tmp_path / 'unchanged.md'
    C.main(cuts, str(out))
    assert hashlib.sha256(out.read_bytes()).hexdigest() == expected
    assert hashlib.sha256(out.read_bytes() + b'\n').hexdigest() != expected


def test_mixed_inventory_keeps_abc_irs_and_reports_d_score_only(monkeypatch, tmp_path):
    def recipe(name):
        return SimpleNamespace(RECIPES={}, SCORE_ONLY=name == 'sound_recipes_D')

    monkeypatch.setattr(C.importlib, 'import_module', recipe)
    monkeypatch.setitem(sys.modules, 'ir_v3', SimpleNamespace(SPACES={}))
    monkeypatch.setattr(C, 'MUSIC', str(tmp_path))
    assert C.used_refs('ABCD') == {'A': set(), 'B': set(), 'C': set(), 'D': set(), 'IR': set()}
    out = tmp_path / 'mixed.md'
    C.main('ABCD', str(out))
    assert 'installed **score-only** draft recipe' in out.read_text()


@pytest.mark.parametrize('cuts', ['', 'E', 'D?'])
def test_invalid_cut_does_not_silently_emit_an_empty_inventory(cuts):
    with pytest.raises(ValueError, match='cuts must contain'):
        C.used_refs(cuts)
