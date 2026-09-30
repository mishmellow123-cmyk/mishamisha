"""D's provisional audio inventory must not claim credits from unrelated recipes.

CUTD_CREDITS_UNDER_TEST selects a deliberately broken source copy for negative controls.
"""
import importlib.util
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
    assert 'D recipe references: 0.' in text
    assert 'No D sound recipes are installed.' in text
    assert 'requires its own source credits' in text
    assert '## Freesound recordings' not in text


def test_d_credits_cli_runs_without_a_d_score(tmp_path):
    out = tmp_path / 'D_credits.md'
    result = subprocess.run([sys.executable, str(SOURCE), '--cuts', 'D', '--out', str(out)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'STAND-IN' in out.read_text()


@pytest.mark.parametrize('cuts', ['', 'E', 'D?'])
def test_invalid_cut_does_not_silently_emit_an_empty_inventory(cuts):
    with pytest.raises(ValueError, match='cuts must contain'):
        C.used_refs(cuts)
