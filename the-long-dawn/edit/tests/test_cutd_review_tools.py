"""Cut D review entry points, with tiny metadata fixtures and no production writes.

CUTD_H9_UNDER_TEST and CUTD_PREVIEWS_UNDER_TEST select copies for mutation controls.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))


def load(name, filename, env):
    spec = importlib.util.spec_from_file_location(name, os.environ.get(env, EDIT / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H9 = load('cutd_h9', 'h9_kit.py', 'CUTD_H9_UNDER_TEST')
P = load('cutd_previews', 'previews.py', 'CUTD_PREVIEWS_UNDER_TEST')


@pytest.mark.parametrize('value, expected', [('D', 'D'), ('d, a', 'DA'), ('AC', 'AC'), ('DD', 'D')])
def test_explicit_cut_selection(value, expected):
    assert H9.cut_selection(value) == expected


@pytest.mark.parametrize('value', ['', 'DE', 'all'])
def test_unknown_cut_is_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        H9.cut_selection(value)


class EmptyPool:
    def __init__(self, workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def imap_unordered(self, fn, jobs, chunksize):
        assert jobs == []
        return iter(())


def h9_metadata_only(monkeypatch):
    calls = []
    monkeypatch.setattr(H9, 'Film', lambda cut, variant=None: calls.append((cut, variant)) or
                        SimpleNamespace(bars=0, counts=lambda: {}))
    monkeypatch.setattr(H9.EDL, 'check', lambda *args: None)
    monkeypatch.setattr(H9, 'Pool', EmptyPool)
    monkeypatch.setattr(H9, 'overview', lambda *args: None)
    monkeypatch.setattr(H9, 'summary_line', lambda *args: 'fixture')
    monkeypatch.setattr(H9.AS, 'resolve_audio_label', lambda cut: 'STAND-IN silence')
    for name in ('coverage_md', 'text_md', 'index_md'):
        monkeypatch.setattr(H9, name, lambda *args: '')
    return calls


def test_h9_cli_selects_d_and_cleans_only_its_old_sheets(monkeypatch, tmp_path):
    calls = h9_metadata_only(monkeypatch)
    d_old, a_old = tmp_path / 'D_bars_01-12.jpg', tmp_path / 'A_overview.jpg'
    d_old.write_bytes(b'stale D')
    a_old.write_bytes(b'accepted A')
    H9.main(['--cuts', 'D', '--out', str(tmp_path), '--workers', '1'])
    assert calls == [('D', None)]
    assert not d_old.exists()
    assert a_old.read_bytes() == b'accepted A'
    assert 'D' in H9.FILM


def test_h9_default_stays_abc(monkeypatch, tmp_path):
    calls = h9_metadata_only(monkeypatch)
    H9.main(['--out', str(tmp_path)])
    assert calls == [('A', None), ('B', None), ('C', None), ('A', 'codedtowers')]


def test_d_text_and_index_use_real_caption_rows(monkeypatch):
    monkeypatch.setattr(H9.AS, '_INDEX', {})
    film = H9.Film('D')
    text = H9.text_md({'D': film}, 'fixture')
    assert '## D · THE LONG DAWN (consolidated edit)' in text
    assert 'The 115-bar consolidated edit: Cormorant italic narration' in text
    assert 'scaffold' not in text.lower()
    for row in film.table:
        assert f"| {row['id']} |" in text
    index = H9.index_md({'D': film}, 'fixture', {'D': 'STAND-IN silence'}, [], [], {'D': 'fixture'})
    assert 'Selected cuts share one grid' in index
    assert 'Three short films' not in index


def test_h9_held_source_provenance_names_the_fixed_frame():
    film = H9.Film.__new__(H9.Film)
    film.shot_i = lambda frame: 0
    film.plans = [dict(kind='take', take=dict(stem='reveal_A', off=-340, hold=3799))]
    assert film.src(4140) == film.src(4159) == 'reveal_A 3799'
    film.plans[0]['take'].pop('hold')
    assert film.src(4159) == 'reveal_A 3819'  # moving-source control must be distinguishable


def preview_metadata_only(monkeypatch, tmp_path):
    delivery = tmp_path / 'delivery'
    delivery.mkdir()
    out = tmp_path / 'previews'
    out.mkdir()
    monkeypatch.setattr(P, 'OUT', str(out))
    monkeypatch.setattr(P, 'MANIFEST', str(out / '.previews.json'))
    monkeypatch.setattr(P.D, 'DELIVERY', str(delivery))
    calls = []
    monkeypatch.setattr(P, 'Film', lambda cut: calls.append(cut) or SimpleNamespace(cut=cut))
    monkeypatch.setattr(P.AS, 'resolve_audio', lambda cut: ('fixture.wav', 'STAND-IN mix'))
    monkeypatch.setattr(P, 'stretches', lambda film: [])
    for cut in 'ACD':
        (delivery / f'{cut}_master.mov').write_bytes(b'not decoded')
    return out, calls


def test_preview_default_stays_ac_and_d_is_explicit(monkeypatch, tmp_path):
    _, calls = preview_metadata_only(monkeypatch, tmp_path)
    P.main()
    assert calls == ['A', 'C']
    calls.clear()
    P.main('D')
    assert calls == ['D']


def test_d_preview_refresh_preserves_ac_files_manifest_and_readme(monkeypatch, tmp_path):
    out, calls = preview_metadata_only(monkeypatch, tmp_path)
    manifest = {f'{cut}_bars01-02_old.mp4': dict(cut=cut, key=f'{cut}-key') for cut in 'ACD'}
    for name in manifest:
        (out / name).write_bytes(name.encode())
    (out / '.previews.json').write_text(json.dumps(manifest))
    old_lines = [f'{name}: original description' for name in manifest]
    (out / 'README.txt').write_text('\n'.join(old_lines))
    P.main('D')
    assert calls == ['D']
    current = json.loads((out / '.previews.json').read_text())
    assert current == {name: row for name, row in manifest.items() if row['cut'] in 'AC'}
    for name, row in manifest.items():
        if row['cut'] in 'AC':
            assert (out / name).read_bytes() == name.encode()
            assert f'{name}: original description' in (out / 'README.txt').read_text()
        else:
            assert not (out / name).exists()


def test_d_preview_labels_recorded_master_audio_without_resolving_current_mix(monkeypatch, tmp_path):
    out, _ = preview_metadata_only(monkeypatch, tmp_path)
    master = str(tmp_path / 'delivery' / 'D_master.mov')
    label = 'STAND-IN mix borrowed.wav (fixture receipt)'
    P.D.write_audio_receipt(master, label)
    monkeypatch.setattr(P.AS, 'resolve_audio', lambda cut: pytest.fail('D resolved a different current mix'))
    monkeypatch.setattr(P, 'Film', lambda cut: SimpleNamespace(cut=cut, shots=[dict(sec='D1', name='fixture')]))
    monkeypatch.setattr(P, 'stretches', lambda film: [(0, 80, {})])
    monkeypatch.setattr(P, 'describe', lambda *args: ([0], 'fixture', []))
    monkeypatch.setattr(P, 'seg_keys', lambda *args: [{'segment'}])
    monkeypatch.setattr(P, 'master_current', lambda *args: True)
    monkeypatch.setattr(P, 'master_segments', lambda *args: ['segment'])
    monkeypatch.setattr(P, 'export', lambda mov, a, b, path: Path(path).write_bytes(b'fixture preview'))
    P.main('D')
    assert f'sound: {label}' in (out / 'README.txt').read_text()
    assert 'silence' not in (out / 'README.txt').read_text()


def test_d_preview_key_tracks_master_media_and_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'master_segments', lambda *args: ['same-picture-segment'])
    master = tmp_path / 'D_master.mov'
    master.write_bytes(b'fixture-audio-one')
    film = SimpleNamespace(cut='D')
    first = P.key(film, [0], str(master), 'STAND-IN one')
    assert P.key(film, [0], str(master), 'STAND-IN two') != first
    stamp = master.stat().st_mtime_ns
    master.write_bytes(b'fixture-audio-two-with-different-length')
    os.utime(master, ns=(stamp, stamp))
    assert P.key(film, [0], str(master), 'STAND-IN one') != first
