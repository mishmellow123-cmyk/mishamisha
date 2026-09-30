"""Portable D source names and watcher parity with explicitly local RGB/layer resolution."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys

import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import edl_v3 as EDL


def test_edl_export_keeps_logical_source_names_across_local_roots(monkeypatch, tmp_path):
    take = EDL.T('cutd_rgb', 0, 'exact', matte='cutd_matte', under=('same', 'cutd_under'))
    row = EDL.S('D0', 0, 1, 'fixture', 'fixture', 'fixture', '', [take])
    monkeypatch.setitem(EDL.EDL, 'D', [row])
    monkeypatch.delenv('CUTD_LOCAL_RENDERS', raising=False)
    before = AS.edl_doc('D')
    legacy = {c: AS.edl_doc(c) for c in 'ABC'}
    monkeypatch.setenv('CUTD_LOCAL_RENDERS', str(tmp_path / 'private-machine-root'))
    assert AS.chain(take, 'D') == [str(tmp_path / 'private-machine-root' / 'cutd_rgb')]
    assert AS.edl_doc('D') == before
    assert AS.edl_doc('D')['shots'][0]['takes'][0]['folders'] == ['renders/cutd_rgb']
    assert {c: AS.edl_doc(c) for c in 'ABC'} == legacy
    assert 'CUTD_LOCAL_RENDERS' in before['lookup']
    original = AS.chain
    with monkeypatch.context() as broken:
        broken.setattr(AS, 'chain', lambda take, cut, variant=None, **kw: original(take, cut, variant))
        # The old exporter depended on the local machine even though the EDL's logical take was unchanged.
        assert AS.edl_doc('D') != before


def watcher_signature(script=None):
    source = script or (EDIT / 'refresh_watch.sh').read_text()
    block = source.split("python3 - <<'EOF'\n", 1)[1].split('\nEOF', 1)[0]
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(block, '<watcher signature>', 'exec'), {})
    return output.getvalue()


@pytest.fixture
def watcher_fixture(monkeypatch, tmp_path):
    roots = [tmp_path / name for name in ('first', 'second')]
    stems = ('cutd_rgb', 'cutd_matte', 'cutd_add', 'cutd_under', 'cutd_under_half',
             'cutd_glow', 'cutd_keep', 'cutd_cover')
    for root in roots:
        for stem in stems:
            folder = root / stem
            folder.mkdir(parents=True)
            os.utime(folder, ns=(1_000_000_000, 1_000_000_000))
    take = EDL.T('cutd_rgb', 0, 'exact', matte='cutd_matte', add='cutd_add', under=('same', 'cutd_under'))
    row = EDL.S('D0', 0, 1, 'fixture', 'fixture', 'fixture', '', [take])
    monkeypatch.setitem(EDL.EDL, 'D', [row])
    monkeypatch.setitem(EDL.TRANS, 'D', [dict(f0=0, f1=1, kind='burn', glow='cutd_glow',
                                           keep='cutd_keep', cover='cutd_cover')])
    monkeypatch.setattr(AS, 'plan_shot', lambda *args: dict(kind='take', take=take, have=1, alt=0))
    monkeypatch.setattr(AS, 'provisional_sources', lambda *args: ())
    monkeypatch.setattr(AS, 'audio_signature', lambda *args: 'fixture')
    monkeypatch.setenv('FILMS', 'D')
    monkeypatch.setenv('CUTD_LOCAL_RENDERS', str(roots[0]))
    return roots, stems


@pytest.mark.parametrize('stem', ['cutd_matte', 'cutd_add', 'cutd_under', 'cutd_under_half',
                                  'cutd_glow', 'cutd_keep', 'cutd_cover'])
def test_watcher_observes_each_local_layer_directory(watcher_fixture, stem):
    roots, _ = watcher_fixture
    before = watcher_signature()
    script = (EDIT / 'refresh_watch.sh').read_text()
    broken = script.replace('AS.render_dir(x)', 'os.path.join(AS.RENDERS, x)').replace(
        'AS.render_dir(t[k])', 'os.path.join(AS.RENDERS, t[k])')
    stale_before = watcher_signature(broken)
    os.utime(roots[0] / stem, ns=(2_000_000_000, 2_000_000_000))
    assert watcher_signature() != before
    assert watcher_signature(broken) == stale_before  # previous hard-coded shared root misses the change


def test_watcher_detects_new_local_root_even_when_all_mtimes_match(watcher_fixture, monkeypatch):
    roots, _ = watcher_fixture
    first = watcher_signature()
    monkeypatch.setenv('CUTD_LOCAL_RENDERS', str(roots[1]))
    assert watcher_signature() != first


@pytest.mark.parametrize('stem,extension', [('cutd_rgb', 'png'), ('cutd_matte', 'png'),
                                           ('cutd_cover', 'png'), ('cutd_coeff', 'npz')])
def test_local_in_place_replacement_invalidates_watch_without_folder_mtime_change(
        watcher_fixture, monkeypatch, stem, extension):
    roots, _ = watcher_fixture
    folder = roots[0] / stem
    folder.mkdir(exist_ok=True)
    if stem == 'cutd_coeff':
        take = EDL.T('cutd_rgb', 0, 'exact')
        take['linear_mix'] = stem
        monkeypatch.setattr(AS, 'plan_shot', lambda *args: dict(kind='take', take=take, have=1, alt=0))
    frame = folder / f'f_00000.{extension}'
    frame.write_bytes(b'first')
    before = watcher_signature()
    directory_time = folder.stat().st_mtime_ns
    broken = (EDIT / 'refresh_watch.sh').read_text().replace(
        "('D source folders', local_sources(dirs))", "('D source folders', dirs)").replace(
        "('D transition folders', local_sources(layers))", "('D transition folders', layers)")
    stale = watcher_signature(broken)
    frame.write_bytes(b'second content')
    assert folder.stat().st_mtime_ns == directory_time
    assert watcher_signature() != before
    assert watcher_signature(broken) == stale  # folder timestamps alone cannot observe an in-place write


def test_watcher_tracks_direct_take_under_and_independent_sweep_coefficients(watcher_fixture, monkeypatch):
    roots, _ = watcher_fixture
    nested = EDL.T('cutd_under', 0, 'exact', need=(7, 7), matte='cutd_matte', add='cutd_add')
    outer = EDL.T('cutd_rgb', 0, 'exact', under=('take', nested, 7, 'D'))
    outer['linear_mix'] = 'cutd_coeff'  # coefficient-backed composition needs its under even without a matte
    monkeypatch.setattr(AS, 'plan_shot', lambda *args: dict(kind='take', take=outer, have=1, alt=0))
    folder = roots[0] / 'cutd_coeff'
    folder.mkdir()
    os.utime(folder, ns=(1_000_000_000, 1_000_000_000))
    before = watcher_signature()
    for stem in ('cutd_under', 'cutd_matte', 'cutd_add', 'cutd_coeff'):
        os.utime(roots[0] / stem, ns=(2_000_000_000, 2_000_000_000))
        assert watcher_signature() != before, stem
        os.utime(roots[0] / stem, ns=(1_000_000_000, 1_000_000_000))
        assert watcher_signature() == before
    broken = (EDIT / 'refresh_watch.sh').read_text().replace(
        "(t.get('matte') or t.get('linear_mix'))", "t.get('matte')")
    stale = watcher_signature(broken)
    os.utime(roots[0] / 'cutd_under', ns=(2_000_000_000, 2_000_000_000))
    assert watcher_signature(broken) == stale  # matte-only dependency checks miss coefficient composition
    assert watcher_signature() != before


def test_legacy_watcher_signature_ignores_local_namespace_flag(monkeypatch, tmp_path):
    take = EDL.T('existing_rgb', 0, 'exact', matte='existing_matte', under=('same', 'existing_under'))
    row = EDL.S('fixture', 0, 1, 'fixture', 'fixture', 'fixture', '', [take])
    for cut in 'ABC':
        monkeypatch.setitem(EDL.EDL, cut, [row])
        monkeypatch.setitem(EDL.TRANS, cut, [dict(f0=0, f1=1, kind='burn', cover='existing_cover')])
    monkeypatch.setattr(AS, 'plan_shot', lambda *args: dict(kind='take', take=take, have=1, alt=0))
    monkeypatch.setattr(AS, 'provisional_sources', lambda *args: ())
    monkeypatch.setattr(AS, 'audio_signature', lambda *args: 'fixture')
    monkeypatch.setenv('FILMS', 'ABC')
    monkeypatch.delenv('CUTD_LOCAL_RENDERS', raising=False)
    before = watcher_signature()
    monkeypatch.setenv('CUTD_LOCAL_RENDERS', str(tmp_path))
    assert watcher_signature() == before
