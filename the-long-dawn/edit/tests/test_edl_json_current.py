"""Each cut's exported EDL (edit/edl/edl_<cut>.json, the departments' copy) must be exactly what the generator emits now.

C's readiness gate already checks C (c5_readiness.check_generator_json); A and B had no check, and edl_A.json went stale
across two integrations that changed TRANS['A'] (a5e4c29, c549e41; re-exported in 44fa783). The generator's document is
compared after a JSON round trip, as the file holds it. Fix a failure with: python3 edit/assemble.py --edl
"""
import json
from pathlib import Path
import sys
from unittest import mock

import pytest

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS  # noqa: E402


def on_disk(cut):
    return json.loads((EDIT / 'edl' / f'edl_{cut}.json').read_text())


def generated(cut):
    return json.loads(json.dumps(AS.edl_doc(cut)))


def stale_keys(cut):
    doc, disk = generated(cut), on_disk(cut)
    return [k for k in sorted(set(doc) | set(disk)) if doc.get(k) != disk.get(k)]


@pytest.mark.parametrize('cut', 'ABCD')
def test_exported_edl_matches_the_generator(cut):
    assert not stale_keys(cut), f'edl_{cut}.json is stale; run python3 edit/assemble.py --edl'


def test_a_window_missing_from_the_export_is_caught():
    """the failure this test exists for: a TRANS change that nobody re-exported"""
    kept = [t for t in AS.EDL.TRANS['A'] if t['kind'] != 'edge_polish']
    assert len(kept) == len(AS.EDL.TRANS['A']) - 1
    with mock.patch.dict(AS.EDL.TRANS, A=kept):
        assert 'transitions' in stale_keys('A')
