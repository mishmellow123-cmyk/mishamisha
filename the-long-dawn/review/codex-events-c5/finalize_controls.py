"""Re-measure the two shots whose control audit changed after the complete sequential scan.

All other event decisions and frame-set hashes come from that completed scan. The original
measurer is also run on TRAP to resolve a saved half-frame discrepancy without guessing.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import cv2

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'music/src'))
import measure_c5_events as M
cv2.setNumThreads(1)
table=json.loads(Path(M.OUT_JSON).read_text())
assert table['generator_dependencies']=={name:M.sha_file(str(ROOT/'music/src'/name)) for name in table['generator_dependencies']}
source=subprocess.check_output(['git','show','HEAD:the-long-dawn/music/src/measure_c5_events.py'],cwd=ROOT).decode()
original=types.ModuleType('original_measurer')
original.__file__=M.__file__
exec(compile(source,'HEAD:measure_c5_events.py','exec'),original.__dict__)
original.RSS_ABORT=1_500_000_000
baseline,_=original.m_trap(M.frames_root())
base=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
          events={e['id']:e['frames'] for e in baseline})
(Path(__file__).parent/'baseline_trap.json').write_text(json.dumps(base,indent=2)+'\n')
for key in ('trap','watch'):
    events,identity=M.run([key],M.frames_root())
    assert identity[key]['frame_set_sha256']==table['shots'][key]['frame_set_sha256']
    for old in [e for e in table['events'] if e['shot']==key]:
        assert old['frames']==next(e['frames'] for e in events if e['id']==old['id'])
    replacements={e['id']:e for e in events}
    table['events']=[replacements.get(e['id'],e) for e in table['events']]
    table['shots'].update(identity)
table=M.build_table(table['events'],table['shots'])
assert not M.validate_table(table)
Path(M.OUT_JSON).write_text(json.dumps(table,indent=1,default=str)+'\n')
assert M.is_current(table)
print('Original TRAP frame decisions:',base['events'])
print('Final table current and valid; only control annotations refreshed after the full scan.')
