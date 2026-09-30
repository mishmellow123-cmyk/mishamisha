"""In-memory adoption states for mechanism tests; never rewrite the owner's live EDL."""
import ast
from pathlib import Path

import edl_v3 as EDL


EDL_PATH = Path(__file__).resolve().parents[1] / 'edl_v3.py'


def _parts(source):
    before, marker, rest = source.partition('D_NEW_TAKES = {\n')
    block, end, after = rest.partition('\n}\n')
    assert marker and end, 'explicit D adoption block must exist'
    return before + marker, block.splitlines(), end + after


def adoption_lines():
    """All literal adoption calls, whether active or commented in the live file."""
    _, lines, _ = _parts(EDL_PATH.read_text())
    calls = []
    for line in lines:
        entry = line.strip().removeprefix('# ')
        if entry.startswith("'D"):
            calls.append(entry)
    tree = ast.parse('{\n' + '\n'.join(calls) + '\n}', mode='eval')
    return eval(compile(tree, '<D adoption literals>', 'eval'), {'__builtins__': {}, 'T': EDL.T})


def commented_source():
    """A source copy with every explicit D adoption line commented."""
    before, lines, after = _parts(EDL_PATH.read_text())
    for i, line in enumerate(lines):
        if line.lstrip().startswith("'D"):
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = indent + '# ' + line.lstrip()
    return before + '\n'.join(lines) + after


def edl_namespace(adopt=()):
    """Execute an isolated EDL copy with exactly the requested literal adoption lines active."""
    wanted = set(adopt)
    unknown = wanted - set(adoption_lines())
    if unknown:
        raise ValueError(f'Unknown synthetic D adoption: {sorted(unknown)}')
    before, lines, after = _parts(commented_source())
    for i, line in enumerate(lines):
        entry = line.strip().removeprefix('# ')
        if any(entry.startswith(repr(code) + ':') for code in wanted):
            lines[i] = line.replace('# ', '', 1)
    namespace = {'__file__': str(EDL_PATH), '__name__': 'cutd_synthetic_edl'}
    exec(compile(before + '\n'.join(lines) + after, '<synthetic D adoption>', 'exec'), namespace)
    return namespace
