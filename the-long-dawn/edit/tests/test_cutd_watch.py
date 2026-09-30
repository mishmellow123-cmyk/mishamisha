"""The watcher must carry an explicitly selected D through change detection and previews."""
from pathlib import Path
import re
import subprocess

EDIT = Path(__file__).resolve().parents[1]


def changed_cuts(script, lines):
    expression = re.search(r"sed -n '(s/\^> .*?/p)'", script).group(1)
    return subprocess.run(['sed', '-n', expression], input=lines, text=True,
                          check=True, capture_output=True).stdout.splitlines()


def test_changed_cut_detector_accepts_d_and_preserves_legacy_selection():
    script = (EDIT / 'refresh_watch.sh').read_text()
    lines = '> A:old\n> B:old\n> C:old\n> D:new\n> E:invalid\n< D:previous\n'
    assert changed_cuts(script, lines) == list('ABCD')
    # Negative control: the former filter silently loses the newly selected cut.
    assert changed_cuts(script.replace('[ABCD]', '[ABC]'), lines) == list('ABC')


def test_watch_and_shell_wrapper_forward_preview_selection():
    watch = (EDIT / 'refresh_watch.sh').read_text()
    wrapper = (EDIT / 'previews.sh').read_text()
    assert 'edit/previews.sh --cuts "${FILMS:-AC}"' in watch
    assert 'python3 edit/previews.py "$@"' in wrapper
