#!/usr/bin/env python3
"""Runtime checks the fake-mode conformance tests can't see.

* ``test_self_test`` runs ``python -m realtimestt_addon --self-test`` against
  the real RealtimeSTT install: Silero VAD must load from the silero_vad
  package's ONNX model (not fall back to torch.hub). 0.1.1 shipped without
  silero-vad, and every recording failed at recorder start. CI runs the same
  flag against the frozen bundle.
* ``test_library_stdio_cannot_touch_protocol`` makes "library code" call
  ``input()`` and ``print()`` mid-request, as torch.hub's trust prompt does,
  and checks the host still gets clean frames and the next request.

    python -m pytest tests/
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(**extra):
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env.update(extra)
    return env


@pytest.mark.skipif(importlib.util.find_spec('RealtimeSTT') is None,
                    reason='RealtimeSTT not installed')
def test_self_test():
    proc = subprocess.run(
        [sys.executable, '-m', 'realtimestt_addon', '--self-test'],
        cwd=REPO_ROOT, env=_env(), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert 'silero backend=raw_onnx' in proc.stderr, proc.stderr
    assert proc.stdout == '', 'self-test must not write protocol frames'


_PROMPTING_ADDON = textwrap.dedent('''
    import sys
    from realtimestt_addon import __main__ as addon

    def prompting_transcribe(pcm, duration, emit_partial, emit_progress):
        print('library chatter on stdout')
        try:
            answer = input('Do you trust this repository (y/N)?')
        except EOFError:
            answer = '<eof>'
        return [{'start': 0.0, 'end': 0.0, 'text': 'answer=' + answer, 'speaker': 'A'}]

    addon._fake_transcribe = prompting_transcribe
    raise SystemExit(addon.main())
''')


def test_library_stdio_cannot_touch_protocol(tmp_path):
    import wave
    wav = tmp_path / 'utt.wav'
    with wave.open(str(wav), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b'\x00\x00' * 1600)

    proc = subprocess.Popen(
        [sys.executable, '-c', _PROMPTING_ADDON], cwd=REPO_ROOT,
        env=_env(SUBTITLD_REALTIMESTT_FAKE='1'),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    frames = [{'type': 'ready', 'host': 'test'}]
    for rid in ('req-1', 'req-2'):
        frames.append({'id': rid, 'type': 'asr.transcribe',
                       'params': {'audio_path': str(wav), 'language': 'en'}})
    frames.append({'type': 'shutdown'})
    # Everything is queued up front, so a stray input() would swallow req-2.
    out, err = proc.communicate(
        ''.join(json.dumps(f) + '\n' for f in frames), timeout=30)

    lines = out.splitlines()
    parsed = [json.loads(line) for line in lines]   # every line is a frame
    results = {f['id']: f['data']['segments'][0]['text']
               for f in parsed if f.get('type') == 'result'}
    assert results == {'req-1': 'answer=<eof>', 'req-2': 'answer=<eof>'}, out
    assert 'library chatter on stdout' in err
    assert proc.returncode == 0, err
