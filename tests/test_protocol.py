#!/usr/bin/env python3
"""Protocol conformance test for the RealtimeSTT add-on.

Drives `python -m realtimestt_addon` exactly as Subtitld's AddonProcess does —
reads the `hello`, sends `ready`, issues one `asr.transcribe`, and checks the
`progress` / `partial` / `result` frames match the shapes the host parses
(modules/addons/process.py + AddonASRProvider). Uses SUBTITLD_REALTIMESTT_FAKE
so it runs in ~1s with no model download.

    python -m pytest tests/            # or:  python tests/test_protocol.py
"""

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_wav(path, seconds=1.0, sr=16000, freq=220.0):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(int(seconds * sr)):
            w.writeframesraw(struct.pack('<h', int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr))))


def run_conformance():
    fails = []
    wav = os.path.join(tempfile.mkdtemp(), 'utt.wav')
    _make_wav(wav)

    env = dict(os.environ)
    env['SUBTITLD_REALTIMESTT_FAKE'] = '1'
    env['SUBTITLD_REALTIMESTT_FAKE_TEXT'] = 'this is a test'
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')

    proc = subprocess.Popen(
        [sys.executable, '-m', 'realtimestt_addon'],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env)

    def send(frame):
        proc.stdin.write(json.dumps(frame) + '\n')
        proc.stdin.flush()

    def read():
        line = proc.stdout.readline()
        return json.loads(line) if line.strip() else None

    hello = read()
    if not hello or hello.get('type') != 'hello':
        return ['no-hello']
    if hello.get('addon') != 'org.subtitld.realtimestt':
        fails.append('bad-addon-id:' + str(hello.get('addon')))
    if hello.get('protocol') != 1:
        fails.append('bad-protocol')
    if not any(c.get('task') == 'asr.transcribe' for c in hello.get('capabilities', [])):
        fails.append('no-asr-cap')

    send({'type': 'ready', 'host': 'test'})
    send({'id': 'req-1', 'type': 'asr.transcribe',
          'params': {'audio_path': wav, 'language': 'en', 'options': {}}})

    saw_progress = saw_partial = False
    result = None
    for _ in range(50):
        frame = read()
        if frame is None:
            break
        t = frame.get('type')
        if t == 'progress':
            saw_progress = True
            if not (0.0 <= float(frame.get('value', -1)) <= 1.0):
                fails.append('bad-progress-value')
        elif t == 'partial':
            saw_partial = True
            seg = frame.get('data', {})
            if not all(k in seg for k in ('start', 'end', 'text')):
                fails.append('bad-partial-shape')
        elif t == 'result':
            result = frame.get('data', {})
            break
        elif t == 'error':
            fails.append('error:' + str(frame.get('message')))
            break

    if not saw_progress:
        fails.append('no-progress')
    if not saw_partial:
        fails.append('no-partial')
    segs = (result or {}).get('segments')
    if not (isinstance(segs, list) and segs and segs[0].get('text') == 'this is a test'):
        fails.append('bad-result-segments')
    elif not all(isinstance(segs[0].get(k), (int, float)) for k in ('start', 'end')):
        fails.append('segment-timing-not-numeric')

    send({'type': 'shutdown'})
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        fails.append('no-clean-shutdown')
    return fails


def test_protocol_conformance():
    fails = run_conformance()
    assert not fails, 'protocol conformance failures: ' + ', '.join(fails)


if __name__ == '__main__':
    problems = run_conformance()
    print('FAIL: ' + ', '.join(problems) if problems else 'PROTOCOL CONFORMANCE OK')
    raise SystemExit(1 if problems else 0)
