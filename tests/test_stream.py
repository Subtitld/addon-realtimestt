#!/usr/bin/env python3
"""Streaming (asr.stream) conformance test for the RealtimeSTT add-on.

Drives `python -m realtimestt_addon` as Subtitld's AddonProcess does for a
live session: hello → asr.stream → several asr.audio → asr.stop, checking the
interim (`final:false`) and committed (`final:true`) partials and the terminal
`result`. Uses SUBTITLD_REALTIMESTT_FAKE so it needs no model.

    python tests/test_stream.py      # or via pytest
"""

import base64
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_stream():
    fails = []
    env = dict(os.environ)
    env['SUBTITLD_REALTIMESTT_FAKE'] = '1'
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
    if not any(c.get('task') == 'asr.stream' for c in hello.get('capabilities', [])):
        fails.append('no-stream-cap')

    sid = 'stream-1'
    send({'type': 'ready', 'host': 'test'})
    send({'id': sid, 'type': 'asr.stream',
          'params': {'language': 'en', 'options': {}, 'samplerate': 16000}})

    # Feed 6 chunks of (silent) PCM; fake commits a "sentence" every 3rd.
    pcm_b64 = base64.b64encode(b'\x00\x00' * 1600).decode('ascii')  # 100 ms
    for _ in range(6):
        send({'id': sid, 'type': 'asr.audio', 'data': {'pcm': pcm_b64}})

    send({'id': sid, 'type': 'asr.stop'})

    interim = finals = 0
    result = None
    for _ in range(60):
        frame = read()
        if frame is None:
            break
        if frame.get('id') != sid:
            fails.append('wrong-id')
            continue
        t = frame.get('type')
        if t == 'partial':
            d = frame.get('data', {})
            if d.get('final'):
                finals += 1
            else:
                interim += 1
            if 'text' not in d:
                fails.append('partial-no-text')
        elif t == 'result':
            result = frame.get('data', {})
            break
        elif t == 'error':
            fails.append('error:' + str(frame.get('message')))
            break

    if interim < 1:
        fails.append('no-interim')
    if finals < 1:
        fails.append('no-final')
    segs = (result or {}).get('segments')
    if not (isinstance(segs, list) and len(segs) >= 2):   # 6 chunks / 3 → 2 sentences
        fails.append('bad-final-segments:' + str(segs))

    send({'type': 'shutdown'})
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        fails.append('no-clean-shutdown')
    print(f'interim={interim} finals={finals} result_segments={len(segs) if isinstance(segs, list) else None}')
    return fails


def test_stream_conformance():
    fails = run_stream()
    assert not fails, 'stream conformance failures: ' + ', '.join(fails)


if __name__ == '__main__':
    problems = run_stream()
    print('FAIL: ' + ', '.join(problems) if problems else 'STREAM CONFORMANCE OK')
    raise SystemExit(1 if problems else 0)
