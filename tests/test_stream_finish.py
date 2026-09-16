#!/usr/bin/env python3
"""Finishing a stream must not drop the phrase that was being spoken.

0.1.1 answered ``asr.stop`` with ``result`` immediately after ``rec.stop()``
and cleared the session, so the sentence ``stop()`` had just handed over for
transcription reached a loop that discarded it: the last thing said before the
user paused was always lost.

These tests drive the real stream handlers against a fake recorder that
follows RealtimeSTT's own semantics (read from RealtimeSTT/core/lifecycle.py
and transcription_api.py):

* ``text()`` returns a queued recording at once; while a phrase is being
  recorded it waits for ``stop_recording_event``; when idle it waits for speech
  to START, which ``stop()`` does not interrupt — only ``abort()`` does.
* ``stop()`` queues the current phrase and fires ``stop_recording_event``.
* ``abort()`` blocks until the pending ``text()`` acknowledges it.

    python -m pytest tests/
"""

import json
import os
import sys
import threading
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from realtimestt_addon import __main__ as addon  # noqa: E402


class FakeRecorder:
    def __init__(self, transcribe_delay=0.0):
        self.state = 'inactive'
        self.is_recording = False
        self.transcribe_delay = transcribe_delay
        self._queue = []
        self._phrase = None
        self._start = threading.Event()
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._interrupted = threading.Event()
        self.aborts = 0
        self.stops = 0
        self.shut_down = threading.Event()
        self.on_realtime_transcription_update = None

    # -- test controls -----------------------------------------------------
    def say(self, text):
        """Speech starts: a phrase is being recorded."""
        self._phrase = text
        self.is_recording = True
        self.state = 'recording'
        self._start.set()

    def finish_phrase(self):
        """The VAD ends the phrase on its own (silence after speech)."""
        self.stop()

    # -- RealtimeSTT surface -------------------------------------------------
    def has_pending_recordings(self):
        return bool(self._queue)

    def text(self):
        self._interrupt.clear()
        if not self._queue:
            if not self.is_recording:
                self.state = 'listening'
                while not self._interrupt.is_set() and not self._start.wait(0.01):
                    pass
            if self.is_recording and not self._interrupt.is_set():
                while not self._interrupt.is_set() and not self._stop.wait(0.01):
                    pass
        if self._interrupt.is_set():
            self._interrupted.set()
            return ''
        self._start.clear()
        self._stop.clear()
        if not self._queue:
            return ''
        text = self._queue.pop(0)
        self.state = 'transcribing'
        time.sleep(self.transcribe_delay)
        self.state = 'inactive'
        return text

    def stop(self):
        self.stops += 1
        self._queue.append(self._phrase or '')
        self._phrase = None
        self.is_recording = False
        self._stop.set()

    def abort(self):
        self.aborts += 1
        self._interrupt.set()
        if self.state != 'inactive':
            self._interrupted.wait(5)
        self._interrupted.clear()

    def shutdown(self):
        self.shut_down.set()


@pytest.fixture
def frames(monkeypatch):
    sent = []
    lock = threading.Lock()

    def _send(frame):
        with lock:
            sent.append(frame)

    monkeypatch.setattr(addon, '_send', _send)
    monkeypatch.setattr(addon, '_STREAM', None)
    monkeypatch.setattr(addon, '_FINISHERS', [])
    return sent


def _start(monkeypatch, sid, recorder):
    monkeypatch.setattr(addon, '_stream_recorder', lambda language: recorder)
    monkeypatch.delenv('SUBTITLD_REALTIMESTT_FAKE', raising=False)
    addon._handle_stream_start({'id': sid, 'params': {'language': 'en'}})


def _result(frames, sid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for f in list(frames):
            if f.get('id') == sid and f.get('type') == 'result':
                return f
        time.sleep(0.01)
    raise AssertionError(f'no result for {sid} within {timeout}s; got {frames}')


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('condition never became true')


def _texts(result):
    return [s['text'] for s in result['data']['segments']]


def test_phrase_in_progress_at_stop_is_kept(monkeypatch, frames):
    """The reported bug: stop mid-sentence, and that sentence must arrive."""
    rec = FakeRecorder(transcribe_delay=0.3)   # final transcription takes time
    _start(monkeypatch, 's1', rec)

    rec.say('first sentence')
    rec.finish_phrase()
    _wait_for(lambda: any(f.get('type') == 'partial' for f in frames))

    rec.say('said right before pause')        # still speaking...
    _wait_for(lambda: rec.state == 'recording')
    addon._handle_stream_stop({'id': 's1'})   # ...when the user pauses

    result = _result(frames, 's1')
    assert _texts(result) == ['first sentence', 'said right before pause']
    # The late sentence was also sent as a final partial, before the result.
    kinds = [(f['type'], f.get('data', {}).get('text')) for f in frames]
    assert kinds.index(('partial', 'said right before pause')) < kinds.index(('result', None))


def test_stop_while_idle_answers_promptly(monkeypatch, frames):
    """Nothing being spoken: stop() cannot wake the wait, abort() must — and
    the answer must not sit out the 60 s last-sentence timeout."""
    rec = FakeRecorder()
    _start(monkeypatch, 's1', rec)
    rec.say('only sentence')
    rec.finish_phrase()
    _wait_for(lambda: rec.state == 'listening')   # back to waiting for speech

    started = time.monotonic()
    addon._handle_stream_stop({'id': 's1'})
    result = _result(frames, 's1')
    assert time.monotonic() - started < 2.0
    assert _texts(result) == ['only sentence']
    assert rec.aborts >= 1
    # stop() on an idle recorder would queue a spurious empty "sentence".
    assert rec.stops == 1


def test_stop_returns_immediately_and_frees_the_session(monkeypatch, frames):
    """The frame loop must not block while the last sentence transcribes."""
    rec = FakeRecorder(transcribe_delay=1.0)
    _start(monkeypatch, 's1', rec)
    rec.say('slow to transcribe')
    _wait_for(lambda: rec.state == 'recording')

    started = time.monotonic()
    addon._handle_stream_stop({'id': 's1'})
    assert time.monotonic() - started < 0.2
    assert addon._STREAM is None
    assert _texts(_result(frames, 's1')) == ['slow to transcribe']


def test_late_sentence_never_leaks_into_the_next_session(monkeypatch, frames):
    """0.1.1's loop read the module global, which after stop belongs to the
    next session."""
    old = FakeRecorder(transcribe_delay=0.5)
    _start(monkeypatch, 's1', old)
    old.say('belongs to s1')
    _wait_for(lambda: old.state == 'recording')
    addon._handle_stream_stop({'id': 's1'})

    new = FakeRecorder()
    _start(monkeypatch, 's2', new)            # user pressed play again at once
    s2 = addon._STREAM

    assert _texts(_result(frames, 's1')) == ['belongs to s1']
    assert s2['segments'] == []
    assert not any(f.get('id') == 's2' and f.get('data', {}).get('text') == 'belongs to s1'
                   for f in frames)
    addon._handle_stream_stop({'id': 's2'})
    _result(frames, 's2')


def test_recorder_is_shut_down_after_the_session(monkeypatch, frames):
    """Each session builds a recorder with two whisper models; they used to
    accumulate for the life of the process."""
    rec = FakeRecorder()
    _start(monkeypatch, 's1', rec)
    _wait_for(lambda: rec.state == 'listening')
    addon._handle_stream_stop({'id': 's1'})
    _result(frames, 's1')
    assert rec.shut_down.wait(2.0)


def test_stale_stop_is_ignored(monkeypatch, frames):
    rec = FakeRecorder()
    _start(monkeypatch, 's1', rec)
    addon._handle_stream_stop({'id': 'not-s1'})
    assert addon._STREAM is not None and addon._STREAM['id'] == 's1'
    addon._handle_stream_stop({'id': 's1'})
    _result(frames, 's1')


def test_shutdown_finishes_a_live_stream(monkeypatch, frames):
    """Closing Subtitld mid-sentence: the host still gets that sentence."""
    import io
    rec = FakeRecorder(transcribe_delay=0.3)

    def host():
        yield json.dumps({'type': 'ready'}) + '\n'
        yield json.dumps({'id': 's1', 'type': 'asr.stream', 'params': {}}) + '\n'
        rec.say('last words')
        _wait_for(lambda: rec.state == 'recording')
        yield json.dumps({'type': 'shutdown'}) + '\n'

    monkeypatch.setattr(addon, '_stream_recorder', lambda language: rec)
    monkeypatch.delenv('SUBTITLD_REALTIMESTT_FAKE', raising=False)
    monkeypatch.setattr(sys, 'argv', ['realtimestt-addon'])
    monkeypatch.setattr(sys, 'stdin', host())
    monkeypatch.setattr(sys, 'stdout', io.StringIO())
    monkeypatch.setattr(addon, '_RECORDER', None)

    assert addon.main() == 0
    # main() returns only after the finisher answered (bounded by 5 s).
    result = [f for f in frames if f.get('id') == 's1' and f.get('type') == 'result']
    assert len(result) == 1
    assert _texts(result[0]) == ['last words']
