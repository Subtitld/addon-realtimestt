#!/usr/bin/env python3
"""Subtitld ASR add-on — RealtimeSTT (KoljaB/RealtimeSTT) entry point.

Serves the ``asr.transcribe`` task using RealtimeSTT
(https://github.com/KoljaB/RealtimeSTT), which wraps faster-whisper with
WebRTC/Silero VAD. Built for Subtitld's **Record mode**: the host cuts the
microphone into per-phrase WAVs and hands each one here; we transcribe it and
stream the text back. It also transcribes whole-file audio from the Import
panel.

Protocol (JSON, one object per line, over stdio):

  us -> host   {"type":"hello","protocol":1,"addon":"realtimestt",...}
  host -> us   {"type":"ready","host":"<ver>"}
  host -> us   {"id","type":"asr.transcribe","params":{audio_path,language,options}}
  us -> host   {"id","type":"progress","value":0..1,"message":str}
  us -> host   {"id","type":"partial","data":{start,end,text,speaker}}   (repeatable)
  us -> host   {"id","type":"result","data":{"segments":[{start,end,text,speaker},...]}}
  us -> host   {"id","type":"error","code":str,"message":str}
  host -> us   {"type":"cancel","target":<id>} | {"type":"shutdown"}

Configuration arrives as environment variables. Subtitld converts each
``config_schema`` option into ``<ID>_<KEY>`` where ``ID`` is the manifest id
uppercased with ``-`` → ``_``, i.e. ``REALTIMESTT_``. We derive that prefix
from ADDON_ID so the two never drift:

  <PREFIX>MODEL         faster-whisper model (tiny|base|small|medium|large-v3|.en)  default: small
  <PREFIX>DEVICE        cuda | cpu | auto      default: auto
  <PREFIX>COMPUTE_TYPE  default | int8 | float16 | ...  default: default
  <PREFIX>REALTIME      1 to stream refining interim partials (default 1)

Set ``SUBTITLD_REALTIMESTT_FAKE=1`` to bypass the model and echo a
deterministic transcript — used by the protocol conformance test so the wire
contract can be verified without downloading a model.
"""

from __future__ import annotations

import base64
import collections
import json
import multiprocessing
import os
import sys
import threading
import time
import wave

try:
    import audioop  # stdlib < 3.13; only needed to convert odd input formats
except ImportError:  # pragma: no cover - Python 3.13+ dropped audioop
    audioop = None

from realtimestt_addon import ADDON_ID, PROTOCOL_VERSION, __version__

# Host env-var convention: `addon_id.replace('-', '_').upper() + '_'` (dots kept).
_ENV_PREFIX = ADDON_ID.replace('-', '_').upper() + '_'

# Languages faster-whisper / RealtimeSTT can handle. The host uses this only to
# surface the engine under a language filter.
LANGUAGES = [
    'en', 'pt', 'pt-br', 'es', 'fr', 'de', 'it', 'nl', 'ru', 'pl', 'uk', 'tr',
    'ar', 'zh', 'ja', 'ko', 'hi', 'id', 'sv', 'fi', 'da', 'no', 'cs', 'el',
    'he', 'ro', 'hu', 'ca', 'th', 'vi',
]


_SEND_LOCK = threading.Lock()   # streaming emits from a worker thread too

# The JSON protocol owns the process's real stdin/stdout. main() hands library
# code harmless stand-ins, so nothing else can read a host frame or write into
# the frame stream. (torch.hub's trust prompt -- RealtimeSTT's last-resort
# Silero VAD loader -- calls input(): it printed its question in front of our
# next frame and consumed an asr.audio frame as the answer.)
_PROTO_IN = sys.stdin
_PROTO_OUT = sys.stdout


def _claim_protocol_stdio() -> None:
    global _PROTO_IN, _PROTO_OUT
    _PROTO_IN, _PROTO_OUT = sys.stdin, sys.stdout
    sys.stdin = open(os.devnull, encoding='utf-8')   # stray input() -> EOFError
    sys.stdout = sys.stderr                          # stray print() -> the log


def _send(frame: dict) -> None:
    line = json.dumps(frame, separators=(',', ':')) + '\n'
    with _SEND_LOCK:
        _PROTO_OUT.write(line)
        _PROTO_OUT.flush()


def _log(msg: str) -> None:
    sys.stderr.write(f'[realtimestt] {msg}\n')
    sys.stderr.flush()


def _env(name: str, default: str = '') -> str:
    return os.environ.get(_ENV_PREFIX + name, default).strip()


# --------------------------------------------------------------------------- #
# Audio                                                                        #
# --------------------------------------------------------------------------- #
def _read_wav_16k_mono(path: str) -> tuple[bytes, float]:
    """Return (int16 PCM bytes at 16 kHz mono, duration_seconds)."""
    with wave.open(path, 'rb') as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    # Subtitld feeds 16 kHz mono int16 already, so conversion is normally a
    # no-op. audioop only gets used (and is only required) for other formats.
    needs_convert = width != 2 or channels > 1 or rate != 16000
    if needs_convert:
        if audioop is None:
            raise RuntimeError(
                'input is not 16kHz mono int16 and audioop is unavailable '
                '(Python 3.13+). Feed 16kHz mono PCM16 WAV.')
        if width != 2:
            frames = audioop.lin2lin(frames, width, 2)
            width = 2
        if channels > 1:
            frames = audioop.tomono(frames, width, 0.5, 0.5)
        if rate != 16000:
            frames, _ = audioop.ratecv(frames, width, 1, rate, 16000, None)
            rate = 16000
    duration = (len(frames) / 2) / 16000.0
    return frames, duration


# --------------------------------------------------------------------------- #
# Transcription back-ends                                                      #
# --------------------------------------------------------------------------- #
_RECORDER = None  # lazily-created RealtimeSTT recorder, reused across requests


def _fake_transcribe(pcm: bytes, duration: float, emit_partial, emit_progress):
    """Deterministic stand-in used for protocol tests (no model needed)."""
    emit_progress(0.5, 'transcribing (fake)')
    text = os.environ.get('SUBTITLD_REALTIMESTT_FAKE_TEXT', 'hello world')
    emit_partial({'start': 0.0, 'end': duration, 'text': text, 'speaker': 'A'})
    return [{'start': 0.0, 'end': round(duration, 3), 'text': text, 'speaker': 'A'}]


def _get_recorder():
    """Create (once) a RealtimeSTT recorder fed from external audio."""
    global _RECORDER
    if _RECORDER is not None:
        return _RECORDER
    from RealtimeSTT import AudioToTextRecorder

    device = _env('DEVICE', 'auto') or 'auto'
    if device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
    compute_type = _env('COMPUTE_TYPE', 'default') or 'default'
    model = _env('MODEL', 'small') or 'small'
    realtime = _env('REALTIME', '1') != '0'

    _log(f'loading RealtimeSTT model={model} device={device} compute={compute_type}')
    _RECORDER = AudioToTextRecorder(
        model=model,
        device=device,
        compute_type=compute_type,
        use_microphone=False,          # host feeds us audio; we never open a mic
        spinner=False,
        enable_realtime_transcription=realtime,
        realtime_model_type=model,
        level=0,
        no_log_file=True,
    )
    return _RECORDER


def _realtimestt_transcribe(pcm: bytes, duration: float, language: str,
                            emit_partial, emit_progress):
    """Transcribe one PCM buffer with RealtimeSTT. Returns a segment list."""
    recorder = _get_recorder()
    forced = _env('LANGUAGE') or language or ''
    try:
        recorder.language = forced.split('-')[0] if forced else ''
    except Exception:
        pass

    def _on_realtime(text):
        t = (text or '').strip()
        if t:
            emit_partial({'start': 0.0, 'end': duration, 'text': t, 'speaker': 'A'})
    try:
        recorder.on_realtime_transcription_update = _on_realtime
    except Exception:
        pass

    emit_progress(0.15, 'feeding audio')
    chunk = 3200 * 2  # 100 ms of 16 kHz int16
    for off in range(0, len(pcm), chunk):
        recorder.feed_audio(pcm[off:off + chunk], original_sample_rate=16000)
    # A tail of silence nudges the VAD to finalize the utterance.
    recorder.feed_audio(b'\x00\x00' * 3200, original_sample_rate=16000)

    emit_progress(0.75, 'decoding')
    text = (recorder.text() or '').strip()
    if not text:
        return []
    return [{'start': 0.0, 'end': round(duration, 3), 'text': text, 'speaker': 'A'}]


# --------------------------------------------------------------------------- #
# Live streaming (asr.stream / asr.audio / asr.stop)                           #
# --------------------------------------------------------------------------- #
_STREAM = None  # current session state, or None

# Finishing a stream means waiting for the phrase that was being spoken when
# the host said stop: stop() hands it to RealtimeSTT for a final transcription,
# which takes a moment on CPU. Generous, because giving up drops that phrase.
_FINAL_SENTENCE_TIMEOUT = 60.0
_FINISHERS = []   # sessions still collecting their last sentence
_STREAM_RATE = 16000   # the host streams 16 kHz mono int16
# Audio the host sent before stopping may still be queued inside RealtimeSTT
# (a slow machine, or frames that arrived while models were loading).
_DRAIN_TIMEOUT = 30.0
# How long a new session waits for the previous one to finish, so it can
# take over its warm recorder instead of loading two models again.
_REUSE_WAIT = 10.0
_ARM_TIMEOUT = 2.0
_IDLE_LOCK = threading.Lock()
_IDLE = {'recorder': None, 'language': None}   # a finished session's recorder


class _StreamClock:
    """Where in the stream each recording RealtimeSTT transcribes lies.

    RealtimeSTT's finals are bare text. Without timing the host has to guess
    which of its own phrase cuts a sentence belongs to, and a guess goes wrong
    as soon as the engine lags behind the speaker. So we stamp each final:

    * every chunk the recording worker takes is counted (`on_recorded_chunk`
      runs in that worker, before the chunk is examined), so the count is the
      stream position the worker has reached;
    * when a recording stops, `last_frames` holds exactly the audio queued
      for transcription (pre-roll included), so it began that long before;
    * `text()` transcribes queued recordings in order, one per call, so the
      spans are consumed in the same order.

    Stamps are stream-relative seconds. A stop that followed trailing silence
    (the VAD's decision) has that silence taken off the end.
    """

    def __init__(self, recorder):
        self._rec = recorder
        self._processed = 0
        self._spans = collections.deque()
        self._lock = threading.Lock()
        self.manual_stop = False

    def attach(self):
        try:
            self._rec.on_recorded_chunk = self._on_chunk
            self._rec.on_recording_stop = self._on_stop
        except Exception:
            pass

    def _on_chunk(self, data):
        self._processed += len(data) // 2

    def _on_stop(self):
        try:
            frames = getattr(self._rec, 'last_frames', None)
            if not frames:
                return   # nothing was queued, so no text() will answer for it
            total = sum(len(f) for f in frames) // 2
            end = self._processed
            start = max(0, end - total)
            if not self.manual_stop:
                silence = float(getattr(self._rec, 'post_speech_silence_duration', 0) or 0)
                end = max(start, end - int(silence * _STREAM_RATE))
            with self._lock:
                self._spans.append((round(start / _STREAM_RATE, 3),
                                    round(end / _STREAM_RATE, 3)))
        except Exception as exc:
            _log(f'stream clock: {exc}')

    def take(self):
        with self._lock:
            return self._spans.popleft() if self._spans else None


def _stream_recorder(language: str):
    """A RealtimeSTT recorder wired for continuous, interim-emitting use."""
    from RealtimeSTT import AudioToTextRecorder
    device = _env('DEVICE', 'auto') or 'auto'
    if device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
    model = _env('MODEL', 'small') or 'small'
    forced = _env('LANGUAGE') or language or ''
    return AudioToTextRecorder(
        model=model,
        device=device,
        compute_type=_env('COMPUTE_TYPE', 'default') or 'default',
        language=forced.split('-')[0] if forced else '',
        use_microphone=False,
        spinner=False,
        enable_realtime_transcription=True,
        realtime_model_type=model,
        level=0,
        no_log_file=True,
        # Dropping a backlog would lose speech AND desynchronise the stream
        # clock; a transcription tool would rather lag.
        handle_buffer_overflow=False,
    )


def _take_idle_recorder(language: str):
    """The previous session's recorder, when it can serve this one.

    Loading a recorder means loading two whisper models: seconds on a fast
    machine, far longer on a slow one, for every press of Play. A session
    that is still finishing is waited for briefly (its last sentence is
    usually a second or two away); the audio the host sends meanwhile just
    queues on stdin, and is fed when we get to it.
    """
    deadline = time.monotonic() + _REUSE_WAIT
    for finisher in list(_FINISHERS):
        finisher.join(timeout=max(0.0, deadline - time.monotonic()))
    with _IDLE_LOCK:
        recorder, parked_for = _IDLE['recorder'], _IDLE['language']
        _IDLE.update(recorder=None, language=None)
    if recorder is None:
        return None
    if parked_for != language:
        threading.Thread(target=_shutdown_quietly, args=(recorder,), daemon=True).start()
        return None
    try:
        # Nothing of the last take may leak into this one's pre-roll.
        recorder.clear_audio_queue()
    except Exception:
        pass
    _log('reusing the loaded recorder')
    return recorder


def _park_or_shutdown(st: dict, recorder) -> None:
    """Keep a cleanly finished recorder for the next session; shut down
    anything else (each holds two whisper models)."""
    if st.get('clean'):
        with _IDLE_LOCK:
            if _IDLE['recorder'] is None:
                _IDLE.update(recorder=recorder, language=st.get('language'))
                return
    threading.Thread(target=_shutdown_quietly, args=(recorder,), daemon=True).start()


def _drain_input(recorder) -> None:
    """Wait until RealtimeSTT has taken every sample already fed."""
    try:
        recorder.flush_audio_input()
    except Exception:
        pass
    try:
        if not recorder.drain_audio_input(timeout=_DRAIN_TIMEOUT):
            _log('audio still queued after the drain timeout')
    except Exception:
        pass


def _handle_stream_start(frame: dict) -> None:
    global _STREAM
    sid = frame.get('id')
    params = frame.get('params', {}) or {}
    language = str(params.get('language', '') or '')
    fake = os.environ.get('SUBTITLD_REALTIMESTT_FAKE') == '1'
    st = {'id': sid, 'fake': fake, 'segments': [], 'audio_count': 0,
          'stopping': threading.Event(), 'recorder': None, 'thread': None,
          'clock': None, 'language': language, 'clean': False}
    _STREAM = st

    if fake:
        return   # interim/final are driven by incoming asr.audio frames

    try:
        recorder = _take_idle_recorder(language) or _stream_recorder(language)
    except ImportError as exc:
        _send({'id': sid, 'type': 'error', 'code': 'not_installed',
               'message': f'RealtimeSTT unavailable: {exc}'})
        _STREAM = None
        return
    except Exception as exc:
        _send({'id': sid, 'type': 'error', 'code': 'internal',
               'message': f'could not start stream: {exc}'})
        _STREAM = None
        return
    _STREAM['recorder'] = recorder
    clock = _StreamClock(recorder)
    clock.attach()
    st['clock'] = clock

    def _on_realtime(text):
        # Interims only while live: after stop they would describe a phrase
        # whose final text is already on its way.
        t = (text or '').strip()
        if t and not st['stopping'].is_set():
            _send({'id': sid, 'type': 'partial',
                   'data': {'start': 0.0, 'end': 0.0, 'text': t, 'speaker': 'A', 'final': False}})
    try:
        recorder.on_realtime_transcription_update = _on_realtime
    except Exception:
        pass

    def _sentence_loop():
        # recorder.text() blocks until a full sentence. Loop on THIS session's
        # state, never the module global: after stop the global belongs to the
        # next session, and a late sentence must not land in its list.
        while True:
            try:
                sentence = recorder.text()
            except Exception:
                break
            # An interrupted text() consumed nothing; any other return, even
            # an empty one, used up one queued recording and its span.
            interrupted = False
            try:
                interrupted = recorder.interrupt_stop_event.is_set()
            except Exception:
                pass
            span = None if interrupted else clock.take()
            text = (sentence or '').strip()
            if text:
                start, end = span if span else (0.0, 0.0)
                seg = {'start': start, 'end': end, 'text': text, 'speaker': 'A'}
                st['segments'].append(seg)
                _send({'id': sid, 'type': 'partial', 'data': {**seg, 'final': True}})
            # Once stopping, drain what stop() queued and then leave, rather
            # than calling text() again and waiting for speech that won't come.
            if st['stopping'].is_set() and not _has_pending(recorder):
                break

    th = threading.Thread(target=_sentence_loop, daemon=True)
    _STREAM['thread'] = th
    th.start()
    # RealtimeSTT only starts a recording on voice once text() has armed it.
    # Audio that queued up meanwhile (models loading, the last take
    # finishing) is fed in a burst, and speech at its head would be lost if
    # it got there first.
    _wait_until_listening(recorder, _ARM_TIMEOUT)


def _wait_until_listening(recorder, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if recorder.is_recording or recorder.start_recording_on_voice_activity:
                return
        except Exception:
            return
        time.sleep(0.005)
    _log('recorder did not start listening in time')


def _handle_stream_audio(frame: dict) -> None:
    st = _STREAM
    if not st or st['id'] != frame.get('id'):
        return
    data = frame.get('data', {}) or {}
    pcm = base64.b64decode(data['pcm']) if data.get('pcm') else b''

    if st['fake']:
        st['audio_count'] += 1
        n = st['audio_count']
        _send({'id': st['id'], 'type': 'partial',
               'data': {'start': 0.0, 'end': 0.0, 'text': f'interim {n}', 'speaker': 'A', 'final': False}})
        if n % 3 == 0:   # commit a "sentence" every 3rd chunk
            seg = {'start': 0.0, 'end': 0.0, 'text': f'sentence {len(st["segments"]) + 1}', 'speaker': 'A'}
            st['segments'].append(seg)
            _send({'id': st['id'], 'type': 'partial', 'data': {**seg, 'final': True}})
        return

    rec = st.get('recorder')
    if rec is not None and pcm:
        try:
            rec.feed_audio(pcm, original_sample_rate=16000)
        except Exception as exc:
            _log(f'feed_audio failed: {exc}')


def _has_pending(recorder) -> bool:
    try:
        return bool(recorder.has_pending_recordings())
    except Exception:
        return False


def _handle_stream_stop(frame: dict) -> None:
    """Finish a stream WITHOUT dropping the phrase in progress.

    This used to send the result right after rec.stop() and clear the session,
    so the sentence stop() had just handed over for transcription arrived to a
    loop that discarded it: the last thing said before pause was always lost.
    Finishing now runs on its own thread, so the frame loop stays free and a
    new stream can start immediately.
    """
    global _STREAM
    st = _STREAM
    if not st or st['id'] != frame.get('id'):
        return
    _STREAM = None
    t = threading.Thread(target=_finish_stream, args=(st,), daemon=True,
                         name=f'realtimestt-finish-{st["id"]}')
    _FINISHERS[:] = [f for f in _FINISHERS if f.is_alive()]
    _FINISHERS.append(t)
    t.start()


def _abort_quietly(recorder) -> None:
    """abort() blocks until a pending text() acknowledges it, so it must not
    run on a thread that has to finish."""
    def _go():
        try:
            recorder.abort()
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _finish_stream(st: dict) -> None:
    rec = st.get('recorder')
    loop = st.get('thread')
    try:
        if rec is not None and loop is not None:
            # Decide on what RealtimeSTT has actually processed: audio fed
            # just before stop may still be queued, and an idle-looking
            # recorder would be aborted with that speech unheard.
            _drain_input(rec)
            # Only stop an ACTIVE recording. stop() on an idle recorder still
            # queues its frame buffer, which would be transcribed as a
            # spurious final sentence.
            try:
                if rec.is_recording:
                    # Cut mid-speech: no trailing silence to take off.
                    if st.get('clock') is not None:
                        st['clock'].manual_stop = True
                    rec.stop()
            except Exception as exc:
                _log(f'stop failed: {exc}')
            st['stopping'].set()

            deadline = time.monotonic() + _FINAL_SENTENCE_TIMEOUT
            aborted = False
            while loop.is_alive() and time.monotonic() < deadline:
                # Idle and waiting for speech: nothing left to collect, and
                # stop() does not wake that wait — only abort() does.
                if (not aborted and getattr(rec, 'state', '') == 'listening'
                        and not rec.is_recording and not _has_pending(rec)):
                    _abort_quietly(rec)
                    aborted = True
                loop.join(timeout=0.05)
            if loop.is_alive():
                _log('last sentence did not arrive in time; answering without it')
                _abort_quietly(rec)
            else:
                st['clean'] = not _has_pending(rec)
    finally:
        _send({'id': st['id'], 'type': 'result',
               'data': {'segments': list(st['segments'])}})
        if rec is not None:
            # Kept for the next session, or shut down: never left behind.
            _park_or_shutdown(st, rec)


def _shutdown_quietly(recorder) -> None:
    try:
        recorder.shutdown()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Request handling                                                             #
# --------------------------------------------------------------------------- #
def _handle_transcribe(frame: dict) -> None:
    req_id = frame.get('id')
    params = frame.get('params', {}) or {}
    audio_path = params.get('audio_path')
    language = str(params.get('language', '') or '')

    if not audio_path or not os.path.isfile(audio_path):
        _send({'id': req_id, 'type': 'error', 'code': 'bad_params',
               'message': f'audio_path missing or not found: {audio_path!r}'})
        return

    def emit_partial(seg):
        _send({'id': req_id, 'type': 'partial', 'data': seg})

    def emit_progress(value, message):
        _send({'id': req_id, 'type': 'progress', 'value': float(value),
               'message': str(message)})

    try:
        pcm, duration = _read_wav_16k_mono(audio_path)
    except Exception as exc:
        _send({'id': req_id, 'type': 'error', 'code': 'bad_audio',
               'message': f'could not read audio: {exc}'})
        return

    try:
        if os.environ.get('SUBTITLD_REALTIMESTT_FAKE') == '1':
            segments = _fake_transcribe(pcm, duration, emit_partial, emit_progress)
        else:
            segments = _realtimestt_transcribe(
                pcm, duration, language, emit_partial, emit_progress)
    except ImportError as exc:
        _send({'id': req_id, 'type': 'error', 'code': 'not_installed',
               'message': f'RealtimeSTT unavailable: {exc}'})
        return
    except Exception as exc:
        _send({'id': req_id, 'type': 'error', 'code': 'internal',
               'message': f'transcription failed: {exc}'})
        return

    _send({'id': req_id, 'type': 'result', 'data': {'segments': segments}})


def _self_test() -> int:
    """`realtimestt-addon --self-test`: check a (frozen) build offline.

    Verifies what a missing package or data file would otherwise only break
    mid-recording: Silero VAD loads from the bundled silero_vad ONNX model
    (no torch.hub download/prompt), and multiprocessing helpers start as
    helpers rather than as a second copy of the add-on. Needs no model.
    """
    failures = []
    try:
        # The whole engine module, as a stream session imports it: a missing
        # or broken dependency anywhere in it fails here, not mid-recording.
        from RealtimeSTT import AudioToTextRecorder  # noqa: F401
        import faster_whisper  # noqa: F401
    except Exception as exc:
        failures.append(f'RealtimeSTT import failed: {exc!r}')
    try:
        import numpy as np
        from RealtimeSTT.core.silero_vad import create_silero_vad_model
        vad = create_silero_vad_model(backend='auto')
        backend = str(getattr(vad, 'backend', '?'))
        prob = float(vad(np.zeros(512, dtype=np.float32), 16000))
        _log(f'self-test: silero backend={backend} p(silence)={prob:.3f}')
        if not backend.startswith('raw_onnx'):
            failures.append(f'silero backend is {backend}, expected raw_onnx*')
    except Exception as exc:
        failures.append(f'silero VAD failed: {exc!r}')

    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    child = ctx.Process(target=queue.put, args=('child-ok',))
    child.start()
    try:
        got = queue.get(timeout=60)
    except Exception:
        got = None
    child.join(10)
    _log(f'self-test: spawn child returned {got!r}')
    if got != 'child-ok':
        failures.append('spawned child did not run (multiprocessing.freeze_support?)')

    for failure in failures:
        _log(f'self-test FAILED: {failure}')
    if not failures:
        _log('self-test OK')
    return 1 if failures else 0


def main() -> int:
    # Frozen builds re-launch this executable for multiprocessing helpers
    # (RealtimeSTT's mp.Event()s start the resource tracker under 'spawn';
    # on Windows/macOS its transcription worker is an mp.Process). Without
    # this, each helper ran *this* main loop instead: a second `hello`, and a
    # second reader competing for the host's stdin. No-op when not frozen.
    multiprocessing.freeze_support()
    _claim_protocol_stdio()
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _send({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': __version__,
        'capabilities': [
            {'task': 'asr.transcribe', 'languages': LANGUAGES},
            {'task': 'asr.stream', 'languages': LANGUAGES},
        ],
    })

    for raw in _PROTO_IN:
        line = raw.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(f'bad frame: {exc}')
            continue

        ftype = frame.get('type')
        if ftype == 'asr.transcribe':
            _handle_transcribe(frame)
        elif ftype == 'asr.stream':
            _handle_stream_start(frame)
        elif ftype == 'asr.audio':
            _handle_stream_audio(frame)
        elif ftype == 'asr.stop':
            _handle_stream_stop(frame)
        elif ftype == 'ready':
            continue
        elif ftype == 'cancel':
            continue  # per-utterance work is short; nothing to abort
        elif ftype == 'shutdown':
            _log('shutdown received')
            # A live stream is finished like a stop, and every session still
            # collecting its last sentence gets a bounded moment to answer.
            live = _STREAM
            if live is not None:
                _handle_stream_stop({'id': live['id']})
            deadline = time.monotonic() + 5.0
            for finisher in list(_FINISHERS):
                finisher.join(timeout=max(0.0, deadline - time.monotonic()))
            with _IDLE_LOCK:
                idle = _IDLE['recorder']
                _IDLE.update(recorder=None, language=None)
            if idle is not None:
                _shutdown_quietly(idle)
            try:
                if _RECORDER is not None:
                    _RECORDER.shutdown()
            except Exception:
                pass
            return 0
        else:
            _log(f'unknown frame type: {ftype!r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
