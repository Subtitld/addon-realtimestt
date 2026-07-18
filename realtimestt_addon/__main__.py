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
import json
import os
import sys
import threading
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


def _send(frame: dict) -> None:
    line = json.dumps(frame, separators=(',', ':')) + '\n'
    with _SEND_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


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
    )


def _handle_stream_start(frame: dict) -> None:
    global _STREAM
    sid = frame.get('id')
    params = frame.get('params', {}) or {}
    language = str(params.get('language', '') or '')
    fake = os.environ.get('SUBTITLD_REALTIMESTT_FAKE') == '1'
    _STREAM = {'id': sid, 'fake': fake, 'segments': [], 'audio_count': 0,
               'stop': threading.Event(), 'recorder': None, 'thread': None}

    if fake:
        return   # interim/final are driven by incoming asr.audio frames

    try:
        recorder = _stream_recorder(language)
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

    def _on_realtime(text):
        t = (text or '').strip()
        if t and _STREAM is not None and _STREAM['id'] == sid:
            _send({'id': sid, 'type': 'partial',
                   'data': {'start': 0.0, 'end': 0.0, 'text': t, 'speaker': 'A', 'final': False}})
    try:
        recorder.on_realtime_transcription_update = _on_realtime
    except Exception:
        pass

    def _sentence_loop():
        # recorder.text() blocks until a full sentence; loop until stopped.
        while _STREAM is not None and not _STREAM['stop'].is_set():
            try:
                sentence = recorder.text()
            except Exception:
                break
            s = (sentence or '').strip()
            if not s or _STREAM is None:
                continue
            seg = {'start': 0.0, 'end': 0.0, 'text': s, 'speaker': 'A'}
            _STREAM['segments'].append(seg)
            _send({'id': sid, 'type': 'partial', 'data': {**seg, 'final': True}})

    th = threading.Thread(target=_sentence_loop, daemon=True)
    _STREAM['thread'] = th
    th.start()


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


def _handle_stream_stop(frame: dict) -> None:
    global _STREAM
    st = _STREAM
    if not st or st['id'] != frame.get('id'):
        return
    sid = st['id']
    st['stop'].set()
    rec = st.get('recorder')
    if rec is not None:
        try:
            rec.stop()      # unblocks a pending text() so the loop can exit
        except Exception:
            pass
    _send({'id': sid, 'type': 'result', 'data': {'segments': st['segments']}})
    _STREAM = None


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


def main() -> int:
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

    for raw in sys.stdin:
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
