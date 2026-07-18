# addon-realtimestt

A downloadable [Subtitld](https://subtitld.org) add-on providing speech-to-text
with **[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)** (faster-whisper +
WebRTC/Silero VAD).

It serves the `asr.transcribe` task, so once installed it appears as a
**Transcription engine** in Subtitld — in the **Audio ▸ Recording** tab (for
Record mode) and in the **Import** panel (for whole-file transcription). During
Record mode, Subtitld cuts the microphone into per-phrase clips and hands each
one to this add-on; it transcribes and streams the text back.

## Why an add-on

RealtimeSTT pulls in faster-whisper, CTranslate2, and (optionally) PyTorch —
gigabytes of dependencies and models. Subtitld stays small and offers this as
an opt-in download instead of bundling it.

## Install (for users)

Once it's in the catalog: **Add-ons → Browse → RealtimeSTT → Install**, then
pick it as the engine in **Audio ▸ Recording**. The chosen Whisper model
downloads from Hugging Face on first use and is cached.

## Configuration

Exposed in the add-on's **Configure** dialog (delivered to the process as
`<ID>_<KEY>` environment variables, where the id is uppercased with `-`→`_`):

| Option | Default | Notes |
|---|---|---|
| Model | `small` | faster-whisper model (`tiny`…`large-v3`, `.en` variants) |
| Device | `auto` | `auto` picks CUDA when available, else CPU |
| Compute type | `default` | `int8` fastest on CPU, `float16` on GPU |
| Stream interim results | on | refining partials while decoding |

## Layout

```
manifest.json            add-on manifest (id, tasks, config_schema, …)
realtimestt_addon/       the protocol server package
  __init__.py            id / version / protocol constants
  __main__.py            hello + asr.transcribe over stdio, backed by RealtimeSTT
pyproject.toml           packaging + console entry point
pyinstaller.spec         builds dist/realtimestt-addon/
tests/test_protocol.py   protocol conformance (no model needed)
.github/workflows/       per-platform release build
```

## Develop / test

Exercise the full wire contract without downloading any model, via a fake seam:

```bash
python tests/test_protocol.py        # or: python -m pytest tests/
```

Run the add-on by hand:

```bash
pip install -e .
echo '{"id":"1","type":"asr.transcribe","params":{"audio_path":"clip.wav","language":"en","options":{}}}' \
  | python -m realtimestt_addon
```

Set `SUBTITLD_REALTIMESTT_FAKE=1` to bypass the model and echo a fixed
transcript (what the test uses).

## Build a release

```bash
pip install -e '.[build]'
pyinstaller pyinstaller.spec --noconfirm      # → dist/realtimestt-addon/
```

Tag a version to have CI build all platforms and attach the zips + sha256 to a
GitHub Release (`manifest.json` is placed at the archive root, where Subtitld's
installer reads it):

```bash
git tag v0.1.0 && git push --tags
```

## Publish to the catalog

Add [`catalog-entry.yml`](./catalog-entry.yml) to `addons.yml` in the
**Subtitld/addons-catalog** repo. The catalog CI regenerates `catalog.json`,
which Subtitld fetches from `https://subtitld.github.io/addons-catalog/catalog.json`.

## Protocol

| Direction | Frame |
|---|---|
| add-on → host | `{"type":"hello","protocol":1,"addon":"org.subtitld.realtimestt","capabilities":[{"task":"asr.transcribe","languages":[…]}]}` |
| host → add-on | `{"type":"ready","host":"<ver>"}` |
| host → add-on | `{"id","type":"asr.transcribe","params":{"audio_path","language","options"}}` |
| add-on → host | `{"id","type":"progress","value":0..1,"message"}` |
| add-on → host | `{"id","type":"partial","data":{"start","end","text","speaker"}}` |
| add-on → host | `{"id","type":"result","data":{"segments":[…]}}` |
| add-on → host | `{"id","type":"error","code","message"}` |

## Live streaming (`asr.stream`)

Besides the per-file `asr.transcribe` task, this add-on serves the live
`asr.stream` task. Subtitld keeps owning the microphone and streams 16 kHz PCM
chunks (`asr.audio`) into an open session; the add-on runs a continuous
RealtimeSTT recognizer and streams back interim (`final:false`) and committed
(`final:true`) `partial` frames, ending with a `result` on `asr.stop`. Subtitld
shows interim text separately and commits only finalized phrases to subtitles.

```
host → us   {"id":<sid>,"type":"asr.stream","params":{language,options,samplerate}}
host → us   {"id":<sid>,"type":"asr.audio","data":{"pcm":"<base64 int16>"}}   (repeated)
host → us   {"id":<sid>,"type":"asr.stop"}
us → host   {"id":<sid>,"type":"partial","data":{start,end,text,speaker,final}}
us → host   {"id":<sid>,"type":"result","data":{"segments":[…]}}
```

Test it without a model:

```bash
python tests/test_stream.py
```

## Notes & limitations

- **Timing.** Streaming segments carry the recognized text; Subtitld bounds each
  committed cue by the playhead (fed audio isn't word-timestamped). A trailing
  phrase still in progress at `asr.stop` may be dropped rather than committed.
- **PyInstaller packaging of torch/faster-whisper** is finicky and needs
  per-platform validation on real hardware; the `.spec` collects the known
  packages but expect to iterate on the first release.
- First transcription of a session loads the model (seconds to a minute for
  large models) — the manifest sets a generous `request_timeout_sec`.

## License

MIT. Bundles [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) (MIT) and
faster-whisper (MIT); Whisper model weights are MIT (OpenAI).
