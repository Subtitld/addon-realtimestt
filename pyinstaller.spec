# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld RealtimeSTT add-on.

Produces `dist/realtimestt-addon/` — the launcher binary
(`realtimestt-addon`, `.exe` on Windows) plus RealtimeSTT / faster-whisper /
CTranslate2 and their runtime data. Communication is over stdio, so it's a
console app with no window.

    pip install -e '.[build]'
    pyinstaller pyinstaller.spec --noconfirm

The release workflow zips `dist/realtimestt-addon/` together with
`manifest.json`, `LICENSE`, and `README.md` — with the manifest at the archive
root, which is where Subtitld's installer reads it from.
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

SPEC_ROOT = Path(SPECPATH).resolve()

# RealtimeSTT / faster-whisper load native libs and data files lazily; collect
# their packages wholesale so nothing is missing at runtime.
datas, binaries, hiddenimports = [], [], []
for pkg in ('RealtimeSTT', 'faster_whisper', 'ctranslate2', 'webrtcvad',
            'tokenizers', 'onnxruntime', 'silero_vad'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # Optional/transitive packages may not import on every platform;
        # PyInstaller resolves what it can.
        pass

a = Analysis(
    [str(SPEC_ROOT / 'realtimestt_addon' / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # No GUI toolkit is used — keep the bundle lean.
        'PySide6', 'PyQt6', 'PyQt5',
        'tkinter', 'Tkinter', '_tkinter',
        'matplotlib', 'pandas',
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='realtimestt-addon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='realtimestt-addon',
)
