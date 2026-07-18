# Local override of the pyinstaller-hooks-contrib webrtcvad hook.
#
# The contrib hook does `copy_metadata('webrtcvad')`, which raises
# PackageNotFoundError when the module is provided by the `webrtcvad-wheels`
# distribution (the prebuilt-wheel fork RealtimeSTT depends on) rather than the
# source `webrtcvad` package. A hook found on hookspath is used instead of the
# contrib hook, so this resolves whichever distribution actually supplies the
# metadata — or none, without failing the build.
from PyInstaller.utils.hooks import copy_metadata

datas = []
for _dist in ('webrtcvad', 'webrtcvad-wheels'):
    try:
        datas += copy_metadata(_dist)
    except Exception:
        pass
