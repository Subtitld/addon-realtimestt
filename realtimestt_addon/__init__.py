"""Subtitld RealtimeSTT add-on.

The package exists so the entry-point can be invoked both directly
(`python -m realtimestt_addon`, useful for development) and as a frozen
PyInstaller binary (`realtimestt-addon`).
"""

__version__ = '0.1.0'
# Dotless id — matches the Subtitld add-on catalog convention (asset names are
# `<id>-<version>-<platform>.zip`, and the catalog's parser rejects dots).
ADDON_ID = 'realtimestt'
PROTOCOL_VERSION = 1
