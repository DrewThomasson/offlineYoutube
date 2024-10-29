# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

# Collect data files for Gradio and Gradio Client
datas = []
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')

a = Analysis(
    ['app.py'],  # Your main application entry point
    pathex=[],  # Add paths if necessary
    binaries=[],  # Include any additional binaries if needed
    datas=datas,
    hiddenimports=[],  # Specify hidden imports if any
    hookspath=[],  # Add hook paths if required
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,  # Optimization level (0 for no optimization)
    module_collection_mode={
        'gradio': 'py',  # Collect Gradio as source .py files
    },
)

# Create the executable in a single-file format
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,  # Ensure single-file build
)

# Final collection step, collecting necessary files and binaries
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
