# stable.spec
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect data files from heavy packages
datas = [
    ('assets', 'assets'),                          # Thai fonts folder
    *collect_data_files('easyocr'),                # EasyOCR model configs
    *collect_data_files('symspellpy'),             # frequency_dictionary_en_82_765.txt
    # torch and transformers excluded — auto-installed at runtime when NLLB-200 is first used
]

hiddenimports = [
    # PyQt6 modules that PyInstaller misses
    'PyQt6.QtOpenGLWidgets',
    'PyQt6.QtOpenGL',
    'OpenGL.GL',
    'OpenGL.platform.win32',
    # Other deps
    'easyocr',
    'kenlm',
    'symspellpy',
    'rapidfuzz',
    'shapely',
    'mss',
    'cv2',
    'google.generativeai',
    'pkg_resources.py2_warn',
]

a = Analysis(
    ['stable.py'],
    pathex=['D:\\anime-translator'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'transformers', 'sentencepiece', 'sacremoses', 'sentence_transformers'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,        # Use one-folder mode (much more reliable than one-file for torch/cuda)
    name='AnimeTranslator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                    # UPX can corrupt torch DLLs — keep off
    console=False,                # No black console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                    # Add 'assets/icon.ico' here if you have one
    onefile=False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AnimeTranslator',
)