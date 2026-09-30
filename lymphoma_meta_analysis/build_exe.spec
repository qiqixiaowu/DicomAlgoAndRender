# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec 文件 — 淋巴瘤 Meta 分析工具
打包为单文件 exe，可在无 Python 环境的 Windows 上运行
"""

import os
import sys
import matplotlib

# 项目目录
PROJECT_DIR = os.path.dirname(os.path.abspath(SPEC))

# 查找 SimHei 字体（Windows 系统字体目录）
simhei_path = None
font_search_paths = [
    os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'simhei.ttf'),
    os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'msyh.ttc'),
]
for fp in font_search_paths:
    if os.path.exists(fp):
        simhei_path = fp
        break

# matplotlib 数据目录
mpl_data_dir = os.path.dirname(matplotlib.__file__)

block_cipher = None

# 数据文件列表
datas = [
    # matplotlib 配置/字体数据
    (mpl_data_dir, 'matplotlib'),
]

# 添加中文字体
if simhei_path:
    datas.append((simhei_path, 'fonts'))

# 添加已有的爬取数据（让 --skip-crawl 可用）
results_dir = os.path.join(PROJECT_DIR, 'results')
if os.path.isdir(results_dir):
    for f in os.listdir(results_dir):
        if f.startswith('pubmed_lymphoma_') and f.endswith('.json'):
            datas.append((os.path.join(results_dir, f), 'results'))
            break

# 隐藏导入
hiddenimports = [
    'scipy',
    'scipy.stats',
    'scipy.special',
    'numpy',
    'pandas',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
    'matplotlib.patches',
    'matplotlib.ticker',
    'requests',
    'lxml',
    'lxml.etree',
    'tqdm',
    'xml.etree.ElementTree',
    'xml.dom',
    'xml.sax',
    'encodings.utf_8',
    'encodings.gbk',
    'encodings.gb2312',
    'encodings.gb18030',
]

a = Analysis(
    ['main.py'],
    pathex=[PROJECT_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'IPython',
        'notebook',
        'jupyter',
        'pytest',
        # 排除不需要的大型 ML 框架
        'torch',
        'torchvision',
        'tensorflow',
        'keras',
        'sklearn',
        'scikit-learn',
        'sympy',
        'cv2',
        'opencv',
        'skimage',
        'scikit-image',
        'h5py',
        'tables',
        'sqlalchemy',
        'pymysql',
        'psycopg2',
        'cryptography',
        'nacl',
        'boto3',
        'botocore',
        'azure',
        'google',
        'aws',
        'win32com',
        'pythoncom',
        'pywintypes',
        'win32evtlog',
        'win32pdh',
        'win32perf',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='lymphoma_meta_analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
