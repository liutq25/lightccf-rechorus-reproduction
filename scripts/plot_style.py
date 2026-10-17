"""Use a real CJK font and Chinese labels in every exported research figure."""
from pathlib import Path
import os


def use_chinese_font():
    from matplotlib import font_manager, rcParams
    root = Path(__file__).resolve().parents[1]
    candidates = [
        os.environ.get('LIGHTCCF_CJK_FONT', ''),
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        str(root / 'tmp/fandol-fonts/FandolHei-Regular.otf'),
    ]
    name = None
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            name = font_manager.FontProperties(fname=candidate).get_name()
            break
    if name is None:
        available = {font.name for font in font_manager.fontManager.ttflist}
        name = next((n for n in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
                                 'Source Han Sans SC', 'FandolHei') if n in available), None)
    if name is None:
        raise RuntimeError('需要中文字体；将 LIGHTCCF_CJK_FONT 设置为可用的中文字体文件路径。')
    rcParams['font.family'] = [name, 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    return name
