"""
matplotlib 한글 폰트 설정 (한 번만 호출하면 됨).

- Mac: AppleGothic
- Windows: Malgun Gothic
- Linux/Colab: NanumGothic (설치 필요 시 font_manager로 로드)
"""
from __future__ import annotations

import platform

import matplotlib as mpl


def setup_mpl_font() -> None:
    """한글 표시 + 마이너스 기호 깨짐 방지."""
    mpl.rcParams["axes.unicode_minus"] = False

    system = platform.system()
    if system == "Darwin":
        mpl.rcParams["font.family"] = "AppleGothic"
    elif system == "Windows":
        mpl.rcParams["font.family"] = "Malgun Gothic"
    else:
        # Linux, Colab 등: 나눔고딕 (없으면 기본 폰트로 fallback)
        mpl.rcParams["font.family"] = "NanumGothic"

    # 폰트가 실제로 없으면 첫 번째 사용 가능한 폰트로 fallback
    try:
        from matplotlib import font_manager
        available = [f.name for f in font_manager.fontManager.ttflist]
        if mpl.rcParams["font.family"] not in available:
            for fallback in ("Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "DejaVu Sans"):
                if fallback in available:
                    mpl.rcParams["font.family"] = fallback
                    break
    except Exception:
        pass
