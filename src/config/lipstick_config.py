"""
립스틱 효과 분석용 업종 3분류 (이론 정합).

핵심 메커니즘: Income ↓ ⇒ Luxury ↓, Small Indulgence(Lipstick) ↑
- Lipstick: 단가 낮음 + 감정적 보상 소비 + 고가 사치 대체재
- Luxury: 경기 민감 고가 소비
- Necessity: 필수/생계/교육·서비스

실험 1(보수적): Core Lipstick만.
실험 2(확장형): Core + 조건부 립스틱(감정 소비 확장).
"""

from __future__ import annotations

from typing import Final, Iterable, Set

# ---------------------------------------------------------------------------
# A. 립스틱 Core (이론적으로 타당, 실험 1)
# 단가 낮음, 자기보상 소비, 고급 소비 대체
# ---------------------------------------------------------------------------
LIPSTICK_CORE: Final[Set[str]] = {
    "네일숍",
    "미용실",
    "피부관리실",
    "화장품",
    "제과점",
    "커피-음료",
}

# ---------------------------------------------------------------------------
# B. 조건부 립스틱 (확장 실험용, 실험 2)
# 감정 소비 성격 있으나 필수/여가/생계가 섞임
# ---------------------------------------------------------------------------
LIPSTICK_EXTENDED_ADDONS: Final[Set[str]] = {
    "치킨전문점",
    "패스트푸드점",
    "분식전문점",
    "호프-간이주점",
    "노래방",
}

LIPSTICK_EXTENDED: Final[Set[str]] = LIPSTICK_CORE | LIPSTICK_EXTENDED_ADDONS

# ---------------------------------------------------------------------------
# C. Luxury (고가 사치, 경기 민감)
# ---------------------------------------------------------------------------
LUXURY_SECTOR_NAMES: Final[Set[str]] = {
    "일반의류",
    "패션잡화",
    "액세서리",
    "가방",
    "시계및귀금속",
    "신발",
    "가전제품",
    "전자상거래업",
    "자동차미용",
    "자동차수리",
    "인테리어",
    "여관",
    "골프연습장",
    "스포츠클럽",
}

# ---------------------------------------------------------------------------
# D. Necessity (필수/생계/교육·서비스) — 나머지는 get_sector_group에서 necessity
# 참고용 리스트 (명시적 매핑이 필요한 경우 확장 가능)
# ---------------------------------------------------------------------------
NECESSITY_SECTOR_NAMES: Final[Set[str]] = {
    "슈퍼마켓",
    "편의점",
    "반찬가게",
    "미곡판매",
    "육류판매",
    "수산물판매",
    "청과상",
    "의약품",
    "일반의원",
    "치과의원",
    "한의원",
    "세탁소",
    "철물점",
    "외국어학원",
    "예술학원",
    "일반교습학원",
    "부동산중개업",
    "의료기기",
    "전자제품수리",
    "가전제품수리",
}

# 하위 호환: NARROW = CORE, 기존 변수명 유지
LIPSTICK_NARROW: Final[Set[str]] = LIPSTICK_CORE
LIPSTICK_SECTOR_NAMES: Final[Set[str]] = LIPSTICK_CORE


def normalize_name(name: str | None) -> str:
    if name is None:
        return ""
    return str(name).strip()


def get_sector_group(
    sector_name: str | None,
    *,
    use_extended_lipstick: bool = False,
) -> str:
    """
    업종명 → "lipstick" | "luxury" | "necessity".

    - lipstick: Core(실험1) 또는 Core+Extended(실험2)
    - luxury: 고가 사치
    - necessity: 그 외 (필수·교육·서비스 등)
    우선순위: luxury → lipstick → necessity.
    """
    n = normalize_name(sector_name)
    if n in LUXURY_SECTOR_NAMES:
        return "luxury"
    lip_set = LIPSTICK_EXTENDED if use_extended_lipstick else LIPSTICK_CORE
    if n in lip_set:
        return "lipstick"
    return "necessity"


def add_sector_group_by_name(
    sector_names: Iterable[str | None],
    *,
    use_extended_lipstick: bool = False,
) -> list[str]:
    """pandas Series 등에 map해서 쓰기 편하도록 리스트로 반환."""
    return [
        get_sector_group(n, use_extended_lipstick=use_extended_lipstick)
        for n in sector_names
    ]


def is_lipstick_sector_name(
    sector_name: str | None,
    *,
    use_narrow: bool = True,
) -> bool:
    """
    립스틱 업종 여부.
    use_narrow=True(기본): Core만 → 실험 1.
    use_narrow=False: Extended까지 → 실험 2.
    """
    n = normalize_name(sector_name)
    return n in (LIPSTICK_CORE if use_narrow else LIPSTICK_EXTENDED)


def is_luxury_sector_name(sector_name: str | None) -> bool:
    """럭셔리(고가 사치) 업종 여부."""
    return normalize_name(sector_name) in LUXURY_SECTOR_NAMES


def add_lipstick_flag_by_name(
    sector_names: Iterable[str | None],
    use_narrow: bool = True,
) -> list[bool]:
    """기본은 Core만(실험 1)."""
    return [is_lipstick_sector_name(n, use_narrow=use_narrow) for n in sector_names]


def add_luxury_flag_by_name(sector_names: Iterable[str | None]) -> list[bool]:
    """업종명 iterable → is_luxury 불리언 리스트."""
    return [is_luxury_sector_name(n) for n in sector_names]
