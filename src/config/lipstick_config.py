from __future__ import annotations

from typing import Final, Iterable, Set


# 기본 립스틱 업종 정의
# - 저가 사치 / 뷰티 / 카페 / 패션 등
# - 필요에 따라 사용자가 직접 수정/확장 가능
LIPSTICK_SECTOR_NAMES: Final[Set[str]] = {
    # 카페/디저트
    "커피-음료",
    "제과점",
    "아이스크림",
    "베이커리",
    # 간단 외식 (저가 사치 성격)
    "분식전문점",
    "패스트푸드점",
    # 뷰티/미용
    "미용실",
    "피부관리실",
    "네일숍",
    "화장품",
    # 패션/액세서리
    "일반의류",
    "패션잡화",
    "액세서리",
}


def normalize_name(name: str | None) -> str:
    if name is None:
        return ""
    return str(name).strip()


def is_lipstick_sector_name(sector_name: str | None) -> bool:
    """
    서비스_업종_코드_명 기준 립스틱 업종 여부 판별.

    사용자가 LIPSTICK_SECTOR_NAMES 집합을 수정해서 기준을 조정할 수 있다.
    """
    n = normalize_name(sector_name)
    return n in LIPSTICK_SECTOR_NAMES


def add_lipstick_flag_by_name(
    sector_names: Iterable[str | None],
) -> list[bool]:
    """
    pandas Series 등에 map해서 쓰기 편하도록 리스트 형태로 변환.
    """
    return [is_lipstick_sector_name(n) for n in sector_names]

