from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TradeAreaChangeConfig:
    """
    상권변화지표 CSV 공통 설정.
    """

    encoding: str = "cp949"
    quarter_col: str = "기준_년분기_코드"
    change_code_col: str = "상권_변화_지표"
    change_name_col: str = "상권_변화_지표_명"
    oper_months_col: str = "운영_영업_개월_평균"
    close_months_col: str = "폐업_영업_개월_평균"
    seoul_oper_months_col: str = "서울_운영_영업_개월_평균"
    seoul_close_months_col: str = "서울_폐업_영업_개월_평균"


def _parse_quarter_code(code: str | int) -> tuple[int, int, str]:
    s = str(code).strip()
    year = int(s[:4])
    q = int(s[4])
    return year, q, f"{year}Q{q}"


def load_trade_change_by_dong(
    path: str | Path,
    config: TradeAreaChangeConfig | None = None,
) -> pd.DataFrame:
    """
    서울시 상권분석서비스(상권변화지표-행정동).csv 로더.

    반환 컬럼 예시
    -------------
    - year, quarter, year_quarter
    - dong_code, dong_name
    - change_code, change_name
    - oper_months_avg, close_months_avg
    - seoul_oper_months_avg, seoul_close_months_avg
    """
    if config is None:
        config = TradeAreaChangeConfig()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p, encoding=config.encoding)

    quarters = df[config.quarter_col].map(_parse_quarter_code)
    df["year"] = quarters.map(lambda t: t[0])
    df["quarter"] = quarters.map(lambda t: t[1])
    df["year_quarter"] = quarters.map(lambda t: t[2])

    out = df[
        [
            "year",
            "quarter",
            "year_quarter",
            "행정동_코드",
            "행정동_코드_명",
            config.change_code_col,
            config.change_name_col,
            config.oper_months_col,
            config.close_months_col,
            config.seoul_oper_months_col,
            config.seoul_close_months_col,
        ]
    ].rename(
        columns={
            "행정동_코드": "dong_code",
            "행정동_코드_명": "dong_name",
            config.change_code_col: "change_code",
            config.change_name_col: "change_name",
            config.oper_months_col: "oper_months_avg",
            config.close_months_col: "close_months_avg",
            config.seoul_oper_months_col: "seoul_oper_months_avg",
            config.seoul_close_months_col: "seoul_close_months_avg",
        }
    )
    return out


def load_trade_change_by_area(
    path: str | Path,
    config: TradeAreaChangeConfig | None = None,
) -> pd.DataFrame:
    """
    서울시 상권분석서비스(상권변화지표-상권).csv 로더.

    반환 컬럼 예시
    -------------
    - year, quarter, year_quarter
    - region_id, region_name
    - change_code, change_name
    - oper_months_avg, close_months_avg
    - seoul_oper_months_avg, seoul_close_months_avg
    """
    if config is None:
        config = TradeAreaChangeConfig()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p, encoding=config.encoding)

    quarters = df[config.quarter_col].map(_parse_quarter_code)
    df["year"] = quarters.map(lambda t: t[0])
    df["quarter"] = quarters.map(lambda t: t[1])
    df["year_quarter"] = quarters.map(lambda t: t[2])

    out = df[
        [
            "year",
            "quarter",
            "year_quarter",
            "상권_코드",
            "상권_코드_명",
            config.change_code_col,
            config.change_name_col,
            config.oper_months_col,
            config.close_months_col,
            config.seoul_oper_months_col,
            config.seoul_close_months_col,
        ]
    ].rename(
        columns={
            "상권_코드": "region_id",
            "상권_코드_명": "region_name",
            config.change_code_col: "change_code",
            config.change_name_col: "change_name",
            config.oper_months_col: "oper_months_avg",
            config.close_months_col: "close_months_avg",
            config.seoul_oper_months_col: "seoul_oper_months_avg",
            config.seoul_close_months_col: "seoul_close_months_avg",
        }
    )
    return out

