from __future__ import annotations

import pandas as pd

from modules.field_mapping import CANONICAL_FIELDS
from modules.metrics import add_metrics


DIMENSION_CONFIG = {
    "广告活动": [CANONICAL_FIELDS["campaign_name"]],
    "广告组": [CANONICAL_FIELDS["campaign_name"], CANONICAL_FIELDS["ad_group_name"]],
    "搜索词": [
        CANONICAL_FIELDS["campaign_name"],
        CANONICAL_FIELDS["ad_group_name"],
        CANONICAL_FIELDS["customer_search_term"],
    ],
    "Targeting": [
        CANONICAL_FIELDS["campaign_name"],
        CANONICAL_FIELDS["ad_group_name"],
        CANONICAL_FIELDS["targeting"],
        CANONICAL_FIELDS["match_type"],
    ],
    "ASIN": [CANONICAL_FIELDS["advertised_asin"], CANONICAL_FIELDS["purchased_asin"]],
}


SUM_COLUMNS = [
    CANONICAL_FIELDS["impressions"],
    CANONICAL_FIELDS["clicks"],
    CANONICAL_FIELDS["spend"],
    CANONICAL_FIELDS["sales"],
    CANONICAL_FIELDS["orders"],
]


def build_dimension_aggregations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    aggregations: dict[str, pd.DataFrame] = {}

    for dimension_name, group_columns in DIMENSION_CONFIG.items():
        if dimension_name == "ASIN":
            asin_df = aggregate_asin_dimension(df)
            if not asin_df.empty:
                aggregations[dimension_name] = asin_df
            continue

        aggregations[dimension_name] = aggregate_by_dimension(df, group_columns, dimension_name)

    return aggregations


def aggregate_by_dimension(
    df: pd.DataFrame,
    group_columns: list[str],
    dimension_name: str,
) -> pd.DataFrame:
    available_columns = [column for column in group_columns if column in df.columns]
    if not available_columns or df.empty:
        return _empty_dimension_frame(available_columns, dimension_name)

    prepared = df.copy()
    for column in available_columns:
        prepared[column] = prepared[column].fillna("").astype(str).str.strip()
        prepared[column] = prepared[column].replace("", "(空)")

    grouped = (
        prepared.groupby(available_columns, dropna=False)[SUM_COLUMNS]
        .sum()
        .reset_index()
    )
    grouped["维度"] = dimension_name
    grouped["层级"] = dimension_name
    grouped = add_metrics(grouped)
    return _sort_by_spend(grouped)


def aggregate_asin_dimension(df: pd.DataFrame) -> pd.DataFrame:
    asin_columns = [
        column
        for column in [
            CANONICAL_FIELDS["advertised_asin"],
            CANONICAL_FIELDS["purchased_asin"],
        ]
        if column in df.columns and df[column].astype(str).str.strip().ne("").any()
    ]
    if not asin_columns:
        return _empty_dimension_frame(["ASIN Type", "ASIN"], "ASIN")

    frames = []
    for asin_column in asin_columns:
        prepared = df[df[asin_column].astype(str).str.strip() != ""].copy()
        prepared["ASIN Type"] = asin_column
        prepared["ASIN"] = prepared[asin_column].astype(str).str.strip()
        frames.append(prepared)

    if not frames:
        return _empty_dimension_frame(["ASIN Type", "ASIN"], "ASIN")

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["ASIN Type", "ASIN"], dropna=False)[SUM_COLUMNS]
        .sum()
        .reset_index()
    )
    grouped["维度"] = "ASIN"
    grouped["层级"] = "ASIN"
    grouped = add_metrics(grouped)
    return _sort_by_spend(grouped)


def _sort_by_spend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        by=[CANONICAL_FIELDS["spend"], CANONICAL_FIELDS["clicks"]],
        ascending=[False, False],
    ).reset_index(drop=True)


def _empty_dimension_frame(columns: list[str], dimension_name: str) -> pd.DataFrame:
    base_columns = columns + SUM_COLUMNS + ["CTR", "CPC", "CVR", "ACOS", "ROAS", "维度", "层级"]
    frame = pd.DataFrame(columns=base_columns)
    frame["维度"] = pd.Series(dtype="object")
    frame["层级"] = pd.Series(dtype="object")
    return frame
