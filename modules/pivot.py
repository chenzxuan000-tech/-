from __future__ import annotations

import pandas as pd

from modules.diagnosis import ACTIONS
from modules.field_mapping import CANONICAL_FIELDS
from modules.metrics import add_metrics


SUM_COLUMNS = [
    CANONICAL_FIELDS["impressions"],
    CANONICAL_FIELDS["clicks"],
    CANONICAL_FIELDS["spend"],
    CANONICAL_FIELDS["sales"],
    CANONICAL_FIELDS["orders"],
]

ACTION_PIVOT_PRESETS = {
    "广告活动": [CANONICAL_FIELDS["campaign_name"]],
    "广告活动 × 广告组": [CANONICAL_FIELDS["campaign_name"], CANONICAL_FIELDS["ad_group_name"]],
    "搜索词": [
        CANONICAL_FIELDS["campaign_name"],
        CANONICAL_FIELDS["ad_group_name"],
        CANONICAL_FIELDS["customer_search_term"],
    ],
    "Targeting": [
        CANONICAL_FIELDS["campaign_name"],
        CANONICAL_FIELDS["ad_group_name"],
        CANONICAL_FIELDS["targeting"],
    ],
    "建议动作": ["建议动作"],
    "优先级": ["优先级"],
}


def build_action_pivot(actions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    output_columns = _output_columns(group_columns)
    if actions.empty:
        return pd.DataFrame(columns=output_columns)

    available_groups = [column for column in group_columns if column in actions.columns]
    if not available_groups:
        return pd.DataFrame(columns=output_columns)

    prepared = actions.copy()
    for column in available_groups:
        prepared[column] = prepared[column].fillna("").astype(str).str.strip()
        prepared[column] = prepared[column].replace("", "(空)")

    metrics = (
        prepared.groupby(available_groups, dropna=False)[SUM_COLUMNS]
        .sum()
        .reset_index()
    )
    metrics = add_metrics(metrics)

    totals = (
        prepared.groupby(available_groups, dropna=False)
        .agg(
            建议数=("建议动作", "size"),
            最高优先级评分=("优先级评分", "max"),
        )
        .reset_index()
    )

    priority_counts = _count_values(prepared, available_groups, "优先级", ["高", "中", "低"], "优先级数")
    action_counts = _count_actions(prepared, available_groups)

    pivot = metrics.merge(totals, on=available_groups, how="left")
    pivot = pivot.merge(priority_counts, on=available_groups, how="left")
    pivot = pivot.merge(action_counts, on=available_groups, how="left")
    pivot = pivot.fillna(0)

    count_columns = [column for column in pivot.columns if column.endswith("数")]
    for column in ["建议数", "最高优先级评分", *count_columns]:
        if column in pivot.columns:
            pivot[column] = pivot[column].astype(int)

    pivot = pivot.sort_values(
        by=["高优先级数", "最高优先级评分", CANONICAL_FIELDS["spend"]],
        ascending=[False, False, False],
    )
    return pivot[[column for column in output_columns if column in pivot.columns]].reset_index(drop=True)


def build_export_pivots(actions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        f"透视-{name}": build_action_pivot(actions, columns)
        for name, columns in ACTION_PIVOT_PRESETS.items()
        if name in {"广告活动", "广告活动 × 广告组", "搜索词", "Targeting", "建议动作", "优先级"}
    }


def _count_values(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    values: list[str],
    suffix: str,
) -> pd.DataFrame:
    base = dataframe[group_columns].drop_duplicates().reset_index(drop=True)
    if value_column not in dataframe.columns:
        for value in values:
            base[f"{value}{suffix}"] = 0
        return base

    if value_column in group_columns:
        counts = (
            dataframe.groupby(group_columns, dropna=False)
            .size()
            .reset_index(name="_count")
        )
        for value in values:
            counts[f"{value}{suffix}"] = counts.apply(
                lambda row: int(row["_count"]) if str(row[value_column]) == value else 0,
                axis=1,
            )
        return counts[group_columns + [f"{value}{suffix}" for value in values]]

    counts = (
        dataframe.groupby(group_columns + [value_column], dropna=False)
        .size()
        .unstack(value_column, fill_value=0)
        .reset_index()
    )
    for value in values:
        if value not in counts.columns:
            counts[value] = 0
    counts = counts[group_columns + values]
    return counts.rename(columns={value: f"{value}{suffix}" for value in values})


def _count_actions(dataframe: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in dataframe.iterrows():
        action_text = str(row.get("合并动作") or row.get("建议动作") or "")
        matched_actions = [action for action in ACTIONS if action in action_text]
        if not matched_actions:
            matched_actions = [str(row.get("建议动作", "")).strip()]
        for action in matched_actions:
            if not action:
                continue
            rows.append({**{column: row[column] for column in group_columns}, "动作": action})

    base = dataframe[group_columns].drop_duplicates().reset_index(drop=True)
    if not rows:
        for action in ACTIONS:
            base[f"{action}数"] = 0
        return base

    exploded = pd.DataFrame(rows)
    counts = (
        exploded.groupby(group_columns + ["动作"], dropna=False)
        .size()
        .unstack("动作", fill_value=0)
        .reset_index()
    )
    for action in ACTIONS:
        if action not in counts.columns:
            counts[action] = 0
    counts = counts[group_columns + ACTIONS]
    return counts.rename(columns={action: f"{action}数" for action in ACTIONS})


def _output_columns(group_columns: list[str]) -> list[str]:
    return [
        *group_columns,
        "建议数",
        "高优先级数",
        "中优先级数",
        "低优先级数",
        "暂停数",
        "否定精准数",
        "否定词组数",
        "降低竞价数",
        "提高竞价数",
        "增加预算数",
        "提取精准投放数",
        "检查 Listing数",
        "继续观察数",
        "最高优先级评分",
        "Impressions",
        "Clicks",
        "Spend",
        "Sales",
        "Orders",
        "CTR",
        "CPC",
        "CVR",
        "ACOS",
        "ROAS",
    ]
