from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from modules.aggregation import aggregate_by_dimension
from modules.field_mapping import CANONICAL_FIELDS
from modules.metrics import format_percent


ACTIONS = [
    "暂停",
    "否定精准",
    "否定词组",
    "降低竞价",
    "提高竞价",
    "增加预算",
    "提取精准投放",
    "检查 Listing",
    "继续观察",
]


PRIORITY_RANK = {"高": 0, "中": 1, "低": 2}
ACTION_RANK = {action: index for index, action in enumerate(ACTIONS)}
IRRELEVANT_TERM_HINTS = [
    "free",
    "used",
    "second hand",
    "pdf",
    "manual",
    "repair",
    "replacement parts",
    "parts only",
    "diy",
    "wholesale",
    "coupon",
    "免费",
    "二手",
    "维修",
    "说明书",
    "配件",
    "批发",
]


ACTION_COLUMNS = [
    "诊断规则",
    "建议动作",
    "合并动作",
    "优先级",
    "优先级评分",
    "诊断层级",
    "诊断对象",
    "原因",
    "Campaign Name",
    "Ad Group Name",
    "Customer Search Term",
    "Targeting",
    "Match Type",
    "Ad Product",
    "Advertised ASIN",
    "Purchased ASIN",
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
    "Source Report",
]


@dataclass(frozen=True)
class DiagnosisConfig:
    target_acos: float = 0.30
    min_waste_clicks: int = 8
    hard_waste_clicks: int = 15
    min_waste_spend: float = 5.0
    high_acos_multiplier: float = 1.25
    low_acos_multiplier: float = 0.70
    min_quality_orders: int = 2
    high_ctr: float = 0.008
    low_ctr: float = 0.002
    low_cvr: float = 0.03
    high_impressions: int = 1000
    low_impressions: int = 300
    min_sales_low_exposure: float = 20.0
    budget_pressure_ratio: float = 0.80
    pause_spend_multiplier: float = 1.50
    exact_opportunity_orders: int = 2


def run_diagnosis(
    df: pd.DataFrame,
    config: DiagnosisConfig | float | None = None,
    mode: str = "完整版",
) -> pd.DataFrame:
    if config is None:
        config = DiagnosisConfig()
    elif isinstance(config, (int, float)):
        config = DiagnosisConfig(target_acos=float(config))

    actions: list[dict[str, object]] = []

    search_term_df = aggregate_by_dimension(
        df,
        [
            CANONICAL_FIELDS["campaign_name"],
            CANONICAL_FIELDS["ad_group_name"],
            CANONICAL_FIELDS["customer_search_term"],
        ],
        "搜索词",
    )
    targeting_df = aggregate_by_dimension(
        df,
        [
            CANONICAL_FIELDS["campaign_name"],
            CANONICAL_FIELDS["ad_group_name"],
            CANONICAL_FIELDS["targeting"],
            CANONICAL_FIELDS["match_type"],
        ],
        "Targeting",
    )
    is_full_mode = mode == "完整版"

    for _, row in search_term_df.iterrows():
        if not _text(row, CANONICAL_FIELDS["customer_search_term"]):
            continue
        actions.extend(_diagnose_keyword_like_row(row, config, "搜索词"))

    for _, row in targeting_df.iterrows():
        if not _text(row, CANONICAL_FIELDS["targeting"]):
            continue
        actions.extend(_diagnose_keyword_like_row(row, config, "Targeting"))

    if is_full_mode:
        campaign_df = aggregate_by_dimension(df, [CANONICAL_FIELDS["campaign_name"]], "广告活动")
        ad_group_df = aggregate_by_dimension(
            df,
            [CANONICAL_FIELDS["campaign_name"], CANONICAL_FIELDS["ad_group_name"]],
            "广告组",
        )

        for _, row in campaign_df.iterrows():
            actions.extend(_diagnose_campaign_row(row, config, df))

        for _, row in ad_group_df.iterrows():
            actions.extend(_diagnose_ad_group_row(row, config))

    if not actions:
        return pd.DataFrame(columns=ACTION_COLUMNS)

    action_df = pd.DataFrame(actions)
    action_df = _deduplicate_actions(action_df)
    action_df["优先级评分"] = action_df.apply(lambda row: _priority_score(row, config), axis=1)
    action_df["动作排序"] = action_df["建议动作"].map(ACTION_RANK).fillna(99)
    action_df["优先级排序"] = action_df["优先级"].map(PRIORITY_RANK).fillna(3)
    action_df = action_df.sort_values(
        by=["优先级排序", "优先级评分", "动作排序", CANONICAL_FIELDS["spend"], CANONICAL_FIELDS["clicks"]],
        ascending=[True, False, True, False, False],
    )
    return action_df[ACTION_COLUMNS].reset_index(drop=True)


def summarize_recommendations(actions: pd.DataFrame) -> dict[str, object]:
    if actions.empty:
        return {
            "总建议数": 0,
            "高优先级": 0,
            "否定建议": 0,
            "暂停建议": 0,
            "调价建议": 0,
            "增长建议": 0,
            "Listing问题": 0,
            "观察项": 0,
            "摘要文本": "暂未发现触发完整诊断规则的问题项，当前数据可以继续观察。",
        }

    action_counter = _count_actions(actions)
    priority_counter = Counter(actions["优先级"])
    summary = {
        "总建议数": int(len(actions)),
        "高优先级": int(priority_counter.get("高", 0)),
        "否定建议": int(action_counter.get("否定精准", 0) + action_counter.get("否定词组", 0)),
        "暂停建议": int(action_counter.get("暂停", 0)),
        "调价建议": int(action_counter.get("降低竞价", 0) + action_counter.get("提高竞价", 0)),
        "增长建议": int(action_counter.get("增加预算", 0) + action_counter.get("提取精准投放", 0)),
        "Listing问题": int(action_counter.get("检查 Listing", 0)),
        "观察项": int(action_counter.get("继续观察", 0)),
    }
    summary["摘要文本"] = (
        f"本次共生成 {summary['总建议数']} 条动作建议，其中高优先级 {summary['高优先级']} 条；"
        f"否定 {summary['否定建议']} 条，暂停 {summary['暂停建议']} 条，"
        f"调价 {summary['调价建议']} 条，增长放量 {summary['增长建议']} 条，"
        f"Listing 检查 {summary['Listing问题']} 条。"
    )
    return summary


def build_negative_keywords(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Campaign Name",
        "Ad Group Name",
        "Negative Keyword",
        "Negative Match Type",
        "Source Action",
        "Reason",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)

    negative_actions = actions[_action_mask(actions, {"否定精准", "否定词组"})].copy()
    if negative_actions.empty:
        return pd.DataFrame(columns=columns)

    negative_actions["Negative Keyword"] = negative_actions["Customer Search Term"].where(
        negative_actions["Customer Search Term"].astype(str).str.strip().ne(""),
        negative_actions["Targeting"],
    )
    negative_actions["Negative Match Type"] = negative_actions.apply(_negative_match_type, axis=1)
    negative_actions["Source Action"] = negative_actions.apply(
        lambda row: _source_action(row, {"否定精准", "否定词组"}),
        axis=1,
    )
    negative_actions["Reason"] = negative_actions["原因"]
    return negative_actions[columns].drop_duplicates().reset_index(drop=True)


def build_bid_adjustments(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Campaign Name",
        "Ad Group Name",
        "Targeting",
        "Customer Search Term",
        "建议调价方向",
        "Source Action",
        "Reason",
        "ACOS",
        "Orders",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)

    bid_actions = actions[_action_mask(actions, {"降低竞价", "提高竞价"})].copy()
    if bid_actions.empty:
        return pd.DataFrame(columns=columns)

    bid_actions["建议调价方向"] = bid_actions.apply(_bid_direction, axis=1)
    bid_actions["Source Action"] = bid_actions.apply(
        lambda row: _source_action(row, {"降低竞价", "提高竞价"}),
        axis=1,
    )
    bid_actions["Reason"] = bid_actions["原因"]
    return bid_actions[columns].drop_duplicates().reset_index(drop=True)


def build_pause_list(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "诊断层级",
        "Campaign Name",
        "Ad Group Name",
        "诊断对象",
        "Reason",
        "Spend",
        "Sales",
        "Orders",
        "ACOS",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)
    pause_actions = actions[_action_mask(actions, {"暂停"})].copy()
    if pause_actions.empty:
        return pd.DataFrame(columns=columns)
    pause_actions["Reason"] = pause_actions["原因"]
    return pause_actions[columns].drop_duplicates().reset_index(drop=True)


def build_growth_list(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "建议动作",
        "诊断层级",
        "Campaign Name",
        "Ad Group Name",
        "Customer Search Term",
        "Targeting",
        "Reason",
        "Impressions",
        "Clicks",
        "Orders",
        "ACOS",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)
    growth_actions = actions[_action_mask(actions, {"提高竞价", "增加预算", "提取精准投放"})].copy()
    if growth_actions.empty:
        return pd.DataFrame(columns=columns)
    growth_actions["Reason"] = growth_actions["原因"]
    return growth_actions[columns].drop_duplicates().reset_index(drop=True)


def build_exact_targeting_opportunities(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Campaign Name",
        "Ad Group Name",
        "Customer Search Term",
        "建议投放方式",
        "Reason",
        "Impressions",
        "Clicks",
        "Spend",
        "Sales",
        "Orders",
        "ACOS",
        "ROAS",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)

    exact_actions = actions[_action_mask(actions, {"提取精准投放"})].copy()
    if exact_actions.empty:
        return pd.DataFrame(columns=columns)

    exact_actions["建议投放方式"] = "Exact"
    exact_actions["Reason"] = exact_actions["原因"]
    return exact_actions[columns].drop_duplicates().reset_index(drop=True)


def build_priority_list(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "优先级",
        "优先级评分",
        "建议动作",
        "诊断规则",
        "合并动作",
        "诊断层级",
        "诊断对象",
        "原因",
        "Spend",
        "Sales",
        "Orders",
        "ACOS",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)

    priority_rank = {"高": 0, "中": 1, "低": 2}
    priority = actions.copy()
    priority["优先级排序"] = priority["优先级"].map(priority_rank).fillna(3)
    priority = priority.sort_values(["优先级排序", "优先级评分", "Spend"], ascending=[True, False, False])
    return priority[columns].reset_index(drop=True)


def _diagnose_keyword_like_row(
    row: pd.Series,
    config: DiagnosisConfig,
    level: str,
) -> list[dict[str, object]]:
    actions = []
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    orders = _value(row, CANONICAL_FIELDS["orders"])
    spend = _value(row, CANONICAL_FIELDS["spend"])
    sales = _value(row, CANONICAL_FIELDS["sales"])
    impressions = _value(row, CANONICAL_FIELDS["impressions"])
    ctr = _value(row, "CTR")
    cvr = _value(row, "CVR")
    acos = _value(row, "ACOS")

    if orders == 0 and clicks < config.min_waste_clicks and spend < config.min_waste_spend:
        actions.append(
            _build_action(
                row,
                rule="无订单样本不足",
                action="继续观察",
                priority="低",
                level=level,
                reason=f"点击 {clicks:.0f} 次、花费 ${spend:.2f}，样本仍偏少，先继续观察。",
            )
        )
    elif orders == 0 and _looks_irrelevant(row, ctr, cvr, impressions, clicks):
        actions.append(
            _build_action(
                row,
                rule="明显不相关无订单词",
                action="否定精准",
                priority="高",
                level=level,
                reason=f"点击 {clicks:.0f} 次、花费 ${spend:.2f} 且无订单，搜索意图疑似不相关，建议否定精准。",
            )
        )
    elif orders == 0 and clicks >= config.hard_waste_clicks:
        actions.append(
            _build_action(
                row,
                rule="相关但高点击无转化",
                action="降低竞价",
                priority="中",
                level=level,
                reason=f"点击 {clicks:.0f} 次仍无订单，但未发现明确不相关信号，建议先降低竞价并继续观察。",
            )
        )
    elif orders == 0 and clicks >= config.min_waste_clicks and spend >= config.min_waste_spend:
        actions.append(
            _build_action(
                row,
                rule="相关但中等消耗无转化",
                action="继续观察",
                priority="低",
                level=level,
                reason=f"点击 {clicks:.0f} 次、花费 ${spend:.2f} 且无订单，暂不直接否定，建议观察更多样本或小幅降价。",
            )
        )

    if orders >= 1 and acos > config.target_acos * config.high_acos_multiplier:
        actions.append(
            _build_action(
                row,
                rule="高 ACOS 低效词",
                action="降低竞价",
                priority="中",
                level=level,
                reason=f"已有订单但 ACOS {format_percent(acos)} 高于目标 {format_percent(config.target_acos)}。",
            )
        )

    if orders >= config.min_quality_orders and 0 < acos <= config.target_acos * config.low_acos_multiplier:
        actions.append(
            _build_action(
                row,
                rule="低 ACOS 优质词",
                action="提高竞价",
                priority="中",
                level=level,
                reason=f"订单 {orders:.0f} 个且 ACOS {format_percent(acos)} 低于目标的 70%，具备放量空间。",
            )
        )

    if (
        level == "搜索词"
        and orders >= config.exact_opportunity_orders
        and 0 < acos <= config.target_acos * config.low_acos_multiplier
    ):
        actions.append(
            _build_action(
                row,
                rule="精准投放机会词",
                action="提取精准投放",
                priority="中",
                level=level,
                reason="搜索词已有稳定转化且 ACOS 健康，建议单独提取为精准投放。",
            )
        )

    if clicks >= 10 and ctr >= config.high_ctr and cvr < config.low_cvr:
        actions.append(
            _build_action(
                row,
                rule="高 CTR 低 CVR",
                action="检查 Listing",
                priority="中",
                level=level,
                reason=f"CTR {format_percent(ctr)} 不低，但 CVR {format_percent(cvr)} 偏低，需检查价格、评价、详情页承接。",
            )
        )

    if impressions >= config.high_impressions and ctr < config.low_ctr:
        actions.append(
            _build_action(
                row,
                rule="低 CTR 高曝光",
                action="检查 Listing",
                priority="中",
                level=level,
                reason=f"曝光 {impressions:.0f} 且 CTR {format_percent(ctr)} 低于 0.20%，建议检查主图、标题、价格和相关性。",
            )
        )

    if sales >= config.min_sales_low_exposure and orders >= 1 and impressions < config.low_impressions:
        actions.append(
            _build_action(
                row,
                rule="有销量但曝光少",
                action="提高竞价",
                priority="低",
                level=level,
                reason=f"已有销售 {sales:.2f} 但曝光仅 {impressions:.0f}，建议提高竞价测试更多流量。",
            )
        )

    if orders == 1 and clicks < config.min_waste_clicks and spend < config.min_waste_spend:
        actions.append(
            _build_action(
                row,
                rule="样本不足",
                action="继续观察",
                priority="低",
                level=level,
                reason="已有早期转化但样本不足，暂不建议激进调价。",
            )
        )

    return actions


def _diagnose_campaign_row(
    row: pd.Series,
    config: DiagnosisConfig,
    raw_df: pd.DataFrame,
) -> list[dict[str, object]]:
    actions = []
    spend = _value(row, CANONICAL_FIELDS["spend"])
    sales = _value(row, CANONICAL_FIELDS["sales"])
    orders = _value(row, CANONICAL_FIELDS["orders"])
    acos = _value(row, "ACOS")
    impressions = _value(row, CANONICAL_FIELDS["impressions"])
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    budget = _campaign_budget(row, raw_df)

    if budget > 0 and spend >= budget * config.budget_pressure_ratio and orders >= 1 and acos <= config.target_acos:
        actions.append(
            _build_action(
                row,
                rule="预算可能不足的广告活动",
                action="增加预算",
                priority="中",
                level="广告活动",
                reason=f"花费 {spend:.2f} 已接近预算 {budget:.2f}，且 ACOS {format_percent(acos)} 不高于目标。",
            )
        )

    if spend >= max(config.min_waste_spend * 3, config.target_acos * max(sales, 1) * config.pause_spend_multiplier) and orders == 0 and clicks >= config.hard_waste_clicks:
        actions.append(
            _build_action(
                row,
                rule="需要暂停的广告活动 / 广告组",
                action="暂停",
                priority="高",
                level="广告活动",
                reason=f"广告活动花费 {spend:.2f}、点击 {clicks:.0f} 且无订单，建议暂停复盘结构。",
            )
        )

    if orders >= 1 and impressions < config.low_impressions:
        actions.append(
            _build_action(
                row,
                rule="有销量但曝光少",
                action="增加预算",
                priority="低",
                level="广告活动",
                reason=f"广告活动已有 {orders:.0f} 个订单但曝光 {impressions:.0f} 偏少，可增加预算或扩大流量。",
            )
        )

    return actions


def _diagnose_ad_group_row(row: pd.Series, config: DiagnosisConfig) -> list[dict[str, object]]:
    actions = []
    spend = _value(row, CANONICAL_FIELDS["spend"])
    orders = _value(row, CANONICAL_FIELDS["orders"])
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    acos = _value(row, "ACOS")

    if orders == 0 and clicks >= config.hard_waste_clicks and spend >= config.min_waste_spend * 3:
        actions.append(
            _build_action(
                row,
                rule="需要暂停的广告活动 / 广告组",
                action="暂停",
                priority="高",
                level="广告组",
                reason=f"广告组点击 {clicks:.0f}、花费 {spend:.2f} 且无订单，建议暂停或重建投放结构。",
            )
        )

    if orders >= config.min_quality_orders and 0 < acos <= config.target_acos * config.low_acos_multiplier:
        actions.append(
            _build_action(
                row,
                rule="低 ACOS 优质词",
                action="增加预算",
                priority="低",
                level="广告组",
                reason=f"广告组订单 {orders:.0f} 且 ACOS {format_percent(acos)} 健康，可承接更多预算。",
            )
        )

    return actions


def _build_action(
    row: pd.Series,
    rule: str,
    action: str,
    priority: str,
    level: str,
    reason: str,
) -> dict[str, object]:
    campaign = _text(row, CANONICAL_FIELDS["campaign_name"])
    ad_group = _text(row, CANONICAL_FIELDS["ad_group_name"])
    search_term = _text(row, CANONICAL_FIELDS["customer_search_term"])
    targeting = _text(row, CANONICAL_FIELDS["targeting"])
    diagnosis_object = _diagnosis_object(level, campaign, ad_group, search_term, targeting, row)

    return {
        "诊断规则": rule,
        "建议动作": action,
        "合并动作": action,
        "优先级": priority,
        "诊断层级": level,
        "诊断对象": diagnosis_object,
        "原因": reason,
        "Campaign Name": campaign,
        "Ad Group Name": ad_group,
        "Customer Search Term": search_term,
        "Targeting": targeting,
        "Match Type": _text(row, CANONICAL_FIELDS["match_type"]),
        "Ad Product": _text(row, CANONICAL_FIELDS["ad_product"]),
        "Advertised ASIN": _text(row, CANONICAL_FIELDS["advertised_asin"]),
        "Purchased ASIN": _text(row, CANONICAL_FIELDS["purchased_asin"]),
        "Impressions": _value(row, CANONICAL_FIELDS["impressions"]),
        "Clicks": _value(row, CANONICAL_FIELDS["clicks"]),
        "Spend": _value(row, CANONICAL_FIELDS["spend"]),
        "Sales": _value(row, CANONICAL_FIELDS["sales"]),
        "Orders": _value(row, CANONICAL_FIELDS["orders"]),
        "CTR": _value(row, "CTR"),
        "CPC": _value(row, "CPC"),
        "CVR": _value(row, "CVR"),
        "ACOS": _value(row, "ACOS"),
        "ROAS": _value(row, "ROAS"),
        "Source Report": _text(row, CANONICAL_FIELDS["source_report"]),
    }


def _campaign_budget(row: pd.Series, raw_df: pd.DataFrame) -> float:
    campaign = _text(row, CANONICAL_FIELDS["campaign_name"])
    budget_column = CANONICAL_FIELDS["budget"]
    if not campaign or budget_column not in raw_df.columns:
        return 0.0

    campaign_rows = raw_df[raw_df[CANONICAL_FIELDS["campaign_name"]].astype(str) == campaign]
    if campaign_rows.empty:
        return 0.0
    return float(campaign_rows[budget_column].max())


def _diagnosis_object(
    level: str,
    campaign: str,
    ad_group: str,
    search_term: str,
    targeting: str,
    row: pd.Series,
) -> str:
    if level == "广告活动":
        return campaign
    if level == "广告组":
        return " / ".join(part for part in [campaign, ad_group] if part)
    if level == "搜索词":
        return search_term or targeting
    if level == "Targeting":
        return targeting or search_term
    asin = _text(row, "ASIN")
    return asin or search_term or targeting or campaign


def _deduplicate_actions(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions

    deduped_rows = []
    identity_columns = ["诊断层级", "诊断对象", "Campaign Name", "Ad Group Name"]
    for _, group in actions.groupby(identity_columns, dropna=False):
        ranked = group.copy()
        ranked["优先级排序"] = ranked["优先级"].map(PRIORITY_RANK).fillna(3)
        ranked["动作排序"] = ranked["建议动作"].map(ACTION_RANK).fillna(99)
        ranked = ranked.sort_values(
            ["优先级排序", "动作排序", CANONICAL_FIELDS["spend"], CANONICAL_FIELDS["clicks"]],
            ascending=[True, True, False, False],
        )
        selected = ranked.iloc[0].copy()
        selected["诊断规则"] = _join_unique(group["诊断规则"])
        selected["合并动作"] = _join_unique(group["建议动作"])
        selected["原因"] = _join_unique(group["原因"])
        deduped_rows.append(selected)

    return pd.DataFrame(deduped_rows).reset_index(drop=True)


def _priority_score(row: pd.Series, config: DiagnosisConfig) -> int:
    priority = _text(row, "优先级")
    action = _text(row, "建议动作")
    rule = _text(row, "诊断规则")
    spend = _value(row, CANONICAL_FIELDS["spend"])
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    orders = _value(row, CANONICAL_FIELDS["orders"])
    sales = _value(row, CANONICAL_FIELDS["sales"])
    acos = _value(row, "ACOS")
    impressions = _value(row, CANONICAL_FIELDS["impressions"])

    score = {"高": 78, "中": 55, "低": 30}.get(priority, 25)
    score += min(spend / max(config.min_waste_spend, 1) * 4, 16)
    score += min(clicks / max(config.min_waste_clicks, 1) * 3, 12)

    if action in {"暂停", "否定精准", "否定词组"}:
        score += 8
    if action == "降低竞价" and orders >= 1 and acos > config.target_acos:
        score += min((acos / max(config.target_acos, 0.01) - 1) * 8, 12)
    if action in {"提高竞价", "增加预算", "提取精准投放"}:
        score += min(orders * 4, 14)
        if sales > 0 and 0 < acos <= config.target_acos * config.low_acos_multiplier:
            score += 8
    if "低 CTR 高曝光" in rule:
        score += min(impressions / max(config.high_impressions, 1) * 4, 10)
    if action == "继续观察":
        score -= 8

    return int(max(0, min(round(score), 100)))


def _action_mask(actions: pd.DataFrame, expected_actions: set[str]) -> pd.Series:
    if actions.empty:
        return pd.Series(dtype=bool)
    primary = actions["建议动作"].isin(expected_actions)
    if "合并动作" not in actions.columns:
        return primary
    merged = actions["合并动作"].fillna("").astype(str).apply(
        lambda value: any(action in value for action in expected_actions)
    )
    return primary | merged


def _count_actions(actions: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    if actions.empty:
        return counter
    for _, row in actions.iterrows():
        merged = _text(row, "合并动作") or _text(row, "建议动作")
        matched = [action for action in ACTIONS if action in merged]
        if not matched:
            matched = [_text(row, "建议动作")]
        counter.update(action for action in matched if action)
    return counter


def _source_action(row: pd.Series, expected_actions: set[str]) -> str:
    primary = _text(row, "建议动作")
    if primary in expected_actions:
        return primary
    merged = _text(row, "合并动作")
    for action in ACTIONS:
        if action in expected_actions and action in merged:
            return action
    return primary


def _bid_direction(row: pd.Series) -> str:
    source_action = _source_action(row, {"降低竞价", "提高竞价"})
    return {"降低竞价": "降低", "提高竞价": "提高"}.get(source_action, "")


def _negative_match_type(row: pd.Series) -> str:
    source_action = _source_action(row, {"否定精准", "否定词组"})
    return {"否定精准": "Negative Exact", "否定词组": "Negative Phrase"}.get(source_action, "")


def _looks_irrelevant(
    row: pd.Series,
    ctr: float,
    cvr: float,
    impressions: float,
    clicks: float,
) -> bool:
    search_text = " ".join(
        [
            _text(row, CANONICAL_FIELDS["customer_search_term"]),
            _text(row, CANONICAL_FIELDS["targeting"]),
        ]
    ).lower()
    if any(term in search_text for term in IRRELEVANT_TERM_HINTS):
        return True
    return clicks >= 20 and impressions >= 1000 and ctr < 0.002 and cvr == 0


def _join_unique(values: pd.Series) -> str:
    seen = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return "；".join(seen)


def _value(row: pd.Series, column: str) -> float:
    if column not in row.index or pd.isna(row[column]):
        return 0.0
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return 0.0


def _text(row: pd.Series, column: str) -> str:
    if column not in row.index or pd.isna(row[column]):
        return ""
    text = str(row[column]).strip()
    return "" if text == "(空)" else text
