from __future__ import annotations

from datetime import datetime

import pandas as pd

from modules.field_mapping import CANONICAL_FIELDS
from modules.metrics import format_percent


REPORT_SECTIONS = [
    "账户整体表现总结",
    "当前最大问题",
    "浪费花费分析",
    "转化效率分析",
    "流量质量分析",
    "关键词机会分析",
    "广告活动结构问题",
    "优先级行动计划",
    "未来 7 天优化建议",
    "预期改善效果",
]


def generate_ai_report(
    overview: dict[str, float],
    actions: pd.DataFrame,
    aggregations: dict[str, pd.DataFrame],
    target_acos: float,
) -> list[dict[str, str]]:
    context = _build_context(overview, actions, aggregations, target_acos)
    sections = [
        ("账户整体表现总结", _overall_summary(context)),
        ("当前最大问题", _largest_problem(context)),
        ("浪费花费分析", _wasted_spend_analysis(context)),
        ("转化效率分析", _conversion_analysis(context)),
        ("流量质量分析", _traffic_quality_analysis(context)),
        ("关键词机会分析", _keyword_opportunity_analysis(context)),
        ("广告活动结构问题", _campaign_structure_analysis(context)),
        ("优先级行动计划", _priority_action_plan(context)),
        ("未来 7 天优化建议", _seven_day_plan(context)),
        ("预期改善效果", _expected_impact(context)),
    ]

    return [
        {
            "章节": section,
            "报告内容": content,
        }
        for section, content in sections
    ]


def report_to_dataframe(report_sections: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(report_sections, columns=["章节", "报告内容"])


def report_to_markdown(report_sections: list[dict[str, str]]) -> str:
    lines = []
    for item in report_sections:
        lines.append(f"### {item['章节']}")
        lines.append(item["报告内容"])
        lines.append("")
    return "\n".join(lines).strip()


def _build_context(
    overview: dict[str, float],
    actions: pd.DataFrame,
    aggregations: dict[str, pd.DataFrame],
    target_acos: float,
) -> dict[str, object]:
    high_priority = _filter(actions, "优先级", "高")
    wasted_actions = actions[actions["建议动作"].isin(["否定精准", "否定词组", "暂停"])] if not actions.empty else pd.DataFrame()
    listing_actions = _filter(actions, "建议动作", "检查 Listing")
    growth_actions = actions[actions["建议动作"].isin(["提高竞价", "增加预算", "提取精准投放"])] if not actions.empty else pd.DataFrame()
    exact_actions = _filter(actions, "建议动作", "提取精准投放")
    bid_down_actions = _filter(actions, "建议动作", "降低竞价")

    campaign_df = aggregations.get("广告活动", pd.DataFrame())
    ad_group_df = aggregations.get("广告组", pd.DataFrame())
    search_term_df = aggregations.get("搜索词", pd.DataFrame())
    targeting_df = aggregations.get("Targeting", pd.DataFrame())

    wasted_spend = float(wasted_actions["Spend"].sum()) if "Spend" in wasted_actions else 0.0
    total_spend = float(overview.get("总花费", 0) or 0)
    wasted_ratio = wasted_spend / total_spend if total_spend else 0.0

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target_acos": target_acos,
        "overview": overview,
        "actions": actions,
        "high_priority": high_priority,
        "wasted_actions": wasted_actions,
        "listing_actions": listing_actions,
        "growth_actions": growth_actions,
        "exact_actions": exact_actions,
        "bid_down_actions": bid_down_actions,
        "campaign_df": campaign_df,
        "ad_group_df": ad_group_df,
        "search_term_df": search_term_df,
        "targeting_df": targeting_df,
        "top_waste": _top_rows(wasted_actions, "Spend", 5),
        "top_listing": _top_rows(listing_actions, "Impressions", 5),
        "top_growth": _top_rows(growth_actions, "Orders", 5),
        "top_campaigns": _top_rows(campaign_df, "Spend", 5),
        "top_bad_campaigns": _top_bad_efficiency(campaign_df, target_acos),
        "wasted_spend": wasted_spend,
        "wasted_ratio": wasted_ratio,
    }


def _overall_summary(context: dict[str, object]) -> str:
    overview = context["overview"]
    target_acos = context["target_acos"]
    return (
        f"本次诊断生成于 {context['generated_at']}。账户合计曝光 {overview['总曝光']:,.0f}，"
        f"点击 {overview['总点击']:,.0f}，花费 ${overview['总花费']:,.2f}，销售额 ${overview['总销售额']:,.2f}，"
        f"订单 {overview['总订单']:,.0f}。整体 CTR 为 {format_percent(overview['CTR'])}，"
        f"CVR 为 {format_percent(overview['CVR'])}，ACOS 为 {format_percent(overview['ACOS'])}，"
        f"ROAS 为 {overview['ROAS']:,.2f}。以目标 ACOS {format_percent(target_acos)} 衡量，"
        f"当前账户{'处于可控区间' if overview['ACOS'] <= target_acos else '需要优先压降无效花费并修复转化效率'}。"
    )


def _largest_problem(context: dict[str, object]) -> str:
    high_priority = context["high_priority"]
    overview = context["overview"]
    if high_priority.empty:
        return (
            "当前没有高优先级异常项。短期重点应放在持续观察样本、放大低 ACOS 单元，"
            "并避免过早否定仍在学习期的搜索词。"
        )

    top = _top_rows(high_priority, "Spend", 3)
    top_items = _format_action_items(top)
    return (
        f"当前最大问题是高优先级动作过多，共 {len(high_priority)} 条，主要集中在无订单消耗、暂停项或明显低效流量。"
        f"这些项目合计花费 ${high_priority['Spend'].sum():,.2f}，占账户总花费 "
        f"{format_percent(high_priority['Spend'].sum() / overview['总花费']) if overview['总花费'] else '0.00%'}。"
        f"优先处理对象包括：{top_items}。"
    )


def _wasted_spend_analysis(context: dict[str, object]) -> str:
    wasted_actions = context["wasted_actions"]
    if wasted_actions.empty:
        return "暂未发现明确应否定或暂停的浪费项。建议继续积累点击和订单样本，下一轮重点观察无订单搜索词。"

    return (
        f"浪费花费主要来自建议“否定精准 / 否定词组 / 暂停”的对象，共 {len(wasted_actions)} 条，"
        f"涉及花费 ${context['wasted_spend']:,.2f}，约占总花费 {format_percent(context['wasted_ratio'])}。"
        f"优先从花费最高且无订单的搜索词、Targeting、广告组入手，先做否定和暂停，再复查对应 Campaign 是否存在预算被低效流量挤占。"
        f"重点对象：{_format_action_items(context['top_waste'])}。"
    )


def _conversion_analysis(context: dict[str, object]) -> str:
    overview = context["overview"]
    bid_down_actions = context["bid_down_actions"]
    listing_actions = context["listing_actions"]
    message = (
        f"账户整体 CVR 为 {format_percent(overview['CVR'])}，CPC 为 ${overview['CPC']:,.2f}。"
        "转化效率要同时看 ACOS 与点击后的成交能力：ACOS 高但仍有订单的项目适合先降竞价，"
        "高 CTR 低 CVR 的项目则不是简单降价能解决，需要检查 Listing 承接。"
    )
    if not bid_down_actions.empty:
        message += f" 当前有 {len(bid_down_actions)} 条降低竞价建议，合计花费 ${bid_down_actions['Spend'].sum():,.2f}。"
    if not listing_actions.empty:
        message += f" 另有 {len(listing_actions)} 条需要检查 Listing 的流量，建议优先检查价格、Coupon、Review、主图和变体承接。"
    return message


def _traffic_quality_analysis(context: dict[str, object]) -> str:
    listing_actions = context["listing_actions"]
    overview = context["overview"]
    if listing_actions.empty:
        return (
            f"当前账户 CTR 为 {format_percent(overview['CTR'])}，未发现大量低 CTR 高曝光或高 CTR 低 CVR 项。"
            "后续可以继续按搜索词相关性和广告位质量观察流量变化。"
        )

    return (
        f"流量质量问题主要体现在 {len(listing_actions)} 条 Listing 检查建议中。"
        f"其中高曝光低 CTR 通常说明主图、标题、价格或关键词相关性不足；高 CTR 低 CVR 则说明点击意图存在，"
        f"但详情页、价格力或评价承接不足。建议优先复查：{_format_action_items(context['top_listing'])}。"
    )


def _keyword_opportunity_analysis(context: dict[str, object]) -> str:
    exact_actions = context["exact_actions"]
    growth_actions = context["growth_actions"]
    if exact_actions.empty and growth_actions.empty:
        return "当前没有明显精准投放或放量机会。建议等待更多订单样本后，再从低 ACOS 且订单稳定的搜索词中提取精准。"

    exact_text = (
        f"其中 {len(exact_actions)} 条适合提取精准投放，建议新建 Exact 或单独广告组承接。"
        if not exact_actions.empty
        else "暂未发现明确精准提取项。"
    )
    return (
        f"关键词机会集中在低 ACOS、有订单且仍有放量空间的对象，共 {len(growth_actions)} 条增长建议。"
        f"{exact_text} 重点对象：{_format_action_items(context['top_growth'])}。"
    )


def _campaign_structure_analysis(context: dict[str, object]) -> str:
    top_bad = context["top_bad_campaigns"]
    campaign_df = context["campaign_df"]
    if campaign_df.empty:
        return "当前没有可分析的广告活动聚合数据。"

    if top_bad.empty:
        return (
            "广告活动层面暂未发现明显结构性失衡。建议继续保持 Campaign 按目标、匹配类型或产品线分层，"
            "避免把探索、收割和品牌防守流量混在同一预算池。"
        )

    return (
        "广告活动结构问题主要来自花费较高但 ACOS 偏高或订单不足的 Campaign。"
        f"建议优先复查预算分配、匹配类型拆分和搜索词否定。重点 Campaign：{_format_dimension_items(top_bad, 'Campaign Name')}。"
    )


def _priority_action_plan(context: dict[str, object]) -> str:
    actions = context["actions"]
    if actions.empty:
        return "优先级行动计划：保持当前投放，未来 3-7 天继续积累数据，重点观察点击增长后的 ACOS 和 CVR 变化。"

    high = len(_filter(actions, "优先级", "高"))
    medium = len(_filter(actions, "优先级", "中"))
    low = len(_filter(actions, "优先级", "低"))
    return (
        f"建议按“先止损、再提效、后放量”的顺序执行。第一优先级：处理 {high} 条高优先级动作，"
        "包括否定词组、否定精准和暂停项，避免预算继续流向无订单流量。"
        f"第二优先级：执行 {medium} 条中优先级动作，包括降低高 ACOS 竞价、检查 Listing、提取精准投放。"
        f"第三优先级：评估 {low} 条低优先级机会，重点用于预算增加和小幅提高竞价。"
    )


def _seven_day_plan(context: dict[str, object]) -> str:
    return (
        "未来 7 天建议：第 1 天先执行否定、暂停和明显降竞价动作；第 2-3 天观察 Spend、CTR、CVR、ACOS 是否回落，"
        "不要同时大幅改动所有 Campaign；第 4 天把低 ACOS 且有订单的搜索词提取精准，并为优质广告组小幅增加预算；"
        "第 5-6 天检查高曝光低 CTR 和高 CTR 低 CVR 的 Listing，优先处理主图、价格、Coupon、标题相关性和评价承接；"
        "第 7 天复盘动作前后数据，保留有效策略，撤回无效调价，并准备下一轮搜索词清理。"
    )


def _expected_impact(context: dict[str, object]) -> str:
    overview = context["overview"]
    wasted_spend = context["wasted_spend"]
    target_acos = context["target_acos"]
    potential_saved = min(wasted_spend * 0.5, overview["总花费"] * 0.2) if overview["总花费"] else 0
    acos_note = (
        "如果销售额保持稳定，ACOS 有机会向目标区间回落。"
        if overview["ACOS"] > target_acos
        else "如果继续放大低 ACOS 流量，销售额有机会在不明显抬高 ACOS 的情况下增长。"
    )
    return (
        f"执行后 7 天内的预期改善主要来自减少浪费点击和提升预算利用率。按当前诊断估算，"
        f"可优先回收或重新分配约 ${potential_saved:,.2f} 的低效花费。{acos_note}"
        "实际效果取决于广告学习期、竞价竞争、库存、价格和 Listing 改动质量，建议用 7 天为一个复盘周期。"
    )


def _top_bad_efficiency(campaign_df: pd.DataFrame, target_acos: float) -> pd.DataFrame:
    if campaign_df.empty or "ACOS" not in campaign_df.columns:
        return pd.DataFrame()
    bad = campaign_df[
        ((campaign_df["Orders"] <= 0) & (campaign_df["Spend"] > 0))
        | ((campaign_df["Orders"] > 0) & (campaign_df["ACOS"] > target_acos))
    ].copy()
    return _top_rows(bad, "Spend", 5)


def _format_action_items(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无明确重点对象"
    items = []
    for _, row in df.iterrows():
        name = _first_text(row, ["诊断对象", "Customer Search Term", "Targeting", "Campaign Name"])
        action = _first_text(row, ["建议动作"])
        spend = _number(row, "Spend")
        acos = _number(row, "ACOS")
        items.append(f"{name}（{action}，花费 ${spend:,.2f}，ACOS {format_percent(acos)}）")
    return "；".join(items)


def _format_dimension_items(df: pd.DataFrame, name_column: str) -> str:
    if df.empty:
        return "暂无明确重点对象"
    items = []
    for _, row in df.iterrows():
        name = _first_text(row, [name_column])
        spend = _number(row, "Spend")
        acos = _number(row, "ACOS")
        orders = _number(row, "Orders")
        items.append(f"{name}（花费 ${spend:,.2f}，订单 {orders:.0f}，ACOS {format_percent(acos)}）")
    return "；".join(items)


def _top_rows(df: pd.DataFrame, sort_column: str, limit: int) -> pd.DataFrame:
    if df.empty or sort_column not in df.columns:
        return pd.DataFrame()
    return df.sort_values(sort_column, ascending=False).head(limit).copy()


def _filter(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    return df[df[column] == value].copy()


def _first_text(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            text = str(row[column]).strip()
            if text:
                return text
    return "未命名对象"


def _number(row: pd.Series, column: str) -> float:
    if column not in row.index or pd.isna(row[column]):
        return 0.0
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return 0.0
