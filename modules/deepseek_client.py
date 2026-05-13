from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import requests

from modules.metrics import format_percent


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]


@dataclass(frozen=True)
class DeepSeekResult:
    ok: bool
    content: str
    error: str = ""


def generate_deepseek_report(
    api_key: str,
    model: str,
    overview: dict[str, float],
    actions: pd.DataFrame,
    aggregations: dict[str, pd.DataFrame],
    target_acos: float,
    timeout: int = 120,
) -> DeepSeekResult:
    api_key = api_key.strip()
    model = model.strip() or DEEPSEEK_MODELS[0]
    if not api_key:
        return DeepSeekResult(False, "", "请先输入 DeepSeek 密钥。")
    if model not in DEEPSEEK_MODELS:
        return DeepSeekResult(False, "", f"不支持的模型：{model}")

    user_prompt = _build_prompt(overview, actions, aggregations, target_acos)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是资深亚马逊广告顾问。请基于用户提供的结构化广告诊断数据，"
                    "输出中文、专业、可执行的广告优化报告。不要编造未提供的数据。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=(15, timeout),
        )
    except requests.exceptions.ConnectionError as exc:
        return DeepSeekResult(False, "", f"无法连接 DeepSeek API：{_s(str(exc))}")
    except requests.exceptions.Timeout:
        return DeepSeekResult(False, "", f"DeepSeek API 请求超时（{timeout}s），请稍后重试。")
    except requests.exceptions.RequestException as exc:
        return DeepSeekResult(False, "", f"DeepSeek API 请求异常：{_s(str(exc))}")

    # Force UTF-8 before reading body
    resp.encoding = "utf-8"

    if resp.status_code == 401:
        return DeepSeekResult(False, "", "密钥无效（401 Unauthorized），请检查 DeepSeek 密钥是否正确。")
    if resp.status_code == 402:
        return DeepSeekResult(False, "", "账户余额不足（402 Payment Required），请充值。")
    if resp.status_code == 429:
        return DeepSeekResult(False, "", "请求频率超限（429），请稍后重试。")
    if not resp.ok:
        return DeepSeekResult(False, "", f"API 返回错误 HTTP {resp.status_code}：{resp.text[:300]}")

    try:
        body = resp.json()
    except ValueError:
        return DeepSeekResult(False, "", f"API 返回非 JSON 数据（前300字符）：{resp.text[:300]}")

    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not content:
        # Show raw response for debugging
        finish_reason = body.get("choices", [{}])[0].get("finish_reason", "unknown")
        return DeepSeekResult(False, "", f"DeepSeek 返回空内容。finish_reason={finish_reason}，raw={json.dumps(body, ensure_ascii=False)[:400]}")
    return DeepSeekResult(True, content)


def deepseek_report_to_dataframe(content: str, model: str) -> pd.DataFrame:
    return pd.DataFrame([{"章节": f"DeepSeek 复核报告（{model}）", "报告内容": content}])


def _s(text: str) -> str:
    return text[:300]


# ── prompt builder ──

def _build_prompt(
    overview: dict[str, float],
    actions: pd.DataFrame,
    aggregations: dict[str, pd.DataFrame],
    target_acos: float,
) -> str:
    context = {
        "目标 ACOS": format_percent(target_acos),
        "账户总览": _overview_payload(overview),
        "高优先级动作 Top 15": _records(actions, 15),
        "广告活动 Top 10": _records(aggregations.get("广告活动", pd.DataFrame()), 10),
        "搜索词 Top 20": _records(aggregations.get("搜索词", pd.DataFrame()), 20),
        "ASIN Top 10": _records(aggregations.get("ASIN", pd.DataFrame()), 10),
    }
    return (
        "请基于以下 Amazon Ads 诊断数据，生成一份专业广告顾问报告。\n"
        "报告必须包含：1. 账户整体判断；2. 最大问题；3. 浪费花费分析；"
        "4. 转化效率分析；5. 流量质量分析；6. 关键词/Targeting 机会；"
        "7. 广告活动结构问题；8. 未来 7 天行动计划；9. 预期改善效果。\n"
        "请给出具体可执行的动作建议，避免泛泛而谈。\n\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def _overview_payload(overview: dict[str, float]) -> dict[str, object]:
    return {
        "总曝光": round(float(overview.get("总曝光", 0)), 2),
        "总点击": round(float(overview.get("总点击", 0)), 2),
        "总花费": round(float(overview.get("总花费", 0)), 2),
        "总销售额": round(float(overview.get("总销售额", 0)), 2),
        "总订单": round(float(overview.get("总订单", 0)), 2),
        "CTR": format_percent(float(overview.get("CTR", 0))),
        "CPC": round(float(overview.get("CPC", 0)), 2),
        "CVR": format_percent(float(overview.get("CVR", 0))),
        "ACOS": format_percent(float(overview.get("ACOS", 0))),
        "ROAS": round(float(overview.get("ROAS", 0)), 2),
    }


def _records(dataframe: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    if dataframe.empty:
        return []
    columns = [
        column
        for column in [
            "优先级", "优先级评分", "建议动作", "合并动作", "诊断规则",
            "诊断层级", "诊断对象", "Campaign Name", "Ad Group Name",
            "Customer Search Term", "Targeting", "ASIN", "ASIN Type",
            "Impressions", "Clicks", "Spend", "Sales", "Orders",
            "CTR", "CPC", "CVR", "ACOS", "ROAS", "原因",
        ]
        if column in dataframe.columns
    ]
    prepared = dataframe[columns].head(limit).copy()
    for col in ["CTR", "CVR", "ACOS"]:
        if col in prepared.columns:
            prepared[col] = prepared[col].apply(lambda v: format_percent(float(v or 0)))
    for col in ["Spend", "Sales", "CPC", "ROAS"]:
        if col in prepared.columns:
            prepared[col] = prepared[col].apply(lambda v: round(float(v or 0), 2))
    return prepared.fillna("").to_dict(orient="records")
