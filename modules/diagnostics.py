from __future__ import annotations

from modules.diagnosis import (
    ACTION_COLUMNS,
    ACTIONS,
    DiagnosisConfig,
    build_bid_adjustments,
    build_exact_targeting_opportunities,
    build_growth_list,
    build_negative_keywords,
    build_pause_list,
    build_priority_list,
    run_diagnosis,
    summarize_recommendations,
)


def generate_recommendations(df, target_acos, mode="完整版"):
    return run_diagnosis(df, target_acos, mode)
