from __future__ import annotations

from dataclasses import dataclass

from modules.diagnosis import DiagnosisConfig


@dataclass(frozen=True)
class AppSettings:
    mode: str
    manual_mapping_enabled: bool
    ai_report_enabled: bool
    diagnosis_config: DiagnosisConfig
