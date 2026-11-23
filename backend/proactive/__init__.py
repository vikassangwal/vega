"""
================================================================================
VEGA PROACTIVE SYSTEM - Background Intelligence Package
================================================================================
Always-on autonomous intelligence:
- Monitoring Daemon: Continuous system/environment observation
- Suggestion Engine: Context-aware proactive recommendations
- Daily Briefing: Morning digest with calendar, weather, news, tasks
- Anomaly Detection: Behavioral & system anomaly identification

References:
- OpenJarvis/OpenJarvis: Morning digest, scheduled agents
- Microsoft JARVIS: Proactive task execution
- Home Assistant: Automation engine
================================================================================
"""

from .monitoring_daemon import MonitoringDaemon, SystemMetrics, HealthStatus
from .suggestion_engine import SuggestionEngine, Suggestion, SuggestionType, Context
from .daily_briefing import DailyBriefing, BriefingSection, BriefingConfig
from .anomaly_detection import AnomalyDetector, Anomaly, AnomalyType, Baseline

__all__ = [
    "MonitoringDaemon",
    "SystemMetrics",
    "HealthStatus",
    "SuggestionEngine",
    "Suggestion",
    "SuggestionType",
    "Context",
    "DailyBriefing",
    "BriefingSection",
    "BriefingConfig",
    "AnomalyDetector",
    "Anomaly",
    "AnomalyType",
    "Baseline",
]