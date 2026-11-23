"""
JARVIS OMEGA - Suggestion Engine
Context-aware proactive recommendations:
- Time-based suggestions (morning routine, break reminders)
- Context-aware (meeting prep, focus time, energy management)
- Pattern-based (learned routines, habit stacking)
- Resource-aware (battery, bandwidth, compute)
- Proactive problem prevention
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from config import get_settings, settings

logger = logging.getLogger("vega.proactive.suggestions")


# =============================================================================
# DATA MODELS
# =============================================================================

class SuggestionType(str, Enum):
    ROUTINE = "routine"
    MEETING_PREP = "meeting_prep"
    FOCUS_TIME = "focus_time"
    BREAK = "break"
    ENERGY = "energy"
    RESOURCE = "resource"
    HABIT = "habit"
    TASK = "task"
    LEARNING = "learning"
    AUTOMATION = "automation"
    SECURITY = "security"
    HEALTH = "health"


class SuggestionPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Context:
    timestamp: float = field(default_factory=time.time)
    hour: int = 0
    day_of_week: int = 0
    is_work_hours: bool = True
    timezone: str = "UTC"
    next_meeting: dict | None = None
    time_to_next_meeting: int | None = None
    meetings_today: int = 0
    free_blocks: list[dict] = field(default_factory=list)
    battery_percent: float | None = None
    battery_plugged: bool | None = None
    cpu_percent: float = 0
    memory_percent: float = 0
    disk_percent: float = 0
    network_quality: str = "good"
    active_app: str | None = None
    idle_time: int = 0
    focus_session_active: bool = False
    focus_duration: int = 0
    pending_tasks: int = 0
    overdue_tasks: int = 0
    due_soon_tasks: int = 0
    habits_due: list[str] = field(default_factory=list)
    habits_completed_today: int = 0
    custom: dict = field(default_factory=dict)


@dataclass
class Suggestion:
    suggestion_id: str
    type: SuggestionType
    priority: SuggestionPriority
    title: str
    message: str
    action: str | None = None
    action_params: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    expires_at: float | None = None
    created_at: float = field(default_factory=time.time)
    dismissed: bool = False
    accepted: bool = False
    source: str = "suggestion_engine"

    def to_dict(self) -> dict:
        return {
            "suggestion_id": self.suggestion_id,
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "action": self.action,
            "action_params": self.action_params,
            "context": self.context,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "dismissed": self.dismissed,
            "accepted": self.accepted,
            "source": self.source,
        }


@dataclass
class SuggestionRule:
    name: str
    type: SuggestionType
    priority: SuggestionPriority
    condition: Callable[[Context], bool]
    generator: Callable[[Context], Suggestion]
    cooldown: int = 3600
    enabled: bool = True


# =============================================================================
# SUGGESTION ENGINE
# =============================================================================

class SuggestionEngine:
    def __init__(
        self,
        event_bus: Any = None,
        websocket_manager: Any = None,
        planner_agent: Any = None,
        memory_manager: Any = None,
    ):
        self.event_bus = event_bus
        self.websocket_manager = websocket_manager
        self.planner_agent = planner_agent
        self.memory_manager = memory_manager

        self._rules: list[SuggestionRule] = []
        self._active_suggestions: dict[str, Suggestion] = {}
        self._suggestion_history: list[Suggestion] = []
        self._last_fired: dict[str, float] = {}
        self._context: Context = Context()
        self._running = False
        self._evaluation_task: asyncio.Task | None = None
        self._on_suggestion: Callable[[Suggestion], Any] | None = None

        self._init_builtin_rules()
        logger.info("SuggestionEngine initialized")

    def _init_builtin_rules(self) -> None:
        # Morning routine
        self.add_rule(SuggestionRule(
            name="morning_routine",
            type=SuggestionType.ROUTINE,
            priority=SuggestionPriority.HIGH,
            condition=lambda ctx: ctx.hour == 7 and ctx.day_of_week < 5,
            generator=lambda ctx: Suggestion(
                suggestion_id="morning_routine",
                type=SuggestionType.ROUTINE,
                priority=SuggestionPriority.HIGH,
                title="Good morning! ☀️",
                message="Time for your morning routine. Would you like me to read your daily briefing?",
                action="daily_briefing",
                expires_at=time.time() + 3600,
            ),
            cooldown=86400,
        ))

        # Meeting preparation
        self.add_rule(SuggestionRule(
            name="meeting_prep_15min",
            type=SuggestionType.MEETING_PREP,
            priority=SuggestionPriority.HIGH,
            condition=lambda ctx: ctx.next_meeting and ctx.time_to_next_meeting is not None and 10 <= ctx.time_to_next_meeting <= 15,
            generator=lambda ctx: Suggestion(
                suggestion_id=f"meeting_prep_{ctx.next_meeting.get('id', 'unknown')}",
                type=SuggestionType.MEETING_PREP,
                priority=SuggestionPriority.HIGH,
                title="Meeting in 15 minutes 📅",
                message=f"'{ctx.next_meeting.get('title', 'Meeting')}' starts soon. Want me to pull up relevant docs or notes?",
                action="meeting_prep",
                action_params={"meeting_id": ctx.next_meeting.get('id')},
                context={"meeting": ctx.next_meeting},
                expires_at=time.time() + 1800,
            ),
            cooldown=1800,
        ))

        self.add_rule(SuggestionRule(
            name="meeting_prep_5min",
            type=SuggestionType.MEETING_PREP,
            priority=SuggestionPriority.URGENT,
            condition=lambda ctx: ctx.next_meeting and ctx.time_to_next_meeting is not None and ctx.time_to_next_meeting <= 5,
            generator=lambda ctx: Suggestion(
                suggestion_id=f"meeting_now_{ctx.next_meeting.get('id', 'unknown')}",
                type=SuggestionType.MEETING_PREP,
                priority=SuggestionPriority.URGENT,
                title="Meeting starting now! 🎯",
                message=f"'{ctx.next_meeting.get('title', 'Meeting')}' is starting. Joining link ready?",
                action="join_meeting",
                action_params={"meeting_id": ctx.next_meeting.get('id')},
                context={"meeting": ctx.next_meeting},
                expires_at=time.time() + 600,
            ),
            cooldown=300,
        ))

        # Focus time suggestion
        self.add_rule(SuggestionRule(
            name="focus_time_available",
            type=SuggestionType.FOCUS_TIME,
            priority=SuggestionPriority.MEDIUM,
            condition=lambda ctx: (
                len(ctx.free_blocks) > 0 and
                any(b['duration'] >= 60 for b in ctx.free_blocks) and
                not ctx.focus_session_active and
                ctx.hour >= 9 and ctx.hour <= 17
            ),
            generator=lambda ctx: Suggestion(
                suggestion_id="focus_time",
                type=SuggestionType.FOCUS_TIME,
                priority=SuggestionPriority.MEDIUM,
                title="Focus time available 🎯",
                message=f"You have a {max(b['duration'] for b in ctx.free_blocks)}min block. Start a focus session?",
                action="start_focus",
                action_params={"duration": max(b['duration'] for b in ctx.free_blocks)},
                expires_at=time.time() + 3600,
            ),
            cooldown=7200,
        ))

        # Break reminder
        self.add_rule(SuggestionRule(
            name="break_reminder",
            type=SuggestionType.BREAK,
            priority=SuggestionPriority.MEDIUM,
            condition=lambda ctx: ctx.focus_session_active and ctx.focus_duration >= 50,
            generator=lambda ctx: Suggestion(
                suggestion_id="break_reminder",
                type=SuggestionType.BREAK,
                priority=SuggestionPriority.MEDIUM,
                title="Time for a break ☕",
                message=f"You've been focusing for {ctx.focus_duration} minutes. Take a 10-minute break?",
                action="take_break",
                action_params={"duration": 10},
                expires_at=time.time() + 1800,
            ),
            cooldown=3600,
        ))

        # Battery warning
        self.add_rule(SuggestionRule(
            name="battery_low",
            type=SuggestionType.RESOURCE,
            priority=SuggestionPriority.HIGH,
            condition=lambda ctx: ctx.battery_percent is not None and ctx.battery_percent < 20 and not ctx.battery_plugged,
            generator=lambda ctx: Suggestion(
                suggestion_id="battery_low",
                type=SuggestionType.RESOURCE,
                priority=SuggestionPriority.HIGH,
                title="Battery low 🔋",
                message=f"Battery at {ctx.battery_percent}%. Plug in your charger.",
                action="show_power_settings",
                expires_at=time.time() + 3600,
            ),
            cooldown=1800,
        ))

        self.add_rule(SuggestionRule(
            name="battery_critical",
            type=SuggestionType.RESOURCE,
            priority=SuggestionPriority.URGENT,
            condition=lambda ctx: ctx.battery_percent is not None and ctx.battery_percent < 10 and not ctx.battery_plugged,
            generator=lambda ctx: Suggestion(
                suggestion_id="battery_critical",
                type=SuggestionType.RESOURCE,
                priority=SuggestionPriority.URGENT,
                title="Battery critical! ⚡",
                message=f"Only {ctx.battery_percent}% remaining. Save work and plug in immediately!",
                action="emergency_save",
                expires_at=time.time() + 600,
            ),
            cooldown=300,
        ))

        # Disk space
        self.add_rule(SuggestionRule(
            name="disk_space_low",
            type=SuggestionType.RESOURCE,
            priority=SuggestionPriority.MEDIUM,
            condition=lambda ctx: ctx.disk_percent > 85,
            generator=lambda ctx: Suggestion(
                suggestion_id="disk_space_low",
                type=SuggestionType.RESOURCE,
                priority=SuggestionPriority.MEDIUM,
                title="Disk space running low 💾",
                message=f"Disk usage at {ctx.disk_percent}%. Run cleanup?",
                action="cleanup_disk",
                expires_at=time.time() + 86400,
            ),
            cooldown=86400,
        ))

        # Habit reminders
        self.add_rule(SuggestionRule(
            name="habit_reminder",
            type=SuggestionType.HABIT,
            priority=SuggestionPriority.MEDIUM,
            condition=lambda ctx: len(ctx.habits_due) > 0,
            generator=lambda ctx: Suggestion(
                suggestion_id="habit_reminder",
                type=SuggestionType.HABIT,
                priority=SuggestionPriority.MEDIUM,
                title="Habit reminder 📋",
                message=f"Time for: {', '.join(ctx.habits_due[:3])}",
                action="complete_habit",
                action_params={"habits": ctx.habits_due},
                expires_at=time.time() + 7200,
            ),
            cooldown=14400,
        ))

        # Overdue tasks
        self.add_rule(SuggestionRule(
            name="overdue_tasks",
            type=SuggestionType.TASK,
            priority=SuggestionPriority.HIGH,
            condition=lambda ctx: ctx.overdue_tasks > 0,
            generator=lambda ctx: Suggestion(
                suggestion_id="overdue_tasks",
                type=SuggestionType.TASK,
                priority=SuggestionPriority.HIGH,
                title="Overdue tasks 📋",
                message=f"You have {ctx.overdue_tasks} overdue tasks. Review them?",
                action="view_overdue",
                expires_at=time.time() + 3600,
            ),
            cooldown=7200,
        ))

        # End of day wrap-up
        self.add_rule(SuggestionRule(
            name="end_of_day_wrapup",
            type=SuggestionType.ROUTINE,
            priority=SuggestionPriority.MEDIUM,
            condition=lambda ctx: ctx.hour == 17 and ctx.day_of_week < 5,
            generator=lambda ctx: Suggestion(
                suggestion_id="end_of_day_wrapup",
                type=SuggestionType.ROUTINE,
                priority=SuggestionPriority.MEDIUM,
                title="End of day wrap-up 📝",
                message="Review today's progress and plan tomorrow?",
                action="daily_wrapup",
                expires_at=time.time() + 3600,
            ),
            cooldown=86400,
        ))

        # Weekend planning
        self.add_rule(SuggestionRule(
            name="weekend_planning",
            type=SuggestionType.ROUTINE,
            priority=SuggestionPriority.LOW,
            condition=lambda ctx: ctx.day_of_week == 4 and ctx.hour == 16,
            generator=lambda ctx: Suggestion(
                suggestion_id="weekend_planning",
                type=SuggestionType.ROUTINE,
                priority=SuggestionPriority.LOW,
                title="Weekend planning 📅",
                message="Plan your weekend activities and goals?",
                action="plan_weekend",
                expires_at=time.time() + 86400,
            ),
            cooldown=604800,
        ))

    def add_rule(self, rule: SuggestionRule) -> None:
        self._rules.append(rule)
        logger.debug(f"Added suggestion rule: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def enable_rule(self, name: str, enabled: bool = True) -> bool:
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = enabled
                return True
        return False

    async def start(self, evaluation_interval: int = 60) -> bool:
        if self._running:
            return True
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop(evaluation_interval))
        logger.info("SuggestionEngine started")
        return True

    async def stop(self) -> None:
        self._running = False
        if self._evaluation_task:
            self._evaluation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._evaluation_task
        logger.info("SuggestionEngine stopped")

    async def _evaluation_loop(self, interval: int) -> None:
        while self._running:
            try:
                await self._update_context()
                await self._evaluate_rules()
                await self._cleanup_expired()
            except Exception as e:
                logger.error(f"Suggestion evaluation error: {e}")
            await asyncio.sleep(interval)

    async def _update_context(self) -> None:
        now = datetime.now()
        self._context.timestamp = time.time()
        self._context.hour = now.hour
        self._context.day_of_week = now.weekday()
        self._context.is_work_hours = 9 <= now.hour < 17 and now.weekday() < 5

        if self.planner_agent:
            try:
                plan = await self.planner_agent._get_daily_plan(Context(conversation_id="suggestion"))
                self._context.meetings_today = len(plan.get("events", []))
            except Exception:
                pass

        if self.memory_manager:
            try:
                pass
            except Exception:
                pass

    async def _evaluate_rules(self) -> None:
        for rule in self._rules:
            if not rule.enabled:
                continue
            last_fired = self._last_fired.get(rule.name, 0)
            if time.time() - last_fired < rule.cooldown:
                continue
            try:
                if rule.condition(self._context):
                    suggestion = rule.generator(self._context)
                    await self._fire_suggestion(suggestion, rule)
            except Exception as e:
                logger.error(f"Rule '{rule.name}' evaluation error: {e}")

    async def _fire_suggestion(self, suggestion: Suggestion, rule: SuggestionRule) -> None:
        existing = self._active_suggestions.get(suggestion.suggestion_id)
        if existing and not existing.dismissed and not existing.accepted:
            return

        self._active_suggestions[suggestion.suggestion_id] = suggestion
        self._suggestion_history.append(suggestion)
        self._last_fired[rule.name] = time.time()

        if self.event_bus:
            await self.event_bus.emit("suggestion.created", suggestion.to_dict())

        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "suggestion",
                "payload": suggestion.to_dict(),
            }, room="hud")

        if self._on_suggestion:
            await self._on_suggestion(suggestion)

        logger.info(f"Fired suggestion: {suggestion.title} ({suggestion.type.value})")

    async def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._active_suggestions.items()
            if s.expires_at and s.expires_at < now
        ]
        for sid in expired:
            del self._active_suggestions[sid]

    def update_context(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)

    def get_context(self) -> Context:
        return self._context

    def get_active_suggestions(self) -> list[Suggestion]:
        return [s for s in self._active_suggestions.values() if not s.dismissed and not s.accepted]

    def get_suggestion(self, suggestion_id: str) -> Suggestion | None:
        return self._active_suggestions.get(suggestion_id)

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        if suggestion_id in self._active_suggestions:
            self._active_suggestions[suggestion_id].dismissed = True
            del self._active_suggestions[suggestion_id]
            return True
        return False

    def accept_suggestion(self, suggestion_id: str) -> Suggestion | None:
        suggestion = self._active_suggestions.get(suggestion_id)
        if suggestion:
            suggestion.accepted = True
            del self._active_suggestions[suggestion_id]
            if self.event_bus:
                await self.event_bus.emit("suggestion.accepted", {
                    "suggestion_id": suggestion_id,
                    "action": suggestion.action,
                    "action_params": suggestion.action_params,
                })
            return suggestion
        return None

    def on_suggestion(self, callback: Callable[[Suggestion], Any]) -> None:
        self._on_suggestion = callback

    def get_stats(self) -> dict:
        return {
            "active_suggestions": len(self.get_active_suggestions()),
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules if r.enabled),
            "history_size": len(self._suggestion_history),
        }

    async def health_check(self) -> dict:
        return {
            "healthy": self._running,
            "active_suggestions": len(self.get_active_suggestions()),
            "context_age": time.time() - self._context.timestamp,
        }


async def create_suggestion_engine(
    event_bus: Any = None,
    websocket_manager: Any = None,
    planner_agent: Any = None,
    memory_manager: Any = None,
) -> SuggestionEngine:
    engine = SuggestionEngine(
        event_bus=event_bus,
        websocket_manager=websocket_manager,
        planner_agent=planner_agent,
        memory_manager=memory_manager,
    )
    await engine.start()
    return engine


__all__ = [
    "SuggestionEngine",
    "Suggestion",
    "SuggestionType",
    "SuggestionPriority",
    "Context",
    "SuggestionRule",
    "create_suggestion_engine",
]