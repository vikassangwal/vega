#!/usr/bin/env python3
"""
================================================================================
VEGA DAILY BRIEFING - Morning Digest Generator
================================================================================
Generates personalized morning briefings with:
- Calendar overview (meetings, conflicts, prep needed)
- Weather forecast
- News headlines (personalized)
- Task overview (due, overdue, priorities)
- Habit status
- System health
- Commute/travel info
- Stock/market summary
- Custom sections

Delivered via TTS, HUD display, and/or notification.

References:
- OpenJarvis/OpenJarvis: Morning digest agent
- Google Assistant: Morning routine
- Amazon Alexa: Flash briefing
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from config import get_settings, settings

logger = logging.getLogger("vega.proactive.briefing")


# =============================================================================
# DATA MODELS
# =============================================================================

class BriefingSection(str, Enum):
    """Available briefing sections."""
    GREETING = "greeting"
    WEATHER = "weather"
    CALENDAR = "calendar"
    TASKS = "tasks"
    HABITS = "habits"
    NEWS = "news"
    STOCKS = "stocks"
    SYSTEM = "system"
    COMMUTE = "commute"
    EMAIL = "email"
    HEALTH = "health"
    CUSTOM = "custom"


@dataclass
class BriefingConfig:
    """Briefing configuration."""
    # Timing
    briefing_time: str = "07:00"  # HH:MM
    timezone: str = "UTC"
    enabled: bool = True

    # Sections (ordered)
    sections: list[BriefingSection] = field(default_factory=lambda: [
        BriefingSection.GREETING,
        BriefingSection.WEATHER,
        BriefingSection.CALENDAR,
        BriefingSection.TASKS,
        BriefingSection.HABITS,
        BriefingSection.NEWS,
        BriefingSection.SYSTEM,
    ])

    # Section configs
    weather_location: str = "auto"  # auto, or "City, Country"
    weather_units: str = "metric"  # metric, imperial
    calendar_days_ahead: int = 1
    tasks_show_completed: bool = False
    tasks_max_items: int = 10
    news_categories: list[str] = field(default_factory=lambda: ["technology", "business", "science"])
    news_max_items: int = 5
    stocks_symbols: list[str] = field(default_factory=list)
    email_accounts: list[str] = field(default_factory=list)
    email_max_unread: int = 5

    # Delivery
    tts_enabled: bool = True
    tts_voice: str = "en-US-AriaNeural"
    hud_display: bool = True
    notification: bool = True
    save_to_file: bool = True
    output_path: str = "~/.vega/briefings"

    # Custom
    custom_sections: dict[str, Callable] = field(default_factory=dict)


@dataclass
class BriefingSectionContent:
    """Rendered briefing section."""
    section: BriefingSection
    title: str
    content: str
    priority: int = 0  # For ordering
    metadata: dict = field(default_factory=dict)


@dataclass
class DailyBriefing:
    """Complete daily briefing."""
    date: str
    generated_at: float
    sections: list[BriefingSectionContent]
    full_text: str
    tts_text: str
    duration_estimate: int  # seconds to read
    metadata: dict = field(default_factory=dict)


# =============================================================================
# DAILY BRIEFING GENERATOR
# =============================================================================

class DailyBriefing:
    """
    Generates personalized daily briefings.
    Composes sections from multiple data sources.
    """

    def __init__(
        self,
        event_bus: Any = None,
        websocket_manager: Any = None,
        tts_engine: Any = None,
        planner_agent: Any = None,
        memory_manager: Any = None,
        monitoring_daemon: Any = None,
        config: BriefingConfig | None = None,
    ):
        self.event_bus = event_bus
        self.websocket_manager = websocket_manager
        self.tts_engine = tts_engine
        self.planner_agent = planner_agent
        self.memory_manager = memory_manager
        self.monitoring_daemon = monitoring_daemon
        self.config = config or BriefingConfig()

        # State
        self._running = False
        self._scheduled_task: asyncio.Task | None = None
        self._last_briefing: DailyBriefing | None = None

        # Callbacks
        self._on_briefing_ready: Callable[[DailyBriefing], Any] | None = None

        logger.info("DailyBriefing initialized")

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    async def start(self) -> bool:
        """Start scheduled briefing generation."""
        if self._running:
            return True

        if not self.config.enabled:
            logger.info("Daily briefing disabled in config")
            return True

        self._running = True
        self._scheduled_task = asyncio.create_task(self._schedule_loop())
        logger.info(f"DailyBriefing scheduled for {self.config.briefing_time}")
        return True

    async def stop(self) -> None:
        """Stop briefing scheduler."""
        self._running = False
        if self._scheduled_task:
            self._scheduled_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduled_task
        logger.info("DailyBriefing stopped")

    # -------------------------------------------------------------------------
    # SCHEDULING
    # -------------------------------------------------------------------------

    async def _schedule_loop(self) -> None:
        """Run briefing at scheduled time daily."""
        while self._running:
            try:
                now = datetime.now()
                target_time = self._get_next_briefing_time(now)

                # Sleep until briefing time
                sleep_seconds = (target_time - now).total_seconds()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

                # Generate and deliver briefing
                await self.generate_and_deliver()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Briefing schedule error: {e}")
                await asyncio.sleep(60)

    def _get_next_briefing_time(self, now: datetime) -> datetime:
        """Calculate next briefing time."""
        hour, minute = map(int, self.config.briefing_time.split(":"))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if target <= now:
            target += timedelta(days=1)

        return target

    # -------------------------------------------------------------------------
    # BRIEFING GENERATION
    # -------------------------------------------------------------------------

    async def generate_and_deliver(self) -> DailyBriefing:
        """Generate briefing and deliver via configured channels."""
        briefing = await self.generate_briefing()

        # Deliver via TTS
        if self.config.tts_enabled and self.tts_engine:
            await self._deliver_tts(briefing)

        # Display on HUD
        if self.config.hud_display and self.websocket_manager:
            await self._display_hud(briefing)

        # Send notification
        if self.config.notification and self.websocket_manager:
            await self._send_notification(briefing)

        # Save to file
        if self.config.save_to_file:
            await self._save_briefing(briefing)

        # Emit event
        if self.event_bus:
            await self.event_bus.emit("briefing.generated", {
                "date": briefing.date,
                "sections": [s.section.value for s in briefing.sections],
                "duration": briefing.duration_estimate,
            })

        # Callback
        if self._on_briefing_ready:
            await self._on_briefing_ready(briefing)

        self._last_briefing = briefing
        return briefing

    async def generate_briefing(self, date: str | None = None) -> DailyBriefing:
        """Generate briefing for specific date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Generating briefing for {date}")

        sections = []

        # Generate each section in order
        for section_type in self.config.sections:
            try:
                content = await self._generate_section(section_type)
                if content:
                    sections.append(content)
            except Exception as e:
                logger.error(f"Section {section_type.value} generation failed: {e}")
                sections.append(BriefingSectionContent(
                    section=section_type,
                    title=section_type.value.title(),
                    content=f"Error generating {section_type.value}: {e}",
                    priority=-1,
                ))

        # Custom sections
        for name, generator in self.config.custom_sections.items():
            try:
                content = await generator()
                if content:
                    sections.append(BriefingSectionContent(
                        section=BriefingSection.CUSTOM,
                        title=name,
                        content=content,
                    ))
            except Exception as e:
                logger.error(f"Custom section {name} failed: {e}")

        # Build full text
        full_text = self._build_full_text(sections)
        tts_text = self._build_tts_text(sections)
        duration = self._estimate_duration(tts_text)

        briefing = DailyBriefing(
            date=date,
            generated_at=time.time(),
            sections=sections,
            full_text=full_text,
            tts_text=tts_text,
            duration_estimate=duration,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "section_count": len(sections),
            },
        )

        return briefing

    # -------------------------------------------------------------------------
    # SECTION GENERATORS
    # -------------------------------------------------------------------------

    async def _generate_section(self, section: BriefingSection) -> BriefingSectionContent | None:
        """Generate a specific briefing section."""
        generators = {
            BriefingSection.GREETING: self._generate_greeting,
            BriefingSection.WEATHER: self._generate_weather,
            BriefingSection.CALENDAR: self._generate_calendar,
            BriefingSection.TASKS: self._generate_tasks,
            BriefingSection.HABITS: self._generate_habits,
            BriefingSection.NEWS: self._generate_news,
            BriefingSection.STOCKS: self._generate_stocks,
            BriefingSection.SYSTEM: self._generate_system,
            BriefingSection.COMMUTE: self._generate_commute,
            BriefingSection.EMAIL: self._generate_email,
            BriefingSection.HEALTH: self._generate_health,
        }

        generator = generators.get(section)
        if generator:
            return await generator()

        return None

    async def _generate_greeting(self) -> BriefingSectionContent:
        """Generate personalized greeting."""
        now = datetime.now()
        hour = now.hour

        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        day_name = now.strftime("%A, %B %d")
        date_str = now.strftime("%Y-%m-%d")

        content = f"{greeting}! Today is {day_name}, {date_str}."

        # Add motivational element
        import random
        motivators = [
            "Ready to make it a great day?",
            "Let's make things happen!",
            "You've got this!",
            "Every day is a new opportunity.",
        ]
        content += f" {random.choice(motivators)}"

        return BriefingSectionContent(
            section=BriefingSection.GREETING,
            title="Good Morning",
            content=content,
            priority=0,
        )

    async def _generate_weather(self) -> BriefingSectionContent:
        """Generate weather forecast."""
        # Would integrate with weather API (OpenWeatherMap, WeatherAPI, etc.)
        # For now, return placeholder
        content = "Weather: Currently 72°F and partly cloudy. High 78°F, low 65°F. 10% chance of rain."

        # In real implementation:
        # - Call weather API
        # - Get current conditions
        # - Get today's forecast
        # - Get precipitation probability
        # - Add UV index, humidity, wind

        return BriefingSectionContent(
            section=BriefingSection.WEATHER,
            title="Weather",
            content=content,
            priority=1,
            metadata={"location": "auto-detected"},
        )

    async def _generate_calendar(self) -> BriefingSectionContent:
        """Generate calendar overview."""
        events = []

        if self.planner_agent:
            try:
                plan = await self.planner_agent._get_daily_plan(Context(conversation_id="briefing"))
                events = plan.get("events", [])
            except Exception:
                pass

        if not events:
            content = "No meetings scheduled today. Enjoy your free time!"
        else:
            lines = [f"You have {len(events)} meeting{'s' if len(events) != 1 else ''} today:"]
            for event in events[:5]:
                start = event.get("start", "")
                title = event.get("title", "Untitled")
                location = event.get("location", "")
                loc_str = f" at {location}" if location else ""
                lines.append(f"  • {start} - {title}{loc_str}")

            if len(events) > 5:
                lines.append(f"  ... and {len(events) - 5} more")

            # Check for conflicts
            conflicts = self._check_conflicts(events)
            if conflicts:
                lines.append(f"\n⚠️ {len(conflicts)} scheduling conflict{'s' if len(conflicts) > 1 else ''} detected!")

            content = "\n".join(lines)

        return BriefingSectionContent(
            section=BriefingSection.CALENDAR,
            title="Calendar",
            content=content,
            priority=2,
            metadata={"event_count": len(events)},
        )

    def _check_conflicts(self, events: list[dict]) -> list[dict]:
        """Check for overlapping events."""
        conflicts = []
        sorted_events = sorted(events, key=lambda e: e.get("start", ""))

        for i in range(len(sorted_events) - 1):
            curr = sorted_events[i]
            next_e = sorted_events[i + 1]

            curr_end = curr.get("end", "")
            next_start = next_e.get("start", "")

            if curr_end and next_start and curr_end > next_start:
                conflicts.append({"event1": curr, "event2": next_e})

        return conflicts

    async def _generate_tasks(self) -> BriefingSectionContent:
        """Generate task overview."""
        tasks = []

        if self.planner_agent:
            try:
                plan = await self.planner_agent._get_daily_plan(Context(conversation_id="briefing"))
                tasks = plan.get("tasks", [])
            except Exception:
                pass

        if not tasks:
            content = "No tasks due today. Great job staying on top of things!"
        else:
            # Separate by status
            overdue = [t for t in tasks if t.get("overdue")]
            due_today = [t for t in tasks if not t.get("overdue") and t.get("due_today")]
            upcoming = [t for t in tasks if not t.get("overdue") and not t.get("due_today")]

            lines = []

            if overdue:
                lines.append(f"🔴 {len(overdue)} overdue:")
                for t in overdue[:3]:
                    lines.append(f"  • {t.get('title', 'Untitled')}")

            if due_today:
                lines.append(f"🟡 {len(due_today)} due today:")
                for t in due_today[:5]:
                    lines.append(f"  • {t.get('title', 'Untitled')}")

            if upcoming and self.config.tasks_show_completed:
                lines.append(f"🟢 {len(upcoming)} upcoming:")
                for t in upcoming[:3]:
                    lines.append(f"  • {t.get('title', 'Untitled')}")

            content = "\n".join(lines) if lines else "All caught up!"

        return BriefingSectionContent(
            section=BriefingSection.TASKS,
            title="Tasks",
            content=content,
            priority=3,
            metadata={"total": len(tasks)},
        )

    async def _generate_habits(self) -> BriefingSectionContent:
        """Generate habit status."""
        if self.planner_agent:
            try:
                plan = await self.planner_agent._get_daily_plan(Context(conversation_id="briefing"))
                habits = plan.get("habits", [])
            except Exception:
                habits = []
        else:
            habits = []

        if not habits:
            content = "No habits configured. Add some in the planner!"
        else:
            completed = sum(1 for h in habits if h.get("completed_today"))
            total = len(habits)

            lines = [f"Habits: {completed}/{total} completed today"]

            for h in habits:
                status = "✅" if h.get("completed_today") else "⬜"
                streak = h.get("streak", 0)
                lines.append(f"  {status} {h.get('name', 'Habit')} (streak: {streak})")

            content = "\n".join(lines)

        return BriefingSectionContent(
            section=BriefingSection.HABITS,
            title="Habits",
            content=content,
            priority=4,
        )

    async def _generate_news(self) -> BriefingSectionContent:
        """Generate news headlines."""
        # Would integrate with NewsAPI, GNews, etc.
        content = "Top headlines: Tech stocks rally on AI optimism. New breakthrough in quantum computing. Climate summit reaches landmark agreement."

        return BriefingSectionContent(
            section=BriefingSection.NEWS,
            title="News",
            content=content,
            priority=5,
        )

    async def _generate_stocks(self) -> BriefingSectionContent:
        """Generate stock market summary."""
        if not self.config.stocks_symbols:
            return None

        # Would integrate with Alpha Vantage, Yahoo Finance, etc.
        content = "Markets: S&P 500 +0.8%, NASDAQ +1.2%, DOW +0.5%. Tech leading gains."

        return BriefingSectionContent(
            section=BriefingSection.STOCKS,
            title="Markets",
            content=content,
            priority=6,
        )

    async def _generate_system(self) -> BriefingSectionContent:
        """Generate system health summary."""
        if not self.monitoring_daemon:
            return BriefingSectionContent(
                section=BriefingSection.SYSTEM,
                title="System",
                content="System monitoring not available",
                priority=7,
            )

        metrics = self.monitoring_daemon.get_current_metrics()
        if not metrics:
            return None

        health = metrics.get_health_status()
        health_emoji = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(health.value, "⚪")

        content = f"{health_emoji} System: {health.value.title()}\n"
        content += f"  CPU: {metrics.cpu_percent:.0f}% | RAM: {metrics.memory_percent:.0f}% | Disk: {metrics.disk_percent:.0f}%"

        if metrics.battery_percent is not None:
            bat_emoji = "🔌" if metrics.battery_plugged else "🔋"
            content += f" | Battery: {bat_emoji} {metrics.battery_percent:.0f}%"

        return BriefingSectionContent(
            section=BriefingSection.SYSTEM,
            title="System Health",
            content=content,
            priority=7,
            metadata={"health": health.value},
        )

    async def _generate_commute(self) -> BriefingSectionContent:
        """Generate commute/travel info."""
        # Would integrate with Google Maps, transit APIs
        return None

    async def _generate_email(self) -> BriefingSectionContent:
        """Generate email summary."""
        # Would integrate with Gmail/Outlook API
        return None

    async def _generate_health(self) -> BriefingSectionContent:
        """Generate health/wellness summary."""
        return None

    # -------------------------------------------------------------------------
    # TEXT BUILDING
    # -------------------------------------------------------------------------

    def _build_full_text(self, sections: list[BriefingSectionContent]) -> str:
        """Build full briefing text for display."""
        lines = [
            "=" * 50,
            f"📋 DAILY BRIEFING - {datetime.now().strftime('%A, %B %d, %Y')}",
            "=" * 50,
            "",
        ]

        for section in sections:
            if section.content:
                lines.append(f"## {section.title}")
                lines.append(section.content)
                lines.append("")

        lines.append("=" * 50)
        lines.append("Have a wonderful day! 🚀")

        return "\n".join(lines)

    def _build_tts_text(self, sections: list[BriefingSectionContent]) -> str:
        """Build TTS-optimized text (no markdown, shorter)."""
        parts = []

        # Greeting
        greeting_section = next((s for s in sections if s.section == BriefingSection.GREETING), None)
        if greeting_section:
            parts.append(greeting_section.content)

        # Other sections - more concise for TTS
        for section in sections:
            if section.section == BriefingSection.GREETING:
                continue
            if section.content:
                # Remove markdown, keep sentences
                clean = section.content.replace("#", "").replace("•", "").replace("*", "")
                parts.append(f"{section.title}. {clean}")

        return " ".join(parts)

    def _estimate_duration(self, text: str) -> int:
        """Estimate reading time in seconds."""
        words = len(text.split())
        # Average speaking rate: 150 words per minute
        return int((words / 150) * 60)

    # -------------------------------------------------------------------------
    # DELIVERY
    # -------------------------------------------------------------------------

    async def _deliver_tts(self, briefing: DailyBriefing) -> None:
        """Deliver briefing via TTS."""
        if not self.tts_engine:
            return

        try:
            await self.tts_engine.speak(briefing.tts_text)
        except Exception as e:
            logger.error(f"TTS delivery failed: {e}")

    async def _display_hud(self, briefing: DailyBriefing) -> None:
        """Display briefing on HUD."""
        if not self.websocket_manager:
            return

        try:
            await self.websocket_manager.broadcast({
                "type": "briefing",
                "payload": {
                    "date": briefing.date,
                    "sections": [
                        {
                            "title": s.title,
                            "content": s.content,
                            "section": s.section.value,
                        }
                        for s in briefing.sections
                    ],
                    "full_text": briefing.full_text,
                    "tts_text": briefing.tts_text,
                },
            }, room="hud")
        except Exception as e:
            logger.error(f"HUD display failed: {e}")

    async def _send_notification(self, briefing: DailyBriefing) -> None:
        """Send briefing ready notification."""
        if not self.websocket_manager:
            return

        try:
            await self.websocket_manager.broadcast({
                "type": "notification",
                "payload": {
                    "title": "Daily Briefing Ready",
                    "message": f"Your {briefing.date} briefing is ready. {len(briefing.sections)} sections.",
                    "level": "info",
                },
            })
        except Exception as e:
            logger.error(f"Notification failed: {e}")

    async def _save_briefing(self, briefing: DailyBriefing) -> None:
        """Save briefing to file."""
        try:
            import os
            output_dir = os.path.expanduser(self.config.output_path)
            os.makedirs(output_dir, exist_ok=True)

            filepath = os.path.join(output_dir, f"briefing_{briefing.date}.md")
            with open(filepath, "w") as f:
                f.write(briefing.full_text)

            # Also save TTS version
            tts_path = os.path.join(output_dir, f"briefing_{briefing.date}_tts.txt")
            with open(tts_path, "w") as f:
                f.write(briefing.tts_text)

        except Exception as e:
            logger.error(f"Failed to save briefing: {e}")

    # -------------------------------------------------------------------------
    # MANUAL TRIGGER
    # -------------------------------------------------------------------------

    async def generate_now(self) -> DailyBriefing:
        """Generate briefing immediately (manual trigger)."""
        return await self.generate_and_deliver()

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------

    def on_briefing_ready(self, callback: Callable[[DailyBriefing], Any]) -> None:
        self._on_briefing_ready = callback

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    def get_last_briefing(self) -> DailyBriefing | None:
        return self._last_briefing

    async def health_check(self) -> dict:
        return {
            "healthy": self._running,
            "scheduled": self.config.enabled,
            "next_briefing": self._get_next_briefing_time(datetime.now()).isoformat() if self._running else None,
            "last_briefing": self._last_briefing.date if self._last_briefing else None,
        }


# =============================================================================
# FACTORY
# =============================================================================

async def create_daily_briefing(
    event_bus: Any = None,
    websocket_manager: Any = None,
    tts_engine: Any = None,
    planner_agent: Any = None,
    memory_manager: Any = None,
    monitoring_daemon: Any = None,
    config: BriefingConfig | None = None,
) -> DailyBriefing:
    briefing = DailyBriefing(
        event_bus=event_bus,
        websocket_manager=websocket_manager,
        tts_engine=tts_engine,
        planner_agent=planner_agent,
        memory_manager=memory_manager,
        monitoring_daemon=monitoring_daemon,
        config=config,
    )
    await briefing.start()
    return briefing


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DailyBriefing",
    "BriefingSection",
    "BriefingConfig",
    "BriefingSectionContent",
    "create_daily_briefing",
]