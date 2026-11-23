"""
JARVIS OMEGA - Planner Agent
Specialist agent for goals, calendar, habits, project management, and productivity.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from agents.base_agent import BaseAgent, AgentConfig, AgentContext
from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from config import get_settings, settings

logger = logging.getLogger("jarvis.agents.planner")


# =============================================================================
# DATA MODELS
# =============================================================================

class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class HabitFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Goal:
    """Goal with progress tracking."""
    goal_id: str
    title: str
    description: str
    status: GoalStatus = GoalStatus.ACTIVE
    # Hierarchy
    parent_goal_id: str | None = None
    sub_goals: list[str] = field(default_factory=list)
    # Metrics
    target_value: float = 1.0
    current_value: float = 0.0
    unit: str = "completion"
    # Time
    created_at: float = field(default_factory=time.time)
    target_date: float | None = None
    completed_at: float | None = None
    # Categorization
    category: str = "personal"
    tags: list[str] = field(default_factory=list)
    # Progress
    milestones: list[dict] = field(default_factory=list)  # {"title": "", "target": 0.5, "completed": false}


@dataclass
class Task:
    """Actionable task."""
    task_id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    # Relations
    goal_id: str | None = None
    project_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    # Scheduling
    priority: Priority = Priority.MEDIUM
    estimated_duration: int = 30  # minutes
    actual_duration: int = 0
    scheduled_at: float | None = None
    due_date: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    # Context
    tags: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


@dataclass
class Habit:
    """Habit with streak tracking."""
    habit_id: str
    name: str
    description: str = ""
    frequency: HabitFrequency = HabitFrequency.DAILY
    custom_schedule: dict = field(default_factory=dict)  # For CUSTOM frequency
    # Tracking
    streak: int = 0
    longest_streak: int = 0
    completions: list[float] = field(default_factory=list)  # Timestamps
    target_per_period: int = 1
    # Status
    active: bool = True
    created_at: float = field(default_factory=time.time)
    last_completed: float | None = None
    # Reminders
    reminder_time: str | None = None  # "07:00"
    reminder_enabled: bool = True


@dataclass
class Project:
    """Project with tasks and timeline."""
    project_id: str
    name: str
    description: str = ""
    status: str = "planning"  # planning, active, on_hold, completed
    # Timeline
    start_date: float | None = None
    end_date: float | None = None
    # Tasks
    tasks: list[str] = field(default_factory=list)  # Task IDs
    # Progress
    progress: float = 0.0
    # Team (for future)
    owner: str = "user"
    members: list[str] = field(default_factory=list)


@dataclass
class CalendarEvent:
    """Calendar event."""
    event_id: str
    title: str
    description: str = ""
    start_time: float
    end_time: float
    all_day: bool = False
    location: str = ""
    # Recurrence
    recurring: bool = False
    recurrence_rule: str = ""  # RRULE format
    # Relations
    task_id: str | None = None
    goal_id: str | None = None
    project_id: str | None = None
    # Reminders
    reminders: list[int] = field(default_factory=list)  # Minutes before


# =============================================================================
# PLANNER AGENT
# =============================================================================

class PlannerAgent(BaseAgent):
    """
    Planner Agent - Goals, calendar, habits, projects, productivity.
    The "architect" of VEGA.
    """

    def __init__(
        self,
        event_bus: Any = None,
        memory_manager: Any = None,
        tool_registry: Any = None,
        llm_client: Any = None,
    ):
        agent_config = AgentConfig(
            name="planner",
            role="planner",
            description="Goal setting, calendar management, habit tracking, project planning, productivity optimization",
            model="planner",
            temperature=0.3,
            max_tokens=8192,
            system_prompt=self._get_system_prompt(),
            tools=[
                "create_goal", "update_goal", "list_goals", "delete_goal",
                "create_task", "update_task", "list_tasks", "complete_task",
                "create_habit", "update_habit", "list_habits", "complete_habit",
                "create_project", "update_project", "list_projects",
                "schedule_event", "list_events", "get_schedule",
                "get_daily_plan", "get_weekly_plan", "optimize_schedule",
                "progress_report", "habit_report",
            ],
            can_delegate=False,
            can_handoff=True,
            stream_response=True,
            timeout=60.0,
        )

        super().__init__(
            config=agent_config,
            event_bus=event_bus,
            memory_manager=memory_manager,
            tool_registry=tool_registry,
            llm_client=llm_client,
        )

        # Data storage
        self._data_dir = Path.home() / ".vega" / "planner"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._goals: dict[str, Goal] = {}
        self._tasks: dict[str, Task] = {}
        self._habits: dict[str, Habit] = {}
        self._projects: dict[str, Project] = {}
        self._events: dict[str, CalendarEvent] = {}

        # Load data
        asyncio.create_task(self._load_all())

        # Register tools
        if self.tool_registry:
            self._register_tools()

    def _get_system_prompt(self) -> str:
        return """You are the Planner Agent - VEGA's productivity architect. You help users achieve their goals through structured planning.

Responsibilities:
1. GOAL MANAGEMENT: Create, track, decompose goals (OKRs, SMART goals)
2. TASK PLANNING: Break goals into actionable tasks with dependencies
3. HABIT FORMATION: Design, track, optimize habits with streaks
4. PROJECT MANAGEMENT: Plan projects with timelines, milestones, resources
5. CALENDAR OPTIMIZATION: Schedule events, time-block, resolve conflicts
6. PRODUCTIVITY ANALYSIS: Reports, insights, schedule optimization

Guidelines:
- Use SMART criteria: Specific, Measurable, Achievable, Relevant, Time-bound
- Decompose large goals into small, daily-actionable tasks
- Respect user's energy patterns (chronotype) for scheduling
- Build buffer time for unexpected tasks
- Review and adjust weekly
- Celebrate progress, learn from setbacks
- Habit stacking: attach new habits to existing routines

Output: Clear plans with priorities, time estimates, and next actions.""".

    # -------------------------------------------------------------------------
    # DATA PERSISTENCE
    # -------------------------------------------------------------------------

    async def _load_all(self) -> None:
        """Load all planner data from disk."""
        await asyncio.gather(
            self._load_goals(),
            self._load_tasks(),
            self._load_habits(),
            self._load_projects(),
            self._load_events(),
        )

    async def _load_goals(self) -> None:
        file = self._data_dir / "goals.json"
        if file.exists():
            with open(file) as f:
                data = json.load(f)
            for g in data:
                goal = Goal(**g)
                self._goals[goal.goal_id] = goal

    async def _load_tasks(self) -> None:
        file = self._data_dir / "tasks.json"
        if file.exists():
            with open(file) as f:
                data = json.load(f)
            for t in data:
                task = Task(**t)
                self._tasks[task.task_id] = task

    async def _load_habits(self) -> None:
        file = self._data_dir / "habits.json"
        if file.exists():
            with open(file) as f:
                data = json.load(f)
            for h in data:
                habit = Habit(**h)
                self._habits[habit.habit_id] = habit

    async def _load_projects(self) -> None:
        file = self._data_dir / "projects.json"
        if file.exists():
            with open(file) as f:
                data = json.load(f)
            for p in data:
                project = Project(**p)
                self._projects[project.project_id] = project

    async def _load_events(self) -> None:
        file = self._data_dir / "events.json"
        if file.exists():
            with open(file) as f:
                data = json.load(f)
            for e in data:
                event = CalendarEvent(**e)
                self._events[event.event_id] = event

    async def _save_all(self) -> None:
        await asyncio.gather(
            self._save_goals(),
            self._save_tasks(),
            self._save_habits(),
            self._save_projects(),
            self._save_events(),
        )

    async def _save_goals(self) -> None:
        file = self._data_dir / "goals.json"
        with open(file, "w") as f:
            json.dump([g.__dict__ for g in self._goals.values()], f, default=str)

    async def _save_tasks(self) -> None:
        file = self._data_dir / "tasks.json"
        with open(file, "w") as f:
            json.dump([t.__dict__ for t in self._tasks.values()], f, default=str)

    async def _save_habits(self) -> None:
        file = self._data_dir / "habits.json"
        with open(file, "w") as f:
            json.dump([h.__dict__ for h in self._habits.values()], f, default=str)

    async def _save_projects(self) -> None:
        file = self._data_dir / "projects.json"
        with open(file, "w") as f:
            json.dump([p.__dict__ for p in self._projects.values()], f, default=str)

    async def _save_events(self) -> None:
        file = self._data_dir / "events.json"
        with open(file, "w") as f:
            json.dump([e.__dict__ for e in self._events.values()], f, default=str)

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def _agent_initialize(self) -> None:
        logger.info("PlannerAgent initialized")

    async def _agent_shutdown(self) -> None:
        await self._save_all()

    # -------------------------------------------------------------------------
    # TOOL REGISTRATION
    # -------------------------------------------------------------------------

    def _register_tools(self) -> None:
        tools = [
            # Goals
            ("create_goal", self._tool_create_goal, "Create a new goal", [
                ToolParameter("title", "string", "Goal title", required=True),
                ToolParameter("description", "string", "Goal description"),
                ToolParameter("category", "string", "Category: personal, work, health, learning, financial", default="personal"),
                ToolParameter("target_date", "string", "Target date (YYYY-MM-DD)"),
                ToolParameter("target_value", "number", "Target value", default=1.0),
                ToolParameter("unit", "string", "Unit: completion, pages, km, dollars, hours", default="completion"),
                ToolParameter("milestones", "array", "Milestones", default=[]),
            ]),
            ("update_goal", self._tool_update_goal, "Update an existing goal", [
                ToolParameter("goal_id", "string", "Goal ID", required=True),
                ToolParameter("title", "string", "Title"),
                ToolParameter("description", "string", "Description"),
                ToolParameter("status", "string", "Status: active, paused, completed, archived, failed"),
                ToolParameter("current_value", "number", "Current progress value"),
                ToolParameter("target_value", "number", "Target value"),
            ]),
            ("list_goals", self._tool_list_goals, "List goals", [
                ToolParameter("status", "string", "Filter by status"),
            ]),
            ("delete_goal", self._tool_delete_goal, "Delete/archive a goal", [
                ToolParameter("goal_id", "string", "Goal ID", required=True),
            ]),
            # Tasks
            ("create_task", self._tool_create_task, "Create a new task", [
                ToolParameter("title", "string", "Task title", required=True),
                ToolParameter("description", "string", "Task description"),
                ToolParameter("goal_id", "string", "Related goal ID"),
                ToolParameter("project_id", "string", "Related project ID"),
                ToolParameter("priority", "string", "Priority: low, medium, high, urgent", default="medium"),
                ToolParameter("estimated_duration", "integer", "Estimated minutes", default=30),
                ToolParameter("due_date", "string", "Due date (YYYY-MM-DD)"),
                ToolParameter("depends_on", "array", "Task IDs this depends on"),
            ]),
            ("update_task", self._tool_update_task, "Update a task", [
                ToolParameter("task_id", "string", "Task ID", required=True),
                ToolParameter("title", "string", "Title"),
                ToolParameter("status", "string", "Status: pending, in_progress, blocked, completed, cancelled"),
                ToolParameter("priority", "string", "Priority"),
                ToolParameter("due_date", "string", "Due date (YYYY-MM-DD)"),
                ToolParameter("estimated_duration", "integer", "Estimated minutes"),
            ]),
            ("list_tasks", self._tool_list_tasks, "List tasks", [
                ToolParameter("status", "string", "Filter by status"),
            ]),
            ("complete_task", self._tool_complete_task, "Mark task as completed", [
                ToolParameter("task_id", "string", "Task ID", required=True),
            ]),
            # Habits
            ("create_habit", self._tool_create_habit, "Create a new habit", [
                ToolParameter("name", "string", "Habit name", required=True),
                ToolParameter("description", "string", "Description"),
                ToolParameter("frequency", "string", "Frequency: daily, weekly, monthly, custom", default="daily"),
                ToolParameter("target_per_period", "integer", "Target completions per period", default=1),
                ToolParameter("reminder_time", "string", "Reminder time (HH:MM)"),
            ]),
            ("update_habit", self._tool_update_habit, "Update a habit", [
                ToolParameter("habit_id", "string", "Habit ID", required=True),
                ToolParameter("name", "string", "Name"),
                ToolParameter("frequency", "string", "Frequency"),
                ToolParameter("reminder_time", "string", "Reminder time"),
                ToolParameter("active", "boolean", "Active status"),
            ]),
            ("list_habits", self._tool_list_habits, "List habits", []),
            ("complete_habit", self._tool_complete_habit, "Mark habit as completed", [
                ToolParameter("habit_id", "string", "Habit ID", required=True),
            ]),
            # Projects
            ("create_project", self._tool_create_project, "Create a new project", [
                ToolParameter("name", "string", "Project name", required=True),
                ToolParameter("description", "string", "Description"),
                ToolParameter("start_date", "string", "Start date (YYYY-MM-DD)"),
                ToolParameter("end_date", "string", "End date (YYYY-MM-DD)"),
            ]),
            ("update_project", self._tool_update_project, "Update a project", [
                ToolParameter("project_id", "string", "Project ID", required=True),
                ToolParameter("name", "string", "Name"),
                ToolParameter("status", "string", "Status: planning, active, on_hold, completed"),
            ]),
            ("list_projects", self._tool_list_projects, "List projects", []),
            # Calendar
            ("schedule_event", self._tool_schedule_event, "Schedule a calendar event", [
                ToolParameter("title", "string", "Event title", required=True),
                ToolParameter("start", "string", "Start time (ISO format)", required=True),
                ToolParameter("end", "string", "End time (ISO format)", required=True),
                ToolParameter("description", "string", "Description"),
                ToolParameter("location", "string", "Location"),
                ToolParameter("reminders", "array", "Reminder minutes before", default=[15]),
            ]),
            ("list_events", self._tool_list_events, "List calendar events", [
                ToolParameter("days", "integer", "Days ahead", default=7),
            ]),
            ("get_schedule", self._tool_get_schedule, "Get schedule for period", [
                ToolParameter("days", "integer", "Days ahead", default=30),
            ]),
            # Planning
            ("get_daily_plan", self._tool_get_daily_plan, "Get daily plan", []),
            ("get_weekly_plan", self._tool_get_weekly_plan, "Get weekly plan", []),
            ("optimize_schedule", self._tool_optimize_schedule, "Optimize schedule", []),
            # Reports
            ("progress_report", self._tool_progress_report, "Get progress report", []),
            ("habit_report", self._tool_habit_report, "Get habit report", []),
        ]

        for name, func, desc, params in tools:
            metadata = ToolMetadata(
                name=name,
                description=desc,
                category="planner",
                parameters=params,
                returns={"type": "object"},
                sandboxed=False,
                requires_approval=False,
                tags=["planning", "productivity"],
            )
            self.tool_registry.register(name, func, metadata)

    # -------------------------------------------------------------------------
    # TASK PROCESSING
    # -------------------------------------------------------------------------

    async def process_task(
        self,
        task: str,
        context: AgentContext,
    ) -> Any:
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["goal", "objective", "target", "okr"]):
            return await self._handle_goals(task, context)
        elif any(kw in task_lower for kw in ["task", "todo", "action item"]):
            return await self._handle_tasks(task, context)
        elif any(kw in task_lower for kw in ["habit", "routine", "streak", "daily"]):
            return await self._handle_habits(task, context)
        elif any(kw in task_lower for kw in ["project", "plan", "timeline", "milestone"]):
            return await self._handle_projects(task, context)
        elif any(kw in task_lower for kw in ["calendar", "schedule", "appointment", "meeting", "event"]):
            return await self._handle_calendar(task, context)
        elif any(kw in task_lower for kw in ["plan", "daily", "weekly", "schedule", "optimize"]):
            return await self._handle_planning(task, context)
        elif any(kw in task_lower for kw in ["report", "progress", "review", "stats"]):
            return await self._handle_reports(task, context)

        return await self._get_daily_plan(context)

    async def process_task_stream(
        self,
        task: str,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        yield f"📅 Planner: {task[:100]}...\n\n"

        result = await self.process_task(task, context)
        if isinstance(result, str):
            yield result
        else:
            yield json.dumps(result, indent=2)

    # -------------------------------------------------------------------------
    # GOAL HANDLING
    # -------------------------------------------------------------------------

    async def _handle_goals(self, task: str, context: AgentContext) -> dict:
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["create", "add", "new", "set"]):
            return await self._create_goal_from_text(task, context)
        elif any(kw in task_lower for kw in ["list", "show", "view"]):
            return await self._list_goals()
        elif any(kw in task_lower for kw in ["update", "modify", "change"]):
            return await self._update_goal_from_text(task)
        elif any(kw in task_lower for kw in ["delete", "remove", "archive"]):
            return await self._delete_goal_from_text(task)
        elif any(kw in task_lower for kw in ["progress", "status"]):
            return await self._goal_progress_report()

        return await self._list_goals()

    async def _create_goal_from_text(self, task: str, context: AgentContext) -> dict:
        if not self.llm_client:
            return {"error": "LLM not available for goal parsing"}

        prompt = f"""Extract goal details from: "{task}"

Return JSON:
{{
  "title": "Clear goal title",
  "description": "Detailed description",
  "category": "personal|work|health|learning|financial",
  "target_date": "YYYY-MM-DD or null",
  "target_value": 1.0,
  "unit": "completion|pages|km|dollars|hours",
  "milestones": [{{"title": "", "target": 0.25}}]
}}"""

        response = await self._call_llm([
            {"role": "system", "content": "Extract goal as JSON."},
            {"role": "user", "content": prompt},
        ])

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return await self._create_goal(data)
        except Exception:
            pass

        return {"error": "Failed to parse goal"}

    async def _create_goal(self, data: dict) -> dict:
        goal_id = str(uuid.uuid4())
        goal = Goal(
            goal_id=goal_id,
            title=data.get("title", "Untitled Goal"),
            description=data.get("description", ""),
            category=data.get("category", "personal"),
            target_value=data.get("target_value", 1.0),
            unit=data.get("unit", "completion"),
            tags=data.get("tags", []),
            milestones=data.get("milestones", []),
        )

        if data.get("target_date"):
            try:
                goal.target_date = datetime.fromisoformat(data["target_date"]).timestamp()
            except Exception:
                pass

        self._goals[goal_id] = goal
        await self._save_goals()

        return {"success": True, "goal_id": goal_id, "title": goal.title}

    async def _list_goals(self, status: GoalStatus | None = None) -> dict:
        goals = []
        for goal in self._goals.values():
            if status is None or goal.status == status:
                progress = goal.current_value / goal.target_value if goal.target_value > 0 else 0
                goals.append({
                    "goal_id": goal.goal_id,
                    "title": goal.title,
                    "description": goal.description,
                    "status": goal.status.value,
                    "progress": progress,
                    "category": goal.category,
                    "target_date": goal.target_date,
                    "tags": goal.tags,
                })

        return {"goals": goals, "total": len(goals)}

    async def _update_goal_from_text(self, task: str) -> dict:
        return {"error": "Use update_goal tool with goal_id"}

    async def _delete_goal_from_text(self, task: str) -> dict:
        return {"error": "Use delete_goal tool with goal_id"}

    async def _goal_progress_report(self) -> dict:
        active_goals = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]
        report = {"active_goals": len(active_goals), "details": []}

        for goal in active_goals:
            progress = goal.current_value / goal.target_value if goal.target_value > 0 else 0
            report["details"].append({
                "title": goal.title,
                "progress": progress,
                "status": goal.status.value,
            })

        return report

    # -------------------------------------------------------------------------
    # TASK HANDLING
    # -------------------------------------------------------------------------

    async def _handle_tasks(self, task: str, context: AgentContext) -> dict:
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["create", "add", "new"]):
            return await self._create_task_from_text(task)
        elif any(kw in task_lower for kw in ["complete", "done", "finish"]):
            return await self._complete_task_from_text(task)
        elif any(kw in task_lower for kw in ["list", "show"]):
            return await self._list_tasks()

        return await self._list_tasks()

    async def _create_task_from_text(self, task: str) -> dict:
        if not self.llm_client:
            return {"error": "LLM not available for task parsing"}

        prompt = f"""Extract task details from: "{task}"

Return JSON:
{{
  "title": "Task title",
  "description": "Details",
  "goal_id": "optional goal ID",
  "project_id": "optional project ID",
  "priority": "low|medium|high|urgent",
  "estimated_duration": 30,
  "due_date": "YYYY-MM-DD or null",
  "depends_on": ["task_id1", "task_id2"]
}}"""

        response = await self._call_llm([
            {"role": "system", "content": "Extract task as JSON."},
            {"role": "user", "content": prompt},
        ])

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return await self._create_task(data)
        except Exception:
            pass

        return {"error": "Failed to parse task"}

    async def _create_task(self, data: dict) -> dict:
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            title=data.get("title", "Untitled Task"),
            description=data.get("description", ""),
            goal_id=data.get("goal_id"),
            project_id=data.get("project_id"),
            priority=Priority(data.get("priority", "medium")),
            estimated_duration=data.get("estimated_duration", 30),
            due_date=data.get("due_date"),
            depends_on=data.get("depends_on", []),
            tags=data.get("tags", []),
        )

        if data.get("due_date"):
            try:
                task.due_date = datetime.fromisoformat(data["due_date"]).timestamp()
            except Exception:
                pass

        self._tasks[task_id] = task
        await self._save_tasks()

        # Link to goal
        if task.goal_id and task.goal_id in self._goals:
            self._goals[task.goal_id].sub_goals.append(task_id)
            await self._save_goals()

        return {"success": True, "task_id": task_id, "title": task.title}

    async def _complete_task_from_text(self, task: str) -> dict:
        return {"error": "Use complete_task tool with task_id"}

    async def _list_tasks(self, status: TaskStatus | None = None) -> dict:
        tasks = []
        for t in self._tasks.values():
            if status is None or t.status == status:
                tasks.append({
                    "task_id": t.task_id,
                    "title": t.title,
                    "status": t.status.value,
                    "priority": t.priority.value,
                    "due_date": t.due_date,
                    "goal_id": t.goal_id,
                    "project_id": t.project_id,
                })

        return {"tasks": tasks, "total": len(tasks)}

    # -------------------------------------------------------------------------
    # HABIT HANDLING
    # -------------------------------------------------------------------------

    async def _handle_habits(self, task: str, context: AgentContext) -> dict:
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["create", "add", "new", "start"]):
            return await self._create_habit_from_text(task)
        elif any(kw in task_lower for kw in ["complete", "done", "check", "log"]):
            return await self._complete_habit_from_text(task)
        elif any(kw in task_lower for kw in ["list", "show", "streak"]):
            return await self._list_habits()
        elif any(kw in task_lower for kw in ["report", "stats"]):
            return await self._habit_report()

        return await self._list_habits()

    async def _create_habit_from_text(self, task: str) -> dict:
        if not self.llm_client:
            return {"error": "LLM not available for habit parsing"}

        prompt = f"""Extract habit details from: "{task}"

Return JSON:
{{
  "name": "Habit name",
  "description": "Details",
  "frequency": "daily|weekly|monthly|custom",
  "target_per_period": 1,
  "reminder_time": "HH:MM or null",
  "custom_schedule": {{}}
}}"""

        response = await self._call_llm([
            {"role": "system", "content": "Extract habit as JSON."},
            {"role": "user", "content": prompt},
        ])

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return await self._create_habit(data)
        except Exception:
            pass

        return {"error": "Failed to parse habit"}

    async def _create_habit(self, data: dict) -> dict:
        habit_id = str(uuid.uuid4())
        habit = Habit(
            habit_id=habit_id,
            name=data.get("name", "Untitled Habit"),
            description=data.get("description", ""),
            frequency=HabitFrequency(data.get("frequency", "daily")),
            target_per_period=data.get("target_per_period", 1),
            reminder_time=data.get("reminder_time"),
            custom_schedule=data.get("custom_schedule", {}),
        )

        self._habits[habit_id] = habit
        await self._save_habits()

        return {"success": True, "habit_id": habit_id, "name": habit.name}

    async def _complete_habit_from_text(self, task: str) -> dict:
        return {"error": "Use complete_habit tool with habit_id"}

    async def _list_habits(self) -> dict:
        habits = []
        for habit in self._habits.values():
            if habit.active:
                today = datetime.now().date()
                streak = 0
                for comp in reversed(habit.completions):
                    comp_date = datetime.fromtimestamp(comp).date()
                    if comp_date == today or comp_date == today - timedelta(days=streak):
                        streak += 1
                    else:
                        break

                habits.append({
                    "habit_id": habit.habit_id,
                    "name": habit.name,
                    "frequency": habit.frequency.value,
                    "streak": streak,
                    "longest_streak": habit.longest_streak,
                    "reminder_time": habit.reminder_time,
                    "completed_today": any(
                        datetime.fromtimestamp(c).date() == today
                        for c in habit.completions
                    ),
                })

        return {"habits": habits, "total": len(habits)}

    async def _habit_report(self) -> dict:
        report = {"habits": [], "summary": {}}

        total_active = len([h for h in self._habits.values() if h.active])
        total_streaks = sum(h.streak for h in self._habits.values() if h.active)

        for habit in self._habits.values():
            if habit.active:
                report["habits"].append({
                    "name": habit.name,
                    "streak": habit.streak,
                    "longest": habit.longest_streak,
                    "completions_30d": len([
                        c for c in habit.completions
                        if c > time.time() - 30*86400
                    ]),
                })

        report["summary"] = {
            "active_habits": total_active,
            "total_current_streak": total_streaks,
            "avg_streak": total_streaks / total_active if total_active > 0 else 0,
        }

        return report

    # -------------------------------------------------------------------------
    # PROJECT HANDLING
    # -------------------------------------------------------------------------

    async def _handle_projects(self, task: str, context: AgentContext) -> dict:
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["create", "add", "new"]):
            return await self._create_project_from_text(task)
        elif any(kw in task_lower for kw in ["list", "show"]):
            return await self._list_projects()

        return await self._list_projects()

    async def _create_project_from_text(self, task: str) -> dict:
        if not self.llm_client:
            return {"error": "LLM not available for project parsing"}

        prompt = f"""Extract project details from: "{task}"

Return JSON:
{{
  "name": "Project name",
  "description": "Details",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
}}"""

        response = await self._call_llm([
            {"role": "system", "content": "Extract project as JSON."},
            {"role": "user", "content": prompt},
        ])

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return await self._create_project(data)
        except Exception:
            pass

        return {"error": "Failed to parse project"}

    async def _create_project(self, data: dict) -> dict:
        project_id = str(uuid.uuid4())
        project = Project(
            project_id=project_id,
            name=data.get("name", "Untitled Project"),
            description=data.get("description", ""),
        )

        if data.get("start_date"):
            try:
                project.start_date = datetime.fromisoformat(data["start_date"]).timestamp()
            except Exception:
                pass
        if data.get("end_date"):
            try:
                project.end_date = datetime.fromisoformat(data["end_date"]).timestamp()
            except Exception:
                pass

        self._projects[project_id] = project
        await self._save_projects()

        return {"success": True, "project_id": project_id, "name": project.name}

    async def _list_projects(self) -> dict:
        projects = []
        for p in self._projects.values():
            task_count = len([t for t in self._tasks.values() if t.project_id == p.project_id])
            completed_tasks = len([t for t in self._tasks.values() if t.project_id == p.project_id and t.status == TaskStatus.COMPLETED])

            projects.append({
                "project_id": p.project_id,
                "name": p.name,
                "status": p.status,
                "progress": p.progress,
                "task_count": task_count,
                "completed_tasks": completed_tasks,
                "start_date": p.start_date,
                "end_date": p.end_date,
            })

        return {"projects": projects, "total": len(projects)}

    # -------------------------------------------------------------------------
    # CALENDAR HANDLING
    # -------------------------------------------------------------------------

    async def _handle_calendar(self, task: str, context: AgentContext) -> dict:
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["schedule", "add", "create", "book"]):
            return await self._schedule_event_from_text(task)
        elif any(kw in task_lower for kw in ["list", "show", "today", "tomorrow", "week"]):
            return await self._list_events()

        return await self._list_events()

    async def _schedule_event_from_text(self, task: str) -> dict:
        if not self.llm_client:
            return {"error": "LLM not available for event parsing"}

        prompt = f"""Extract event details from: "{task}"

Return JSON:
{{
  "title": "Event title",
  "description": "Details",
  "start_time": "YYYY-MM-DDTHH:MM:SS",
  "end_time": "YYYY-MM-DDTHH:MM:SS",
  "all_day": false,
  "location": "",
  "reminders": [15, 60]
}}"""

        response = await self._call_llm([
            {"role": "system", "content": "Extract event as JSON."},
            {"role": "user", "content": prompt},
        ])

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return await self._schedule_event(data)
        except Exception:
            pass

        return {"error": "Failed to parse event"}

    async def _schedule_event(self, data: dict) -> dict:
        event_id = str(uuid.uuid4())
        event = CalendarEvent(
            event_id=event_id,
            title=data.get("title", "Untitled Event"),
            description=data.get("description", ""),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time", time.time() + 3600),
            all_day=data.get("all_day", False),
            location=data.get("location", ""),
            reminders=data.get("reminders", [15]),
            task_id=data.get("task_id"),
            goal_id=data.get("goal_id"),
            project_id=data.get("project_id"),
        )

        self._events[event_id] = event
        await self._save_events()

        return {"success": True, "event_id": event_id, "title": event.title}

    async def _list_events(self, days: int = 7) -> dict:
        now = time.time()
        end = now + days * 86400

        events = []
        for event in self._events.values():
            if now <= event.start_time <= end:
                events.append({
                    "event_id": event.event_id,
                    "title": event.title,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "all_day": event.all_day,
                    "location": event.location,
                })

        events.sort(key=lambda e: e["start_time"])

        return {"events": events, "total": len(events)}

    # -------------------------------------------------------------------------
    # PLANNING
    # -------------------------------------------------------------------------

    async def _handle_planning(self, task: str, context: AgentContext) -> dict:
        task_lower = task.lower()

        if "daily" in task_lower:
            return await self._get_daily_plan(context)
        elif "weekly" in task_lower:
            return await self._get_weekly_plan(context)
        elif "optimize" in task_lower:
            return await self._optimize_schedule()

        return await self._get_daily_plan(context)

    async def _get_daily_plan(self, context: AgentContext) -> dict:
        today = datetime.now().date()
        start_of_day = datetime.combine(today, datetime.min.time()).timestamp()
        end_of_day = datetime.combine(today, datetime.max.time()).timestamp()

        events = [
            e for e in self._events.values()
            if start_of_day <= e.start_time <= end_of_day
        ]
        events.sort(key=lambda e: e.start_time)

        due_tasks = [
            t for t in self._tasks.values()
            if t.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
            and t.due_date and t.due_date <= end_of_day
        ]
        due_tasks.sort(key=lambda t: t.due_date or 0)

        active_habits = [h for h in self._habits.values() if h.active]

        active_goals = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

        plan = {
            "date": today.isoformat(),
            "events": [
                {
                    "title": e.title,
                    "start": datetime.fromtimestamp(e.start_time).strftime("%H:%M"),
                    "end": datetime.fromtimestamp(e.end_time).strftime("%H:%M"),
                    "location": e.location,
                }
                for e in events
            ],
            "tasks": [
                {
                    "title": t.title,
                    "priority": t.priority.value,
                    "estimated_minutes": t.estimated_duration,
                    "goal_id": t.goal_id,
                }
                for t in due_tasks[:10]
            ],
            "habits": [
                {"name": h.name, "completed": any(
                    datetime.fromtimestamp(c).date() == today for c in h.completions
                )}
                for h in active_habits
            ],
            "goals": [
                {"title": g.title, "progress": g.current_value / g.target_value if g.target_value > 0 else 0}
                for g in active_goals[:5]
            ],
        }

        return plan

    async def _get_weekly_plan(self, context: AgentContext) -> dict:
        today = datetime.now().date()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        days = []
        for i in range(7):
            day = start_of_week + timedelta(days=i)
            day_start = datetime.combine(day, datetime.min.time()).timestamp()
            day_end = datetime.combine(day, datetime.max.time()).timestamp()

            day_events = [
                e for e in self._events.values()
                if day_start <= e.start_time <= day_end
            ]

            days.append({
                "date": day.isoformat(),
                "day_name": day.strftime("%A"),
                "events": len(day_events),
                "event_details": [
                    {"title": e.title, "time": datetime.fromtimestamp(e.start_time).strftime("%H:%M")}
                    for e in day_events[:3]
                ],
            })

        return {"week_start": start_of_week.isoformat(), "days": days}

    async def _optimize_schedule(self) -> dict:
        suggestions = []

        # Check for overloaded days
        for event in self._events.values():
            day_events = [
                e for e in self._events.values()
                if datetime.fromtimestamp(e.start_time).date() == datetime.fromtimestamp(event.start_time).date()
            ]
            if len(day_events) > 6:
                suggestions.append({
                    "type": "overloaded",
                    "message": f"{datetime.fromtimestamp(event.start_time).strftime('%A')} has {len(day_events)} events",
                    "action": "Consider rescheduling or declining",
                })

        # Check for tasks without time blocks
        unscheduled_tasks = [
            t for t in self._tasks.values()
            if t.status == TaskStatus.PENDING and t.scheduled_at is None and t.due_date
        ]
        if unscheduled_tasks:
            suggestions.append({
                "type": "unscheduled",
                "message": f"{len(unscheduled_tasks)} tasks due soon without time blocks",
                "action": "Schedule time blocks for important tasks",
            })

        # Habit stacking suggestions
        for habit in self._habits.values():
            if habit.active and habit.reminder_time and not habit.completions:
                suggestions.append({
                    "type": "habit",
                    "message": f"Habit '{habit.name}' not started yet",
                    "action": f"Set reminder for {habit.reminder_time}",
                })

        return {"suggestions": suggestions, "optimized": False}

    # -------------------------------------------------------------------------
    # REPORTS
    # -------------------------------------------------------------------------

    async def _handle_reports(self, task: str, context: AgentContext) -> dict:
        return {
            "daily_plan": await self._get_daily_plan(context),
            "goals": await self._goal_progress_report(),
            "habits": await self._habit_report(),
            "tasks": await self._list_tasks(),
        }

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_create_goal(self, params: dict) -> dict:
        return await self._create_goal(params)

    async def _tool_update_goal(self, params: dict) -> dict:
        goal_id = params.get("goal_id")
        if goal_id not in self._goals:
            return {"success": False, "error": "Goal not found"}

        goal = self._goals[goal_id]
        for key, value in params.items():
            if key != "goal_id" and hasattr(goal, key):
                setattr(goal, key, value)

        await self._save_goals()
        return {"success": True}

    async def _tool_list_goals(self, params: dict) -> dict:
        status = params.get("status")
        return await self._list_goals(GoalStatus(status) if status else None)

    async def _tool_delete_goal(self, params: dict) -> dict:
        goal_id = params.get("goal_id")
        if goal_id in self._goals:
            self._goals[goal_id].status = GoalStatus.ARCHIVED
            await self._save_goals()
            return {"success": True}
        return {"success": False, "error": "Goal not found"}

    async def _tool_create_task(self, params: dict) -> dict:
        return await self._create_task(params)

    async def _tool_update_task(self, params: dict) -> dict:
        task_id = params.get("task_id")
        if task_id not in self._tasks:
            return {"success": False, "error": "Task not found"}

        task = self._tasks[task_id]
        for key, value in params.items():
            if key != "task_id" and hasattr(task, key):
                if key == "status":
                    value = TaskStatus(value)
                elif key == "priority":
                    value = Priority(value)
                setattr(task, key, value)

        await self._save_tasks()
        return {"success": True}

    async def _tool_list_tasks(self, params: dict) -> dict:
        status = params.get("status")
        return await self._list_tasks(TaskStatus(status) if status else None)

    async def _tool_complete_task(self, params: dict) -> dict:
        task_id = params.get("task_id")
        if task_id not in self._tasks:
            return {"success": False, "error": "Task not found"}

        task = self._tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        await self._save_tasks()

        # Update goal progress
        if task.goal_id and task.goal_id in self._goals:
            goal = self._goals[task.goal_id]
            goal.current_value = min(goal.target_value, goal.current_value + 1)
            if goal.current_value >= goal.target_value:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = time.time()
            await self._save_goals()

        return {"success": True}

    async def _tool_create_habit(self, params: dict) -> dict:
        return await self._create_habit(params)

    async def _tool_update_habit(self, params: dict) -> dict:
        habit_id = params.get("habit_id")
        if habit_id not in self._habits:
            return {"success": False, "error": "Habit not found"}

        habit = self._habits[habit_id]
        for key, value in params.items():
            if key != "habit_id" and hasattr(habit, key):
                if key == "frequency":
                    value = HabitFrequency(value)
                setattr(habit, key, value)

        await self._save_habits()
        return {"success": True}

    async def _tool_list_habits(self, params: dict) -> dict:
        return await self._list_habits()

    async def _tool_complete_habit(self, params: dict) -> dict:
        habit_id = params.get("habit_id")
        if habit_id not in self._habits:
            return {"success": False, "error": "Habit not found"}

        habit = self._habits[habit_id]
        habit.completions.append(time.time())
        habit.last_completed = time.time()

        # Update streak
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        recent = [c for c in habit.completions if datetime.fromtimestamp(c).date() >= yesterday]
        if len(recent) >= habit.target_per_period:
            habit.streak += 1
            habit.longest_streak = max(habit.longest_streak, habit.streak)

        await self._save_habits()
        return {"success": True, "streak": habit.streak}

    async def _tool_create_project(self, params: dict) -> dict:
        return await self._create_project(params)

    async def _tool_update_project(self, params: dict) -> dict:
        project_id = params.get("project_id")
        if project_id not in self._projects:
            return {"success": False, "error": "Project not found"}

        project = self._projects[project_id]
        for key, value in params.items():
            if key != "project_id" and hasattr(project, key):
                setattr(project, key, value)

        await self._save_projects()
        return {"success": True}

    async def _tool_list_projects(self, params: dict) -> dict:
        return await self._list_projects()

    async def _tool_schedule_event(self, params: dict) -> dict:
        return await self._schedule_event(params)

    async def _tool_list_events(self, params: dict) -> dict:
        days = params.get("days", 7)
        return await self._list_events(days)

    async def _tool_get_schedule(self, params: dict) -> dict:
        return await self._list_events(30)

    async def _tool_get_daily_plan(self, params: dict) -> dict:
        return await self._get_daily_plan(AgentContext(conversation_id="tool"))

    async def _tool_get_weekly_plan(self, params: dict) -> dict:
        return await self._get_weekly_plan(AgentContext(conversation_id="tool"))

    async def _tool_optimize_schedule(self, params: dict) -> dict:
        return await self._optimize_schedule()

    async def _tool_progress_report(self, params: dict) -> dict:
        return await self._handle_reports("", AgentContext(conversation_id="tool"))

    async def _tool_habit_report(self, params: dict) -> dict:
        return await self._habit_report()

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "PlannerAgent",
    "Goal",
    "Task",
    "Habit",
    "Project",
    "CalendarEvent",
    "GoalStatus",
    "TaskStatus",
    "HabitFrequency",
    "Priority",
]