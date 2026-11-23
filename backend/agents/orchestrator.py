"""
JARVIS OMEGA - Orchestrator (The Butler)
Multi-agent orchestration using LangGraph patterns:
- Task analysis & decomposition
- Agent routing & delegation
- Hierarchical supervision
- Human-in-the-loop checkpoints
- Streaming response aggregation
- Cross-agent memory sharing
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

from config import get_settings, settings
from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentState,
    AgentMessage,
    MessageType,
    AgentContext,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger("jarvis.agents.orchestrator")


# =============================================================================
# DATA MODELS
# =============================================================================

class RoutingStrategy(str, Enum):
    DIRECT = "direct"              # Single agent
    SEQUENTIAL = "sequential"      # Pipeline: A -> B -> C
    PARALLEL = "parallel"          # Fan-out: A + B + C
    HIERARCHICAL = "hierarchical"  # Supervisor delegates
    GROUP_CHAT = "group_chat"      # Collaborative discussion


@dataclass
class TaskPlan:
    """Decomposed task plan."""
    task_id: str
    original_task: str
    subtasks: list[dict]  # {"agent": "", "task": "", "depends_on": []}
    strategy: RoutingStrategy
    estimated_steps: int
    requires_human: bool = False


@dataclass
class AgentRegistry:
    """Registry of available agents."""
    agents: dict[str, BaseAgent] = field(default_factory=dict)
    capabilities: dict[str, list[str]] = field(default_factory=dict)  # agent -> capabilities

    def register(self, agent: BaseAgent) -> None:
        self.agents[agent.config.name] = agent
        self.capabilities[agent.config.name] = agent.config.tools

    def get_agent(self, name: str) -> BaseAgent | None:
        return self.agents.get(name)

    def find_agent_for_task(self, task: str) -> str | None:
        """Find best agent for task based on capabilities."""
        task_lower = task.lower()

        # Keyword-based routing
        routing_keywords = {
            "executor": ["code", "run", "execute", "script", "terminal", "command", "install", "file", "folder", "browser", "click", "type", "automat"],
            "researcher": ["search", "find", "research", "look up", "investigate", "analyze", "summarize", "compare", "news", "weather", "document"],
            "creative": ["create", "generate", "image", "video", "music", "write", "design", "draw", "art", "presentation", "slide"],
            "guardian": ["security", "auth", "face", "intruder", "lock", "unlock", "permission", "monitor", "alert", "camera"],
            "planner": ["plan", "schedule", "calendar", "remind", "deadline", "goal", "habit", "routine", "organize", "todo"],
            "toolsmith": ["tool", "function", "api", "create tool", "build tool", "automation", "workflow", "script"],
        }

        scores = {}
        for agent, keywords in routing_keywords.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[agent] = score

        if scores:
            return max(scores, key=scores.get)

        return "executor"  # Default


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    # Routing
    default_strategy: RoutingStrategy = RoutingStrategy.HIERARCHICAL
    max_delegation_depth: int = 3
    max_parallel_agents: int = 3

    # LLM for planning
    planner_model: str = "orchestrator"
    planner_temperature: float = 0.3

    # Human-in-the-loop
    enable_human_in_loop: bool = True
    human_approval_required: list[str] = field(default_factory=lambda: [
        "financial", "security", "system_modification", "external_api", "delete"
    ])

    # Streaming
    stream_final_response: bool = True

    # Memory
    share_memory_across_agents: bool = True

    # Timeouts
    task_timeout: float = 120.0
    agent_response_timeout: float = 60.0


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class Orchestrator(BaseAgent):
    """
    Master orchestrator - The Butler.
    Coordinates all specialist agents, plans tasks, routes work.
    """

    def __init__(
        self,
        event_bus: Any = None,
        memory_manager: Any = None,
        task_queue: Any = None,
        websocket_manager: Any = None,
        llm_client: Any = None,
        config: OrchestratorConfig | None = None,
    ):
        # Create agent config
        agent_config = AgentConfig(
            name="orchestrator",
            role="supervisor",
            description="Master coordinator - analyzes tasks, delegates to specialists, synthesizes results",
            model="orchestrator",
            temperature=0.3,
            system_prompt=self._get_system_prompt(),
            tools=["delegate_task", "query_agent", "broadcast_message"],
            can_delegate=True,
            can_handoff=True,
            stream_response=True,
        )

        super().__init__(
            config=agent_config,
            event_bus=event_bus,
            memory_manager=memory_manager,
            tool_registry=None,  # Will be set later
            llm_client=llm_client,
        )

        self.orchestrator_config = config or OrchestratorConfig()
        self.task_queue = task_queue
        self.websocket_manager = websocket_manager

        # Agent registry
        self.registry = AgentRegistry()

        # Active workflows
        self._active_workflows: dict[str, dict] = {}

        # Planning LLM (separate from main)
        self._planner_llm = None

        # Human-in-the-loop
        self._pending_approvals: dict[str, asyncio.Future] = {}

        # Stats
        self._orchestrator_stats = {
            "tasks_planned": 0,
            "tasks_delegated": 0,
            "tasks_completed": 0,
            "human_approvals": 0,
        }

    def _get_system_prompt(self) -> str:
        return """You are the Orchestrator (The Butler) - the master coordinator of VEGA's multi-agent system.

Your role:
1. ANALYZE incoming tasks and understand user intent
2. DECOMPOSE complex tasks into subtasks for specialist agents
3. ROUTE work to the right agents based on their capabilities
4. COORDINATE multi-agent workflows (sequential, parallel, hierarchical)
5. SYNTHESIZE results from multiple agents into coherent responses
6. ESCALATE to human for approval when needed

Available Specialist Agents:
- executor: Code execution, OS control, browser automation, file operations, terminal commands
- researcher: Web search, document analysis, fact-finding, synthesis, news/weather
- creative: Image/video/music generation, content creation, design, presentations
- guardian: Security, face authentication, intrusion detection, access control, monitoring
- planner: Goal setting, calendar management, habit tracking, project planning, reminders
- toolsmith: Dynamic tool creation, API integration, workflow automation, self-evolution

Routing Rules:
- Code/terminal/file/browser tasks -> executor
- Search/research/analysis -> researcher
- Creative generation -> creative
- Security/auth/monitoring -> guardian
- Planning/scheduling/goals -> planner
- Tool creation/automation -> toolsmith
- Complex multi-step -> decompose and route to multiple agents

Always think step-by-step. Be concise but thorough in delegation.""".

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def _agent_initialize(self) -> None:
        """Initialize orchestrator-specific components."""
        # Register built-in tools
        if self.tool_registry:
            self.tool_registry.register("delegate_task", self._tool_delegate_task)
            self.tool_registry.register("query_agent", self._tool_query_agent)
            self.tool_registry.register("broadcast_message", self._tool_broadcast_message)

        # Initialize planner LLM
        await self._init_planner_llm()

        logger.info("Orchestrator initialized")

    async def _agent_shutdown(self) -> None:
        """Cleanup."""
        # Cancel all active workflows
        for workflow_id in list(self._active_workflows.keys()):
            await self._cancel_workflow(workflow_id)

        # Resolve pending approvals
        for future in self._pending_approvals.values():
            if not future.done():
                future.set_exception(RuntimeError("Orchestrator shutting down"))

    async def _init_planner_llm(self) -> None:
        """Initialize dedicated planner LLM."""
        # This would use the configured planner model
        # For now, use main LLM client
        self._planner_llm = self.llm_client

    # -------------------------------------------------------------------------
    # AGENT REGISTRATION
    # -------------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialist agent."""
        self.registry.register(agent)
        logger.info(f"Registered agent: {agent.config.name}")

    def get_agent(self, name: str) -> BaseAgent | None:
        return self.registry.get_agent(name)

    # -------------------------------------------------------------------------
    # MAIN TASK PROCESSING
    # -------------------------------------------------------------------------

    async def process_task(
        self,
        task: str,
        context: AgentContext,
    ) -> Any:
        """Main entry point - process task through orchestration."""
        self._orchestrator_stats["tasks_planned"] += 1

        # Check for human approval requirement
        if self._requires_human_approval(task):
            approved = await self._request_human_approval(task, context)
            if not approved:
                return {"status": "rejected", "reason": "Human approval denied"}

        # Create task plan
        plan = await self._create_task_plan(task, context)

        # Execute plan
        result = await self._execute_plan(plan, context)

        self._orchestrator_stats["tasks_completed"] += 1
        return result

    async def process_task_stream(
        self,
        task: str,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream orchestration response."""
        yield f"🔍 Analyzing task: {task[:100]}...\n\n"

        # Check approval
        if self._requires_human_approval(task):
            yield "⏳ Requesting human approval...\n"
            approved = await self._request_human_approval(task, context)
            if not approved:
                yield "❌ Task rejected by human.\n"
                return
            yield "✅ Approved. Proceeding...\n\n"

        # Plan
        yield "📋 Creating task plan...\n"
        plan = await self._create_task_plan(task, context)
        yield f"📋 Plan: {plan.strategy.value} with {len(plan.subtasks)} steps\n\n"

        # Execute with streaming
        async for chunk in self._execute_plan_stream(plan, context):
            yield chunk

    # -------------------------------------------------------------------------
    # TASK PLANNING
    # -------------------------------------------------------------------------

    async def _create_task_plan(self, task: str, context: AgentContext) -> TaskPlan:
        """Create execution plan using planner LLM."""
        task_id = str(uuid.uuid4())

        # Simple heuristic planning (replace with LLM planning for complex tasks)
        if self._is_simple_task(task):
            # Direct routing
            agent_name = self.registry.find_agent_for_task(task)
            plan = TaskPlan(
                task_id=task_id,
                original_task=task,
                subtasks=[{
                    "agent": agent_name,
                    "task": task,
                    "depends_on": [],
                }],
                strategy=RoutingStrategy.DIRECT,
                estimated_steps=1,
            )
        else:
            # Complex task - use LLM to decompose
            plan = await self._llm_decompose_task(task, task_id)

        return plan

    def _is_simple_task(self, task: str) -> bool:
        """Check if task can be handled by single agent."""
        simple_indicators = [
            "open", "close", "run", "execute", "search", "find",
            "play", "pause", "volume", "screenshot", "remind",
        ]
        task_lower = task.lower()
        return any(ind in task_lower for ind in simple_indicators) and len(task) < 200

    async def _llm_decompose_task(self, task: str, task_id: str) -> TaskPlan:
        """Use LLM to decompose complex task."""
        if not self._planner_llm:
            return self._fallback_decompose(task, task_id)

        prompt = f"""Decompose this task into subtasks for specialist agents:

Task: {task}

Available agents:
- executor: code, terminal, files, browser, OS automation
- researcher: web search, documents, analysis, synthesis
- creative: images, video, music, writing, design
- guardian: security, auth, monitoring, alerts
- planner: calendar, goals, habits, scheduling
- toolsmith: create tools, APIs, automation

Respond with JSON:
{{
  "strategy": "sequential|parallel|hierarchical",
  "subtasks": [
    {{"agent": "executor", "task": "specific task", "depends_on": []}},
    ...
  ]
}}"""

        try:
            response = await self._planner_llm([
                {"role": "system", "content": "Decompose tasks into subtasks for specialist agents. Return JSON only."},
                {"role": "user", "content": prompt},
            ])

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                return TaskPlan(
                    task_id=task_id,
                    original_task=task,
                    subtasks=plan_data.get("subtasks", []),
                    strategy=RoutingStrategy(plan_data.get("strategy", "sequential")),
                    estimated_steps=len(plan_data.get("subtasks", [])),
                )
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")

        return self._fallback_decompose(task, task_id)

    def _fallback_decompose(self, task: str, task_id: str) -> TaskPlan:
        """Fallback heuristic decomposition."""
        task_lower = task.lower()
        subtasks = []

        if any(w in task_lower for w in ["search", "find", "research"]):
            subtasks.append({"agent": "researcher", "task": f"Research: {task}", "depends_on": []})

        if any(w in task_lower for w in ["code", "script", "program", "run", "execute"]):
            subtasks.append({"agent": "executor", "task": f"Execute: {task}", "depends_on": []})

        if any(w in task_lower for w in ["create", "generate", "image", "video", "music"]):
            subtasks.append({"agent": "creative", "task": f"Create: {task}", "depends_on": []})

        if any(w in task_lower for w in ["plan", "schedule", "calendar", "remind"]):
            subtasks.append({"agent": "planner", "task": f"Plan: {task}", "depends_on": []})

        if any(w in task_lower for w in ["security", "auth", "monitor", "alert"]):
            subtasks.append({"agent": "guardian", "task": f"Security: {task}", "depends_on": []})

        if not subtasks:
            subtasks.append({"agent": "executor", "task": task, "depends_on": []})

        if len(subtasks) == 1:
            strategy = RoutingStrategy.DIRECT
        elif any("depends_on" in s and s["depends_on"] for s in subtasks):
            strategy = RoutingStrategy.SEQUENTIAL
        else:
            strategy = RoutingStrategy.PARALLEL

        return TaskPlan(
            task_id=task_id,
            original_task=task,
            subtasks=subtasks,
            strategy=strategy,
            estimated_steps=len(subtasks),
        )

    # -------------------------------------------------------------------------
    # PLAN EXECUTION
    # -------------------------------------------------------------------------

    async def _execute_plan(self, plan: TaskPlan, context: AgentContext) -> Any:
        """Execute task plan based on strategy."""
        if plan.strategy == RoutingStrategy.DIRECT:
            return await self._execute_direct(plan, context)
        elif plan.strategy == RoutingStrategy.SEQUENTIAL:
            return await self._execute_sequential(plan, context)
        elif plan.strategy == RoutingStrategy.PARALLEL:
            return await self._execute_parallel(plan, context)
        elif plan.strategy == RoutingStrategy.HIERARCHICAL:
            return await self._execute_hierarchical(plan, context)
        else:
            return await self._execute_direct(plan, context)

    async def _execute_plan_stream(
        self,
        plan: TaskPlan,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream plan execution."""
        if plan.strategy == RoutingStrategy.DIRECT:
            async for chunk in self._execute_direct_stream(plan, context):
                yield chunk
        elif plan.strategy == RoutingStrategy.SEQUENTIAL:
            async for chunk in self._execute_sequential_stream(plan, context):
                yield chunk
        elif plan.strategy == RoutingStrategy.PARALLEL:
            async for chunk in self._execute_parallel_stream(plan, context):
                yield chunk
        else:
            async for chunk in self._execute_direct_stream(plan, context):
                yield chunk

    async def _execute_direct(self, plan: TaskPlan, context: AgentContext) -> Any:
        """Execute single agent task."""
        subtask = plan.subtasks[0]
        agent_name = subtask["agent"]
        task = subtask["task"]

        agent = self.registry.get_agent(agent_name)
        if not agent:
            return {"error": f"Agent '{agent_name}' not available"}

        self._orchestrator_stats["tasks_delegated"] += 1

        message = AgentMessage.create_request(
            source="orchestrator",
            target=agent_name,
            task=task,
            context={"parent_context": context.metadata},
            conversation_id=context.conversation_id,
        )

        result = await agent.process_task(task, context)
        return result

    async def _execute_direct_stream(
        self,
        plan: TaskPlan,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream direct execution."""
        subtask = plan.subtasks[0]
        agent_name = subtask["agent"]
        task = subtask["task"]

        yield f"🤖 Delegating to **{agent_name}**: {task[:100]}...\n\n"

        agent = self.registry.get_agent(agent_name)
        if not agent:
            yield f"❌ Agent '{agent_name}' not available\n"
            return

        self._orchestrator_stats["tasks_delegated"] += 1

        if agent.config.stream_response:
            async for chunk in agent.process_task_stream(task, context):
                yield chunk
        else:
            result = await agent.process_task(task, context)
            yield str(result)

    async def _execute_sequential(
        self,
        plan: TaskPlan,
        context: AgentContext,
    ) -> Any:
        """Execute subtasks sequentially, passing results."""
        results = {}
        previous_result = None

        for i, subtask in enumerate(plan.subtasks):
            agent_name = subtask["agent"]
            task = subtask["task"]

            # Inject previous result into context
            if previous_result:
                task = f"{task}\n\nPrevious result: {previous_result}"

            agent = self.registry.get_agent(agent_name)
            if not agent:
                results[agent_name] = {"error": f"Agent not available"}
                continue

            self._orchestrator_stats["tasks_delegated"] += 1
            result = await agent.process_task(task, context)
            results[agent_name] = result
            previous_result = result

        return self._synthesize_results(results, plan)

    async def _execute_sequential_stream(
        self,
        plan: TaskPlan,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream sequential execution."""
        previous_result = None

        for i, subtask in enumerate(plan.subtasks):
            agent_name = subtask["agent"]
            task = subtask["task"]

            yield f"🔄 **Step {i+1}/{len(plan.subtasks)}** → **{agent_name}**\n"

            if previous_result:
                task = f"{task}\n\nPrevious result: {previous_result}"

            agent = self.registry.get_agent(agent_name)
            if not agent:
                yield f"❌ Agent '{agent_name}' not available\n\n"
                continue

            self._orchestrator_stats["tasks_delegated"] += 1

            if agent.config.stream_response:
                async for chunk in agent.process_task_stream(task, context):
                    yield chunk
            else:
                result = await agent.process_task(task, context)
                yield str(result) + "\n\n"

            previous_result = result if isinstance(result, str) else str(result)

        # Final synthesis
        yield "\n---\n✅ **Sequence complete**\n"

    async def _execute_parallel(
        self,
        plan: TaskPlan,
        context: AgentContext,
    ) -> Any:
        """Execute subtasks in parallel."""
        tasks = []

        for subtask in plan.subtasks[:self.orchestrator_config.max_parallel_agents]:
            agent_name = subtask["agent"]
            task = subtask["task"]

            agent = self.registry.get_agent(agent_name)
            if not agent:
                continue

            self._orchestrator_stats["tasks_delegated"] += 1
            tasks.append(agent.process_task(task, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results back to agents
        result_map = {}
        for i, subtask in enumerate(plan.subtasks):
            agent_name = subtask["agent"]
            if i < len(results):
                result = results[i]
                if isinstance(result, Exception):
                    result_map[agent_name] = {"error": str(result)}
                else:
                    result_map[agent_name] = result

        return self._synthesize_results(result_map, plan)

    async def _execute_parallel_stream(
        self,
        plan: TaskPlan,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream parallel execution."""
        yield f"⚡ **Parallel execution** ({len(plan.subtasks)} agents)\n\n"

        async def run_agent_stream(agent_name: str, task: str):
            agent = self.registry.get_agent(agent_name)
            if not agent:
                return agent_name, "Agent not available"

            self._orchestrator_stats["tasks_delegated"] += 1
            chunks = []
            async for chunk in agent.process_task_stream(task, context):
                chunks.append(chunk)
            return agent_name, "".join(chunks)

        tasks = [
            run_agent_stream(s["agent"], s["task"])
            for s in plan.subtasks[:self.orchestrator_config.max_parallel_agents]
        ]

        results = await asyncio.gather(*tasks)

        for agent_name, result in results:
            yield f"📦 **{agent_name}**:\n{result}\n\n"

        yield "\n---\n✅ **Parallel execution complete**\n"

    async def _execute_hierarchical(
        self,
        plan: TaskPlan,
        context: AgentContext,
    ) -> Any:
        """Hierarchical: orchestrator manages sub-orchestrations."""
        return await self._execute_sequential(plan, context)

    # -------------------------------------------------------------------------
    # RESULT SYNTHESIS
    # -------------------------------------------------------------------------

    def _synthesize_results(self, results: dict[str, Any], plan: TaskPlan) -> Any:
        """Synthesize results from multiple agents."""
        if len(results) == 1:
            return list(results.values())[0]

        synthesis = f"Task: {plan.original_task}\n\n"
        synthesis += "Results from specialist agents:\n\n"

        for agent_name, result in results.items():
            synthesis += f"--- {agent_name.upper()} ---\n"
            if isinstance(result, dict) and "error" in result:
                synthesis += f"❌ Error: {result['error']}\n"
            else:
                synthesis += f"{result}\n"
            synthesis += "\n"

        # If we have LLM, do intelligent synthesis
        if self.llm_client:
            try:
                synth_prompt = f"""Synthesize these agent results into a coherent response:

Original task: {plan.original_task}

Results:
{json.dumps(results, indent=2)}

Provide a unified, well-structured response."""
                # This would call LLM
                pass
            except Exception:
                pass

        return synthesis

    # -------------------------------------------------------------------------
    # HUMAN-IN-THE-LOOP
    # -------------------------------------------------------------------------

    def _requires_human_approval(self, task: str) -> bool:
        """Check if task requires human approval."""
        if not self.orchestrator_config.enable_human_in_loop:
            return False

        task_lower = task.lower()
        return any(keyword in task_lower for keyword in self.orchestrator_config.human_approval_required)

    async def _request_human_approval(self, task: str, context: AgentContext) -> bool:
        """Request human approval for sensitive task."""
        approval_id = str(uuid.uuid4())
        future = asyncio.Future()
        self._pending_approvals[approval_id] = future

        # Emit approval request
        if self.event_bus:
            await self.event_bus.emit("human.approval_requested", {
                "approval_id": approval_id,
                "task": task,
                "context": context.metadata,
                "timeout": 300,  # 5 minutes
            })

        # Broadcast to HUD
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "approval_request",
                "payload": {
                    "approval_id": approval_id,
                    "task": task,
                    "message": f"Approval required for: {task}",
                },
            }, room="hud")

        self._orchestrator_stats["human_approvals"] += 1

        try:
            approved = await asyncio.wait_for(future, timeout=300)
            return approved
        except asyncio.TimeoutError:
            logger.warning(f"Approval timeout for task: {task}")
            return False
        finally:
            self._pending_approvals.pop(approval_id, None)

    def approve_task(self, approval_id: str, approved: bool) -> bool:
        """Resolve human approval (called from UI)."""
        future = self._pending_approvals.get(approval_id)
        if future and not future.done():
            future.set_result(approved)
            return True
        return False

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_delegate_task(self, parameters: dict) -> dict:
        """Tool: Delegate task to specialist agent."""
        target = parameters.get("agent")
        task = parameters.get("task")
        context = parameters.get("context", {})

        if not target or not task:
            return {"error": "agent and task required"}

        agent = self.registry.get_agent(target)
        if not agent:
            return {"error": f"Agent '{target}' not found"}

        result = await agent.process_task(task, AgentContext(
            conversation_id=str(uuid.uuid4()),
            metadata=context,
        ))

        return {"result": result}

    async def _tool_query_agent(self, parameters: dict) -> dict:
        """Tool: Query agent status/capabilities."""
        target = parameters.get("agent")

        if target:
            agent = self.registry.get_agent(target)
            if agent:
                return {"agent": target, "status": agent.state.value, "stats": agent.get_stats()}
            return {"error": f"Agent '{target}' not found"}

        # Return all agents
        return {
            "agents": [
                {"name": name, "status": agent.state.value, "role": agent.config.role}
                for name, agent in self.registry.agents.items()
            ]
        }

    async def _tool_broadcast_message(self, parameters: dict) -> dict:
        """Tool: Broadcast message to all agents/HUD."""
        message = parameters.get("message", "")
        room = parameters.get("room", "hud")

        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "orchestrator_broadcast",
                "payload": {"message": message},
            }, room=room)

        return {"status": "broadcast_sent"}

    # -------------------------------------------------------------------------
    # WORKFLOW MANAGEMENT
    # -------------------------------------------------------------------------

    async def _cancel_workflow(self, workflow_id: str) -> None:
        """Cancel active workflow."""
        workflow = self._active_workflows.pop(workflow_id, None)
        if workflow:
            for task in workflow.get("tasks", []):
                if not task.done():
                    task.cancel()

    # -------------------------------------------------------------------------
    # CHAT INTERFACE (for direct conversation)
    # -------------------------------------------------------------------------

    async def chat(self, message: str, context: dict, stream: bool = False) -> Any:
        """Direct chat with orchestrator."""
        agent_context = AgentContext(
            conversation_id=context.get("conversation_id", str(uuid.uuid4())),
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            metadata=context,
        )

        if stream:
            return self.process_task_stream(message, agent_context)
        else:
            return await self.process_task(message, agent_context)

    async def chat_stream(self, message: str, context: dict) -> AsyncGenerator[str, None]:
        """Streaming chat."""
        agent_context = AgentContext(
            conversation_id=context.get("conversation_id", str(uuid.uuid4())),
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            metadata=context,
        )

        async for chunk in self.process_task_stream(message, agent_context):
            yield chunk

    # -------------------------------------------------------------------------
    # COMMAND HANDLING (from WebSocket)
    # -------------------------------------------------------------------------

    async def handle_command(self, client_id: str, payload: dict) -> dict:
        """Handle command from WebSocket."""
        command = payload.get("command")
        args = payload.get("args", {})

        if command == "execute":
            task = args.get("task", "")
            agent_name = args.get("agent")
            context = args.get("context", {})

            agent_context = AgentContext(
                conversation_id=context.get("conversation_id", str(uuid.uuid4())),
                metadata=context,
            )

            if agent_name:
                agent = self.registry.get_agent(agent_name)
                if agent:
                    result = await agent.process_task(task, agent_context)
                    return {"status": "success", "result": result}
                return {"status": "error", "message": f"Agent '{agent_name}' not found"}
            else:
                result = await self.process_task(task, agent_context)
                return {"status": "success", "result": result}

        elif command == "get_agents":
            return {"status": "success", "agents": list(self.registry.agents.keys())}

        elif command == "get_agent_status":
            agent_name = args.get("agent")
            agent = self.registry.get_agent(agent_name)
            if agent:
                return {"status": "success", "agent": agent_name, "state": agent.state.value}
            return {"status": "error", "message": "Agent not found"}

        return {"status": "error", "message": f"Unknown command: {command}"}

    # -------------------------------------------------------------------------
    # STATS & HEALTH
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        base_stats = super().get_stats()
        return {
            **base_stats,
            **self._orchestrator_stats,
            "registered_agents": list(self.registry.agents.keys()),
            "active_workflows": len(self._active_workflows),
            "pending_approvals": len(self._pending_approvals),
        }

    async def health_check(self) -> dict[str, Any]:
        health = await super().health_check()
        health["agents"] = {}
        for name, agent in self.registry.agents.items():
            health["agents"][name] = await agent.health_check()
        return health


# =============================================================================
# FACTORY
# =============================================================================

async def create_orchestrator(
    event_bus: Any = None,
    memory_manager: Any = None,
    task_queue: Any = None,
    websocket_manager: Any = None,
    llm_client: Any = None,
    config: OrchestratorConfig | None = None,
) -> Orchestrator:
    """Factory to create and initialize orchestrator."""
    orchestrator = Orchestrator(
        event_bus=event_bus,
        memory_manager=memory_manager,
        task_queue=task_queue,
        websocket_manager=websocket_manager,
        llm_client=llm_client,
        config=config,
    )
    await orchestrator.initialize()
    return orchestrator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "RoutingStrategy",
    "TaskPlan",
    "AgentRegistry",
    "create_orchestrator",
]