"""
================================================================================
VEGA AGENTS - Multi-Agent Swarm Package
================================================================================
Specialist agents orchestrated by the Butler (Orchestrator):
- Orchestrator: Master controller, task routing, coordination
- Executor: Code execution, OS control, browser automation
- Researcher: Web search, document analysis, synthesis
- Creative: Image/video/audio generation, content writing, presentations
- Guardian: Security, authentication, anomaly detection
- Planner: Goals, calendar, habits, project management
- Toolsmith: Dynamic tool creation, testing, deployment, evolution

References:
- langchain-ai/langgraph: Stateful cyclic agent graphs, multi-agent handoffs
- microsoft/autogen: Multi-agent conversation patterns
- crewAIInc/crewAI: Role-based agent design
- devansh2506/JARVIS-A-Multi-Agentic-Personal-Assistant: LangGraph supervisor
- open-interpreter/open-interpreter: Code execution sandbox
================================================================================
"""

from .base_agent import BaseAgent, AgentConfig, AgentState, AgentMessage
from .orchestrator import Orchestrator, OrchestratorConfig
from .executor_agent import ExecutorAgent
from .researcher_agent import ResearcherAgent
from .creative_agent import CreativeAgent
from .guardian_agent import GuardianAgent
from .planner_agent import PlannerAgent
from .toolsmith_agent import ToolsmithAgent

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentState",
    "AgentMessage",
    "Orchestrator",
    "OrchestratorConfig",
    "ExecutorAgent",
    "ResearcherAgent",
    "CreativeAgent",
    "GuardianAgent",
    "PlannerAgent",
    "ToolsmithAgent",
]