from typing import Dict, Optional
from .base import BaseAgent
from .orchestrator import orchestrator_agent
from .decomposition import decomposition_agent
from .retrieval import retrieval_agent
from .critique import critique_agent
from .synthesis import synthesis_agent
from .compression import compression_agent


class AgentRegistry:
    """Registry for all agents"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {
            "orchestrator": orchestrator_agent,
            "decomposition_agent": decomposition_agent,
            "retrieval_agent": retrieval_agent,
            "critique_agent": critique_agent,
            "synthesis_agent": synthesis_agent,
            "compression_agent": compression_agent
        }

    def register_agent(self, agent: BaseAgent):
        """Register a new agent"""
        self._agents[agent.name] = agent

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self._agents.get(name)

    def list_agents(self) -> list:
        """List all registered agents"""
        return [
            {
                "name": name,
                "max_context_budget": agent.max_context_budget
            }
            for name, agent in self._agents.items()
        ]

    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all agents"""
        return self._agents


# Global registry
agent_registry = AgentRegistry()