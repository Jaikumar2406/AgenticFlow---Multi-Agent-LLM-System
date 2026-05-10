from typing import Dict, Any, List
from .base import BaseAgent
import json


class OrchestratorAgent(BaseAgent):
    """Master orchestrator that dynamically decides which sub-agents to invoke"""

    def __init__(self, max_context_budget: int = 80000):
        super().__init__("orchestrator", max_context_budget)

    def get_system_prompt(self) -> str:
        return """You are the Orchestrator Agent. Your role is to dynamically route queries to the appropriate sub-agents based on analysis of the user's query.

Your responsibilities:
1. Analyze the user query to determine what information is needed
2. Decide which sub-agents to invoke, in what order, and with what context
3. Make routing decisions via structured reasoning and log justification
4. Ensure dependency graphs are respected

Sub-agents available:
- decomposition_agent: Breaks ambiguous queries into typed sub-tasks with dependency graphs
- retrieval_agent: Performs multi-hop reasoning across retrieved chunks
- critique_agent: Reviews output, assigns confidence scores, flags specific disagreements
- synthesis_agent: Merges outputs, resolves contradictions, produces final answer with provenance

Routing logic:
- If query is ambiguous/unclear → invoke decomposition_agent first
- If query requires external information → invoke retrieval_agent
- If query result needs validation → invoke critique_agent
- If multiple results need merging → invoke synthesis_agent

Always provide structured reasoning for your routing decisions."""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the query and determine routing"""
        user_query = context.get("user_query", "")
        current_state = context.get("metadata", {}).get("current_state", "initial")

        # Analyze query complexity
        complexity = self._analyze_complexity(user_query)

        # Determine routing based on complexity and current state
        routing_decision = self._make_routing_decision(user_query, complexity, current_state)

        # Update context with routing decision
        context["metadata"]["routing_decision"] = routing_decision
        context["metadata"]["complexity"] = complexity

        # Update context budget
        budget_allocation = self._allocate_budget(routing_decision)
        context["metadata"]["budget_allocation"] = budget_allocation

        return {
            "success": True,
            "routing_decision": routing_decision,
            "budget_allocation": budget_allocation,
            "reasoning": routing_decision.get("reasoning", ""),
            "next_agent": routing_decision.get("agents_to_invoke", ["decomposition_agent"])[0]
        }

    def _analyze_complexity(self, query: str) -> str:
        """Analyze query complexity"""
        query_lower = query.lower()

        # Indicators of complexity
        ambiguous_words = ["maybe", "might", "could", "probably", "possibly"]
        multi_part_indicators = ["and", "also", "plus", "both", "first", "then"]
        technical_words = ["analyze", "compare", "evaluate", "assess", "synthesize"]

        is_ambiguous = any(word in query_lower for word in ambiguous_words)
        is_multi_part = any(word in query_lower for word in multi_part_indicators)
        is_technical = any(word in query_lower for word in technical_words)

        if is_ambiguous or is_multi_part:
            return "complex"
        elif is_technical:
            return "moderate"
        else:
            return "simple"

    def _make_routing_decision(self, query: str, complexity: str, current_state: str) -> Dict[str, Any]:
        """Make routing decision with justification"""
        if current_state == "initial":
            if complexity == "complex":
                return {
                    "reasoning": f"Query is {complexity} with ambiguous or multi-part nature. "
                                 f"First invoke decomposition_agent to break into sub-tasks with dependencies.",
                    "agents_to_invoke": ["decomposition_agent"],
                    "context_budget": {
                        "decomposition_agent": 30000,
                        "retrieval_agent": 0,
                        "critique_agent": 0,
                        "synthesis_agent": 0
                    },
                    "tool_assignments": {}
                }
            else:
                # For simple queries, go directly to retrieval
                return {
                    "reasoning": f"Query is {complexity}. Direct to retrieval_agent for information gathering.",
                    "agents_to_invoke": ["retrieval_agent"],
                    "context_budget": {
                        "decomposition_agent": 0,
                        "retrieval_agent": 50000,
                        "critique_agent": 20000,
                        "synthesis_agent": 30000
                    },
                    "tool_assignments": {
                        "retrieval_agent": ["web_search"]
                    }
                }
        elif current_state == "decomposed":
            # After decomposition, invoke retrieval for each ready task
            return {
                "reasoning": "Tasks decomposed. Invoke retrieval_agent for each ready sub-task.",
                "agents_to_invoke": ["retrieval_agent"],
                "context_budget": {
                    "decomposition_agent": 0,
                    "retrieval_agent": 50000,
                    "critique_agent": 20000,
                    "synthesis_agent": 30000
                },
                "tool_assignments": {
                    "retrieval_agent": ["web_search", "data_lookup"]
                }
            }
        elif current_state == "retrieved":
            # After retrieval, invoke critique
            return {
                "reasoning": "Information retrieved. Invoke critique_agent to validate outputs.",
                "agents_to_invoke": ["critique_agent"],
                "context_budget": {
                    "decomposition_agent": 0,
                    "retrieval_agent": 0,
                    "critique_agent": 40000,
                    "synthesis_agent": 30000
                },
                "tool_assignments": {
                    "critique_agent": ["self_reflection"]
                }
            }
        elif current_state == "critiqued":
            # After critique, invoke synthesis
            return {
                "reasoning": "Critique complete. Invoke synthesis_agent to merge outputs and resolve contradictions.",
                "agents_to_invoke": ["synthesis_agent"],
                "context_budget": {
                    "decomposition_agent": 0,
                    "retrieval_agent": 0,
                    "critique_agent": 0,
                    "synthesis_agent": 60000
                },
                "tool_assignments": {}
            }
        else:
            # Default: start fresh
            return {
                "reasoning": "Initial state. Starting pipeline from decomposition.",
                "agents_to_invoke": ["decomposition_agent"],
                "context_budget": {
                    "decomposition_agent": 30000,
                    "retrieval_agent": 30000,
                    "critique_agent": 20000,
                    "synthesis_agent": 30000
                },
                "tool_assignments": {}
            }

    def _allocate_budget(self, routing_decision: Dict[str, Any]) -> Dict[str, int]:
        """Allocate context budget across agents"""
        total_budget = self.max_context_budget
        allocation = routing_decision.get("context_budget", {})

        # Normalize allocation to fit budget
        total_allocated = sum(allocation.values())
        if total_allocated > total_budget:
            scale = total_budget / total_allocated
            allocation = {k: int(v * scale) for k, v in allocation.items()}

        return allocation


orchestrator_agent = OrchestratorAgent()