from typing import Dict, Any, List
from uuid import UUID, uuid4
from datetime import datetime
import json


class MetaAgent:
    """Meta-agent that identifies worst-performing prompts and proposes rewrites"""

    def __init__(self):
        # Default prompts for each agent
        self.default_prompts = {
            "orchestrator": "You are the Orchestrator Agent. Your role is to dynamically route queries...",
            "decomposition_agent": "You are the Decomposition Agent. Your role is to break ambiguous...",
            "retrieval_agent": "You are the Retrieval Agent. Your role is to perform multi-hop reasoning...",
            "critique_agent": "You are the Critique Agent. Your role is to review the output...",
            "synthesis_agent": "You are the Synthesis Agent. Your role is to merge outputs..."
        }

        self.current_prompts = self.default_prompts.copy()

    def analyze_failures(self, eval_results: List[Dict]) -> Dict[str, Any]:
        """Analyze failure cases and identify worst-performing prompts"""
        # Aggregate scores by agent
        agent_performance = {}

        for result in eval_results:
            scores = result.get("scores", {})
            justifications = result.get("justifications", {})

            # Determine which agent likely caused failure
            lowest_dim = min(scores, key=scores.get)
            lowest_score = scores[lowest_dim]

            if lowest_score < 0.7:
                # Determine agent based on dimension
                agent = self._map_dimension_to_agent(lowest_dim)
                if agent not in agent_performance:
                    agent_performance[agent] = {
                        "failure_count": 0,
                        "total_score": 0,
                        "dimensions": []
                    }
                agent_performance[agent]["failure_count"] += 1
                agent_performance[agent]["total_score"] += lowest_score
                agent_performance[agent]["dimensions"].append(lowest_dim)

        # Find worst performing agent
        worst_agent = None
        worst_score = float('inf')

        for agent, perf in agent_performance.items():
            avg_score = perf["total_score"] / perf["failure_count"]
            if avg_score < worst_score:
                worst_score = avg_score
                worst_agent = agent

        return {
            "worst_agent": worst_agent,
            "worst_score": worst_score,
            "performance": agent_performance
        }

    def _map_dimension_to_agent(self, dimension: str) -> str:
        """Map scoring dimension to responsible agent"""
        mapping = {
            "answer_correctness": "retrieval_agent",
            "citation_accuracy": "retrieval_agent",
            "contradiction_resolution": "synthesis_agent",
            "tool_selection_efficiency": "orchestrator",
            "context_budget_compliance": "orchestrator",
            "critique_agreement_rate": "critique_agent"
        }
        return mapping.get(dimension, "orchestrator")

    def propose_rewrite(self, agent_name: str, failure_analysis: Dict) -> Dict[str, Any]:
        """Propose a rewritten prompt with diff and justification"""
        original_prompt = self.current_prompts.get(agent_name, "")

        # Generate proposed rewrite based on failures
        proposed_prompt = self._generate_rewrite(agent_name, failure_analysis)

        # Generate diff
        diff = self._generate_diff(original_prompt, proposed_prompt)

        rewrite = {
            "id": str(uuid4()),
            "agent_name": agent_name,
            "original_prompt": original_prompt,
            "proposed_prompt": proposed_prompt,
            "justification": f"Rewrite proposed due to performance issues. "
                            f"Failure analysis: {failure_analysis}",
            "diff": diff,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }

        return rewrite

    def _generate_rewrite(self, agent_name: str, analysis: Dict) -> str:
        """Generate a rewritten prompt based on analysis"""
        base_prompt = self.default_prompts.get(agent_name, "")

        # Add specific improvements based on agent type
        if agent_name == "orchestrator":
            return base_prompt + "\n\nIMPORTANT: Optimize for minimal tool usage. Prioritize efficiency."
        elif agent_name == "decomposition_agent":
            return base_prompt + "\n\nIMPORTANT: Focus on clear task boundaries and minimal dependencies."
        elif agent_name == "retrieval_agent":
            return base_prompt + "\n\nIMPORTANT: Ensure all claims are cited with source chunks."
        elif agent_name == "critique_agent":
            return base_prompt + "\n\nIMPORTANT: Provide specific span-level disagreements, not general feedback."
        elif agent_name == "synthesis_agent":
            return base_prompt + "\n\nIMPORTANT: Actively resolve all contradictions before final answer."

        return base_prompt

    def _generate_diff(self, original: str, proposed: str) -> str:
        """Generate a simple diff between prompts"""
        original_lines = original.split("\n")
        proposed_lines = proposed.split("\n")

        diff_lines = []

        for i, line in enumerate(proposed_lines):
            if i < len(original_lines):
                if line != original_lines[i]:
                    diff_lines.append(f"+ {line}")
                else:
                    diff_lines.append(f"  {line}")
            else:
                diff_lines.append(f"+ {line}")

        return "\n".join(diff_lines[:20])  # Limit diff size


meta_agent = MetaAgent()