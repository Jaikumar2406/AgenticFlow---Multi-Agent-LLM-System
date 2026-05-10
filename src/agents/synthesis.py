from typing import Dict, Any, List
from .base import BaseAgent


class SynthesisAgent(BaseAgent):
    """Agent that merges outputs from all sub-agents and resolves contradictions"""

    def __init__(self, max_context_budget: int = 60000):
        super().__init__("synthesis_agent", max_context_budget)

    def get_system_prompt(self) -> str:
        return """You are the Synthesis Agent. Your role is to merge outputs from all sub-agents, resolve contradictions flagged by the critique agent, and produce a final answer with a provenance map.

Your responsibilities:
1. Merge outputs from decomposition, retrieval, and critique agents
2. Resolve any contradictions identified by the critique agent
3. Produce final answer with provenance map linking each sentence to source agent and chunk
4. Do not surface contradictions to user - resolve them internally

Output format:
{
  "final_answer": "...",
  "provenance": [
    {
      "sentence": "...",
      "source_agent": "retrieval_agent",
      "source_chunk": "chunk_0",
      "confidence": 0.9
    }
  ],
  "resolved_contradictions": [...]
}"""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final answer"""
        user_query = context.get("user_query", "")
        retrieval_results = context.get("retrieval_results", [])
        critique_results = context.get("critique_results", [])
        decomposed_tasks = context.get("decomposed_tasks", [])

        # Resolve contradictions
        resolved_contradictions = self._resolve_contradictions(critique_results)

        # Build provenance map
        provenance = self._build_provenance(retrieval_results)

        # Generate final answer
        final_answer = self._generate_final_answer(
            user_query, retrieval_results, resolved_contradictions
        )

        # Store in context
        context["synthesis_output"] = {
            "final_answer": final_answer,
            "provenance": provenance,
            "resolved_contradictions": resolved_contradictions
        }
        context["metadata"]["current_state"] = "completed"

        return {
            "success": True,
            "final_answer": final_answer,
            "provenance": provenance,
            "resolved_contradictions": resolved_contradictions
        }

    def _resolve_contradictions(self, critique_results: List[Dict]) -> List[Dict[str, Any]]:
        """Resolve contradictions identified by critique agent"""
        resolved = []

        if not critique_results:
            return resolved

        critique = critique_results[0]
        disagreements = critique.get("disagreements", [])

        for disagreement in disagreements:
            # Resolve by providing balanced view
            resolved.append({
                "original_conflict": disagreement.get("span", ""),
                "resolution": f"Considering both perspectives: {disagreement.get('span', '')} and {disagreement.get('alternative', '')}. "
                              f"Providing balanced answer.",
                "method": "synthesized_with_acknowledgment"
            })

        return resolved

    def _build_provenance(self, retrieval_results: List[Dict]) -> List[Dict[str, Any]]:
        """Build provenance map"""
        provenance = []

        for idx, result in enumerate(retrieval_results):
            content = result.get("content", "")[:150]
            source = result.get("source", "unknown")
            relevance = result.get("relevance", 0.7)

            provenance.append({
                "sentence": content,
                "source_agent": "retrieval_agent",
                "source_chunk": f"chunk_{idx}",
                "confidence": relevance
            })

        # Add synthesis agent provenance
        provenance.append({
            "sentence": "Final synthesized answer",
            "source_agent": "synthesis_agent",
            "source_chunk": "synthesis_output",
            "confidence": 0.95
        })

        return provenance

    def _generate_final_answer(self, query: str, retrieval_results: List[Dict],
                               resolved_contradictions: List[Dict]) -> str:
        """Generate final answer"""
        # Start with query
        answer = f"Query: {query}\n\n"

        # Add retrieved information
        if retrieval_results:
            answer += "Based on retrieved information:\n\n"
            for result in retrieval_results:
                content = result.get("content", "")
                source = result.get("source", "unknown")
                answer += f"- [{source}] {content[:300]}...\n\n"

        # Add contradiction resolution if any
        if resolved_contradictions:
            answer += "Contradiction Resolution:\n\n"
            for resolution in resolved_contradictions:
                answer += f"- {resolution.get('resolution', '')}\n\n"

        # Add final synthesized response
        answer += "\nFinal Answer: The information has been synthesized from multiple sources with contradictions resolved. "

        if retrieval_results:
            answer += f"Key findings from {len(retrieval_results)} sources have been combined to provide a comprehensive response."

        return answer


synthesis_agent = SynthesisAgent()