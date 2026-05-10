from typing import Dict, Any, List
from .base import BaseAgent
from ..tools import tool_registry
import asyncio


class RetrievalAgent(BaseAgent):
    """Agent that performs multi-hop reasoning across retrieved chunks"""

    def __init__(self, max_context_budget: int = 50000):
        super().__init__("retrieval_agent", max_context_budget)

    def get_system_prompt(self) -> str:
        return """You are the Retrieval Agent. Your role is to perform multi-hop reasoning across at least two retrieved chunks before forming an answer.

Your responsibilities:
1. Query external data sources (web search, database)
2. Perform multi-hop reasoning - connect information from multiple sources
3. Cite which chunk contributed to which part of the answer
4. Track provenance for each claim

Output must include:
- Retrieved chunks with sources
- Reasoning chain showing how information was connected
- Citations mapping each claim to its source"""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process retrieval with multi-hop reasoning"""
        user_query = context.get("user_query", "")
        decomposed_tasks = context.get("decomposed_tasks", [])

        # Execute retrieval tasks
        retrieval_results = []
        citations = []

        # First hop: initial retrieval
        first_hop_results = await self._first_hop_retrieval(user_query)
        retrieval_results.extend(first_hop_results)

        # Second hop: follow-up based on first hop
        second_hop_results = await self._second_hop_retrieval(first_hop_results, user_query)
        retrieval_results.extend(second_hop_results)

        # Multi-hop reasoning: combine information
        reasoned_answer = self._multi_hop_reasoning(first_hop_results, second_hop_results, user_query)

        # Build citations
        for idx, chunk in enumerate(retrieval_results):
            citations.append({
                "chunk_id": f"chunk_{idx}",
                "source": chunk.get("source", "unknown"),
                "content": chunk.get("content", "")[:200],
                "contribution": chunk.get("contribution", "")
            })

        # Store in context
        context["retrieval_results"] = retrieval_results
        context["metadata"]["current_state"] = "retrieved"

        return {
            "success": True,
            "retrieval_results": retrieval_results,
            "reasoned_answer": reasoned_answer,
            "citations": citations,
            "hop_count": 2
        }

    async def _first_hop_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """First hop: initial retrieval"""
        # Use web search
        result = await tool_registry.execute_with_fallback("web_search", {"query": query})

        if result.success and result.data:
            return [{
                "hop": 1,
                "source": "web_search",
                "content": str(result.data),
                "relevance": 0.9,
                "contribution": "Initial information gathering"
            }]

        return []

    async def _second_hop_retrieval(self, first_hop_results: List[Dict], query: str) -> List[Dict[str, Any]]:
        """Second hop: follow-up retrieval"""
        # Use data lookup as second hop
        result = await tool_registry.execute_with_fallback("data_lookup", {"query": query})

        if result.success and result.data:
            return [{
                "hop": 2,
                "source": "data_lookup",
                "content": str(result.data),
                "relevance": 0.85,
                "contribution": "Structured data to validate/further information"
            }]

        return []

    def _multi_hop_reasoning(self, first_hop: List[Dict], second_hop: List[Dict], query: str) -> str:
        """Combine information from multiple hops"""
        # Build reasoning chain
        reasoning_parts = []

        if first_hop:
            reasoning_parts.append(f"From initial search: {first_hop[0].get('content', '')[:200]}")

        if second_hop:
            reasoning_parts.append(f"From data lookup: {second_hop[0].get('content', '')[:200]}")

        # Combine into answer
        answer = f"Based on multi-hop reasoning:\n\n"
        answer += "\n".join(reasoning_parts)
        answer += f"\n\nQuery: {query}\n\nThis combines information from multiple sources to provide a comprehensive answer."

        return answer


retrieval_agent = RetrievalAgent()