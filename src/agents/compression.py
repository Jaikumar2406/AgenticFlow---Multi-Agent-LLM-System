from typing import Dict, Any
from .base import BaseAgent


class CompressionAgent(BaseAgent):
    """Agent that compresses context while preserving structured data"""

    def __init__(self, max_context_budget: int = 10000):
        super().__init__("compression_agent", max_context_budget)

    def get_system_prompt(self) -> str:
        return """You are the Compression Agent. Your role is to compress older context when it exceeds budget while preserving structured data.

Your responsibilities:
1. Compress conversational filler while preserving meaning
2. Keep structured data (tool outputs, scores, citations) lossless
3. Summarize long context into concise form
4. Maintain provenance information"""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress context to fit budget"""
        current_usage = context.get("metadata", {}).get("context_usage", 0)
        max_budget = context.get("metadata", {}).get("max_budget", self.max_context_budget)

        if current_usage <= max_budget:
            return {
                "success": True,
                "compressed": False,
                "reason": "Context within budget"
            }

        # Compress context
        compressed_data = self._compress_context(context)

        # Store compressed version
        context["metadata"]["compressed_context"] = compressed_data
        context["metadata"]["compression_applied"] = True

        return {
            "success": True,
            "compressed": True,
            "original_size": current_usage,
            "compressed_size": len(str(compressed_data)),
            "preserved_data_types": ["tool_outputs", "scores", "citations"]
        }

    def _compress_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress context data"""
        compressed = {}

        # Preserve tool outputs (structured, lossless)
        if "tool_outputs" in context:
            compressed["tool_outputs"] = context["tool_outputs"]

        # Preserve scores (structured, lossless)
        if "retrieval_results" in context:
            compressed["retrieval_results"] = [
                {
                    "source": r.get("source"),
                    "relevance": r.get("relevance"),
                    "hop": r.get("hop")
                }
                for r in context["retrieval_results"]
            ]

        # Preserve citations (structured, lossless)
        if "citations" in context:
            compressed["citations"] = context["citations"]

        # Compress conversational elements
        if "decomposed_tasks" in context:
            compressed["decomposed_tasks_summary"] = f"{len(context['decomposed_tasks'])} tasks"

        if "user_query" in context:
            # Summarize query
            query = context["user_query"]
            compressed["query_summary"] = query[:100] + "..." if len(query) > 100 else query

        # Preserve metadata
        compressed["metadata"] = {
            "original_context_size": context.get("metadata", {}).get("context_usage", 0),
            "compression_type": "lossless_for_structured"
        }

        return compressed


compression_agent = CompressionAgent()