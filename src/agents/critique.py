from typing import Dict, Any, List
from .base import BaseAgent
from ..tools import tool_registry


class CritiqueAgent(BaseAgent):
    """Agent that reviews output, assigns confidence scores, and flags specific disagreements"""

    def __init__(self, max_context_budget: int = 40000):
        super().__init__("critique_agent", max_context_budget)

    def get_system_prompt(self) -> str:
        return """You are the Critique Agent. Your role is to review the output of every other agent, assign a structured confidence score per claim, and flag specific spans of text you disagree with.

Your responsibilities:
1. Review outputs from decomposition, retrieval, and synthesis agents
2. Assign confidence scores (0.0-1.0) to each claim
3. Flag specific text spans with disagreement and provide alternatives
4. Do not evaluate the output as a whole - evaluate specific claims

Output format:
{
  "confidence_scores": {
    "claim_1": 0.9,
    "claim_2": 0.5
  },
  "disagreements": [
    {
      "span": "text span being disputed",
      "reason": "why you disagree",
      "alternative": "suggested alternative"
    }
  ],
  "overall_confidence": 0.75
}"""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process critique of other agent outputs"""
        retrieval_results = context.get("retrieval_results", [])
        decomposed_tasks = context.get("decomposed_tasks", [])

        # Analyze retrieval results for claims
        claims = self._extract_claims(retrieval_results)

        # Assign confidence scores
        confidence_scores = self._assign_confidence_scores(claims)

        # Find disagreements using self-reflection tool
        disagreements = await self._flag_disagreements(retrieval_results, claims)

        # Calculate overall confidence
        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0

        # Store in context
        context["critique_results"] = [
            {
                "claims": claims,
                "confidence_scores": confidence_scores,
                "disagreements": disagreements,
                "overall_confidence": overall_confidence
            }
        ]
        context["metadata"]["current_state"] = "critiqued"

        return {
            "success": True,
            "confidence_scores": confidence_scores,
            "disagreements": disagreements,
            "overall_confidence": overall_confidence,
            "claims_analyzed": len(claims)
        }

    def _extract_claims(self, retrieval_results: List[Dict]) -> List[str]:
        """Extract claims from retrieval results"""
        claims = []

        for result in retrieval_results:
            content = result.get("content", "")
            # Simple claim extraction - split by sentences
            sentences = content.split(". ")
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    claims.append(f"claim_{i}: {sentence.strip()}")

        return claims[:10]  # Limit to 10 claims

    def _assign_confidence_scores(self, claims: List[str]) -> Dict[str, float]:
        """Assign confidence scores to claims"""
        scores = {}

        for claim in claims:
            # Simple heuristic scoring
            if "high" in claim.lower() or "certain" in claim.lower():
                scores[claim] = 0.9
            elif "maybe" in claim.lower() or "might" in claim.lower():
                scores[claim] = 0.5
            else:
                scores[claim] = 0.7

        return scores

    async def _flag_disagreements(self, retrieval_results: List[Dict], claims: List[str]) -> List[Dict[str, Any]]:
        """Flag specific disagreements using self-reflection"""
        # Use self-reflection tool to find contradictions
        outputs = [{"text": r.get("content", "")} for r in retrieval_results]

        if not outputs:
            return []

        result = await tool_registry.execute_with_fallback(
            "self_reflection",
            {"outputs": outputs}
        )

        if result.success and result.data:
            contradictions = result.data.get("contradictions", [])

            disagreements = []
            for contr in contradictions:
                disagreements.append({
                    "span": contr.get("statement_1", ""),
                    "reason": f"Contradiction: found conflicting terms {contr.get('conflicting_terms')}",
                    "alternative": contr.get("statement_2", "")
                })

            return disagreements

        return []


critique_agent = CritiqueAgent()