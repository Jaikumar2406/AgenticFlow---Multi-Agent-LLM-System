from typing import Dict, Any, List
from .base import BaseTool
import hashlib


class SelfReflectionTool(BaseTool):
    """Tool for agents to re-read their previous outputs and identify contradictions"""

    def __init__(self):
        super().__init__("self_reflection", timeout=3)

    def get_failure_contract(self) -> Dict[str, str]:
        return {
            "timeout": "{'success': false, 'error': 'Reflection timeout after 3s', 'contradictions': []}",
            "empty_results": "{'success': true, 'contradictions': [], 'insights': [], 'error': null}",
            "malformed_input": "{'success': false, 'error': 'Invalid input: must provide outputs to reflect on', 'contradictions': []}"
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute self-reflection"""
        # Validate input
        if not isinstance(input_data, dict):
            return {
                "success": False,
                "error": "Invalid input: must be a dictionary",
                "contradictions": [],
                "insights": []
            }

        outputs = input_data.get("outputs", [])
        if not outputs or not isinstance(outputs, list):
            return {
                "success": False,
                "error": "Invalid input: must provide outputs list to reflect on",
                "contradictions": [],
                "insights": []
            }

        # Analyze outputs for contradictions
        contradictions = self._find_contradictions(outputs)
        insights = self._generate_insights(outputs, contradictions)

        # Generate analysis hash
        analysis_hash = hashlib.sha256(
            f"{str(outputs)}{str(contradictions)}".encode()
        ).hexdigest()[:16]

        return {
            "success": True,
            "data": {
                "contradictions": contradictions,
                "insights": insights,
                "outputs_analyzed": len(outputs)
            },
            "analysis_hash": analysis_hash,
            "error": None
        }

    def _find_contradictions(self, outputs: List[Dict]) -> List[Dict[str, Any]]:
        """Find contradictions in outputs"""
        contradictions = []

        # Simple contradiction detection
        claims = []
        for idx, output in enumerate(outputs):
            text = output.get("text", "")
            claims.append((idx, text))

        # Check for opposite statements
        opposite_pairs = [
            ("true", "false"),
            ("yes", "no"),
            ("correct", "incorrect"),
            ("increase", "decrease"),
            ("more", "less"),
            ("positive", "negative"),
        ]

        for i, (idx1, text1) in enumerate(claims):
            for idx2, text2 in claims[i+1:]:
                text1_lower = text1.lower()
                text2_lower = text2.lower()

                for word1, word2 in opposite_pairs:
                    if word1 in text1_lower and word2 in text2_lower:
                        contradictions.append({
                            "type": "opposite_statements",
                            "source_output_1": idx1,
                            "source_output_2": idx2,
                            "statement_1": text1[:100],
                            "statement_2": text2[:100],
                            "conflicting_terms": (word1, word2)
                        })
                    elif word2 in text1_lower and word1 in text2_lower:
                        contradictions.append({
                            "type": "opposite_statements",
                            "source_output_1": idx1,
                            "source_output_2": idx2,
                            "statement_1": text1[:100],
                            "statement_2": text2[:100],
                            "conflicting_terms": (word2, word1)
                        })

        return contradictions

    def _generate_insights(self, outputs: List[Dict], contradictions: List[Dict]) -> List[str]:
        """Generate insights from analysis"""
        insights = []

        if len(outputs) > 3:
            insights.append(f"Analyzed {len(outputs)} outputs for consistency")

        if contradictions:
            insights.append(f"Found {len(contradictions)} potential contradictions requiring resolution")

        if not contradictions and outputs:
            insights.append("All outputs appear consistent with each other")

        return insights


self_reflection_tool = SelfReflectionTool()