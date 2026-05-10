from typing import Dict, Any, List
from .base import BaseAgent
import uuid


class DecompositionAgent(BaseAgent):
    """Agent that breaks ambiguous queries into typed sub-tasks with dependency graphs"""

    def __init__(self, max_context_budget: int = 30000):
        super().__init__("decomposition_agent", max_context_budget)

    def get_system_prompt(self) -> str:
        return """You are the Decomposition Agent. Your role is to break ambiguous or complex queries into typed sub-tasks with explicit dependency graphs.

Your responsibilities:
1. Identify distinct sub-tasks within the user's query
2. Assign task types: retrieval, code_execution, data_lookup, reflection
3. Define explicit dependency relationships between tasks
4. Ensure dependent tasks cannot execute until dependencies resolve

Output format:
{
  "tasks": [
    {
      "task_id": "unique_id",
      "task_type": "retrieval|code_execution|data_lookup|reflection",
      "description": "clear task description",
      "dependencies": ["task_id_1", "task_id_2"]  // empty if no dependencies
    }
  ]
}"""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose the query into tasks"""
        user_query = context.get("user_query", "")

        # Analyze query and decompose
        tasks = self._decompose_query(user_query)

        # Store in context
        context["decomposed_tasks"] = [
            task.dict() for task in tasks
        ]
        context["metadata"]["current_state"] = "decomposed"

        return {
            "success": True,
            "tasks": [task.dict() for task in tasks],
            "task_count": len(tasks),
            "ready_tasks": [t.task_id for t in tasks if not t.dependencies]
        }

    def _decompose_query(self, query: str) -> List[Dict]:
        """Decompose query into tasks"""
        query_lower = query.lower()
        tasks = []

        # Common decomposition patterns
        if "search" in query_lower or "find" in query_lower:
            tasks.append({
                "task_id": f"task_{uuid.uuid4().hex[:8]}",
                "task_type": "retrieval",
                "description": "Search for relevant information",
                "dependencies": []
            })

        if "calculate" in query_lower or "compute" in query_lower or "run" in query_lower:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            tasks.append({
                "task_id": task_id,
                "task_type": "code_execution",
                "description": "Execute code for computation",
                "dependencies": []
            })

        if "database" in query_lower or "data" in query_lower or "list" in query_lower:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            tasks.append({
                "task_id": task_id,
                "task_type": "data_lookup",
                "description": "Query structured data",
                "dependencies": []
            })

        if "compare" in query_lower or "validate" in query_lower:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            # This might depend on retrieval
            deps = [t["task_id"] for t in tasks if t["task_type"] == "retrieval"]
            tasks.append({
                "task_id": task_id,
                "task_type": "reflection",
                "description": "Reflect on and validate results",
                "dependencies": deps
            })

        # If no tasks matched, create a default retrieval task
        if not tasks:
            tasks.append({
                "task_id": f"task_{uuid.uuid4().hex[:8]}",
                "task_type": "retrieval",
                "description": "Retrieve general information",
                "dependencies": []
            })

        return tasks


decomposition_agent = DecompositionAgent()