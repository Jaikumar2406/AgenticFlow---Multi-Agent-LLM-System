from typing import Dict, Any, Optional
from .base import BaseTool, ToolResult
from .web_search import web_search_tool
from .code_execution import code_execution_tool
from .data_lookup import data_lookup_tool
from .self_reflection import self_reflection_tool
import time


class ToolRegistry:
    """Registry for all available tools with fallback logic"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {
            "web_search": web_search_tool,
            "code_execution": code_execution_tool,
            "data_lookup": data_lookup_tool,
            "self_reflection": self_reflection_tool
        }
        self._call_history: Dict[str, list] = {}

    def register_tool(self, tool: BaseTool):
        """Register a new tool"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> list:
        """List all available tools"""
        return [
            {
                "name": name,
                "failure_contract": tool.get_failure_contract()
            }
            for name, tool in self._tools.items()
        ]

    async def execute_with_fallback(self, tool_name: str, input_data: Dict[str, Any],
                                    max_retries: int = 2) -> ToolResult:
        """Execute tool with fallback logic for different failure modes"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool {tool_name} not found",
                latency_ms=0
            )

        # Track call history
        if tool_name not in self._call_history:
            self._call_history[tool_name] = []

        for retry in range(max_retries + 1):
            start_time = time.time()

            try:
                result = await tool.execute(input_data)

                # Check result for failure modes
                if not result.get("success", False):
                    # Tool returned failure - decide on fallback
                    error = result.get("error", "Unknown error")

                    if "timeout" in error.lower():
                        # Timeout fallback: try with simplified input
                        if retry < max_retries:
                            simplified = self._simplify_input(input_data, tool_name)
                            input_data = simplified
                            continue
                        return ToolResult(
                            success=False,
                            error=f"Timeout after {max_retries} retries",
                            latency_ms=(time.time() - start_time) * 1000
                        )

                    if "invalid" in error.lower():
                        # Invalid input - no retry, return error
                        return ToolResult(
                            success=False,
                            error=error,
                            latency_ms=(time.time() - start_time) * 1000
                        )

                # Check for empty results
                data = result.get("data")
                if data is None or (isinstance(data, list) and len(data) == 0):
                    if retry < max_retries:
                        # Try with modified query
                        modified = self._modify_query(input_data, tool_name)
                        input_data = modified
                        continue

                # Success
                latency = (time.time() - start_time) * 1000

                # Track in history
                self._call_history[tool_name].append({
                    "input": input_data,
                    "output": result,
                    "latency_ms": latency,
                    "accepted": True,
                    "retry": retry
                })

                return ToolResult(
                    success=True,
                    data=result.get("data"),
                    latency_ms=latency
                )

            except Exception as e:
                if retry < max_retries:
                    continue
                return ToolResult(
                    success=False,
                    error=str(e),
                    latency_ms=(time.time() - start_time) * 1000
                )

        return ToolResult(success=False, error="Max retries exceeded", latency_ms=0)

    def _simplify_input(self, input_data: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Simplify input for retry"""
        if tool_name == "web_search":
            # Take only first few words
            query = input_data.get("query", "")
            words = query.split()[:5]
            return {"query": " ".join(words)}
        elif tool_name == "code_execution":
            # Return minimal code
            return {"code": "print('retry')"}
        else:
            return input_data

    def _modify_query(self, input_data: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Modify query for retry"""
        if tool_name == "web_search":
            query = input_data.get("query", "")
            return {"query": f"{query} simplified"}
        elif tool_name == "data_lookup":
            query = input_data.get("query", "")
            return {"query": f"list {query}"}
        return input_data

    def get_call_history(self, tool_name: str = None) -> Dict[str, list]:
        """Get call history"""
        if tool_name:
            return {tool_name: self._call_history.get(tool_name, [])}
        return self._call_history


# Global registry instance
tool_registry = ToolRegistry()