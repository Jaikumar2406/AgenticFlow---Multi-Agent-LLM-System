from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import time


class BaseTool(ABC):
    """Base class for all tools with failure contract"""

    def __init__(self, name: str, timeout: int = 30):
        self.name = name
        self.timeout = timeout

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given input"""
        pass

    def get_failure_contract(self) -> Dict[str, str]:
        """Return the failure contract for this tool"""
        return {
            "timeout": f"Returns {{'success': false, 'error': 'Timeout after {self.timeout}s', 'data': null}}",
            "empty_results": "Returns {'success': true, 'data': [], 'error': null}",
            "malformed_input": "Returns {'success': false, 'error': 'Invalid input: <specific error>', 'data': null}"
        }


class ToolResult:
    def __init__(self, success: bool, data: Any = None, error: Optional[str] = None,
                 latency_ms: float = 0.0):
        self.success = success
        self.data = data
        self.error = error
        self.latency_ms = latency_ms
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat()
        }


async def timed_execute(tool, input_data: Dict[str, Any]) -> ToolResult:
    """Execute tool with timing"""
    start = time.time()
    try:
        result = await tool.execute(input_data)
        latency = (time.time() - start) * 1000
        return ToolResult(success=True, data=result, latency_ms=latency)
    except Exception as e:
        latency = (time.time() - start) * 1000
        return ToolResult(success=False, error=str(e), latency_ms=latency)