from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from uuid import UUID
import time
import hashlib
import json


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str, max_context_budget: int = 50000):
        self.name = name
        self.max_context_budget = max_context_budget
        self.current_context_usage = 0

    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the context and return result"""
        pass

    def check_budget(self, additional_tokens: int) -> bool:
        """Check if agent has remaining budget"""
        return (self.current_context_usage + additional_tokens) <= self.max_context_budget

    def consume_tokens(self, tokens: int):
        """Consume tokens from budget"""
        self.current_context_usage += tokens

    def get_remaining_budget(self) -> int:
        """Get remaining budget"""
        return self.max_context_budget - self.current_context_usage

    def format_log_event(self, event_type: str, input_data: Any = None,
                        output_data: Any = None, metadata: Dict = None) -> Dict[str, Any]:
        """Format a structured log event"""
        input_hash = None
        output_hash = None

        if input_data:
            input_hash = hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest()[:16]

        if output_data:
            output_hash = hashlib.sha256(
                json.dumps(output_data, sort_keys=True).encode()
            ).hexdigest()[:16]

        return {
            "agent_id": self.name,
            "event_type": event_type,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent - to be overridden"""
        return f"You are {self.name}, a specialized AI agent."


class AgentResult:
    """Result from agent processing"""

    def __init__(self, success: bool, output: Any = None, error: str = None,
                 metadata: Dict = None):
        self.success = success
        self.output = output
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }