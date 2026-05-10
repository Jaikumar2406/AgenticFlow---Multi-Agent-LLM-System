from typing import Dict, Any, List, Optional, Callable
from uuid import UUID
import asyncio
import json
import time

from ..agents.graph import run_agent_pipeline
from ..tools import tool_registry
from ..db.database import db
from config.settings import settings


class PipelineRunner:
    """Runs the multi-agent pipeline using LangGraph"""

    async def run(self, job_id: UUID, query: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run the full pipeline using LangGraph"""
        start_time = time.time()

        try:
            # Send start event
            if callback:
                await callback({
                    "type": "start",
                    "job_id": str(job_id),
                    "query": query
                })

            # Run LangGraph pipeline
            result = await run_agent_pipeline(str(job_id), query)

            # Log events
            await self._log_event(job_id, {
                "agent_id": "graph",
                "event_type": "pipeline_complete",
                "latency_ms": (time.time() - start_time) * 1000,
                "metadata": {"success": result.get("success", False)}
            })

            if callback:
                await callback({
                    "type": "complete",
                    "result": result
                })

            return result

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "final_answer": f"Error: {str(e)}"
            }

            if callback:
                await callback({
                    "type": "error",
                    "error": str(e)
                })

            return error_result

    async def _log_event(self, job_id: UUID, event: Dict):
        """Log execution event"""
        try:
            await db.log_event(job_id, event)
        except Exception as e:
            print(f"Failed to log event: {e}")


pipeline_runner = PipelineRunner()