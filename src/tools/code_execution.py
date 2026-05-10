import asyncio
from typing import Dict, Any
from .base import BaseTool, ToolResult
import io
import sys
import traceback
import hashlib


class CodeExecutionTool(BaseTool):
    """Code execution sandbox for Python snippets"""

    def __init__(self):
        super().__init__("code_execution", timeout=30)
        self._execution_count = 0

    def get_failure_contract(self) -> Dict[str, str]:
        return {
            "timeout": "{'success': false, 'error': 'Execution timeout after 30s', 'stdout': '', 'stderr': '', 'exit_code': -1}",
            "empty_results": "{'success': true, 'stdout': '', 'stderr': '', 'exit_code': 0}",
            "malformed_input": "{'success': false, 'error': 'Invalid input: code must be a non-empty string', 'stdout': '', 'stderr': 'Input validation failed', 'exit_code': 1}"
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code"""
        # Validate input
        if not isinstance(input_data, dict):
            return {
                "success": False,
                "error": "Invalid input: must be a dictionary",
                "stdout": "",
                "stderr": "Input validation failed",
                "exit_code": 1
            }

        code = input_data.get("code", "")
        if not code or not isinstance(code, str):
            return {
                "success": False,
                "error": "Invalid input: code must be a non-empty string",
                "stdout": "",
                "stderr": "Input validation failed",
                "exit_code": 1
            }

        self._execution_count += 1

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute in async context
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_code, code),
                timeout=self.timeout
            )

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            exit_code = 0

        except asyncio.TimeoutError:
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue() + "\nExecution timed out after 30s"
            exit_code = -1
            return {
                "success": False,
                "error": "Execution timeout after 30s",
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }
        except Exception as e:
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue() + "\n" + traceback.format_exc()
            exit_code = 1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Generate execution hash
        exec_hash = hashlib.sha256(f"{code}{stdout}{stderr}".encode()).hexdigest()[:16]

        return {
            "success": exit_code == 0,
            "data": {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            },
            "execution_hash": exec_hash,
            "error": None if exit_code == 0 else stderr
        }

    def _execute_code(self, code: str):
        """Execute code in sandbox (simplified)"""
        exec(code, {"__builtins__": __builtins__})


code_execution_tool = CodeExecutionTool()