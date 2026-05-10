import pytest
import asyncio
from uuid import uuid4

from src.agents import (
    orchestrator_agent,
    decomposition_agent,
    retrieval_agent,
    critique_agent,
    synthesis_agent
)
from src.tools import tool_registry
from src.utils.pipeline import PipelineRunner
from src.eval.pipeline import evaluation_pipeline


class TestAgents:
    """Test individual agents"""

    @pytest.mark.asyncio
    async def test_orchestrator(self):
        """Test orchestrator routing"""
        context = {
            "user_query": "What is Python?",
            "metadata": {"current_state": "initial"}
        }
        result = await orchestrator_agent.process(context)

        assert result["success"] is True
        assert "routing_decision" in result
        assert "next_agent" in result

    @pytest.mark.asyncio
    async def test_decomposition(self):
        """Test task decomposition"""
        context = {
            "user_query": "Find and analyze data about machine learning",
            "metadata": {}
        }
        result = await decomposition_agent.process(context)

        assert result["success"] is True
        assert "tasks" in result
        assert len(result["tasks"]) > 0

    @pytest.mark.asyncio
    async def test_retrieval(self):
        """Test retrieval with multi-hop"""
        context = {
            "user_query": "What is machine learning?",
            "decomposed_tasks": [],
            "metadata": {}
        }
        result = await retrieval_agent.process(context)

        assert result["success"] is True
        assert "retrieval_results" in result
        assert "citations" in result

    @pytest.mark.asyncio
    async def test_critique(self):
        """Test critique agent"""
        context = {
            "user_query": "Test query",
            "retrieval_results": [
                {"content": "This is a test result", "source": "test"}
            ],
            "decomposed_tasks": [],
            "metadata": {}
        }
        result = await critique_agent.process(context)

        assert result["success"] is True
        assert "confidence_scores" in result

    @pytest.mark.asyncio
    async def test_synthesis(self):
        """Test synthesis agent"""
        context = {
            "user_query": "Test query",
            "retrieval_results": [
                {"content": "Result 1", "source": "web"}
            ],
            "critique_results": [],
            "decomposed_tasks": [],
            "metadata": {}
        }
        result = await synthesis_agent.process(context)

        assert result["success"] is True
        assert "final_answer" in result
        assert "provenance" in result


class TestTools:
    """Test tools"""

    @pytest.mark.asyncio
    async def test_web_search(self):
        """Test web search tool"""
        result = await tool_registry.execute_with_fallback(
            "web_search",
            {"query": "python"}
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_code_execution(self):
        """Test code execution"""
        result = await tool_registry.execute_with_fallback(
            "code_execution",
            {"code": "print('hello')"}
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_data_lookup(self):
        """Test data lookup"""
        result = await tool_registry.execute_with_fallback(
            "data_lookup",
            {"query": "list employees"}
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_self_reflection(self):
        """Test self reflection"""
        result = await tool_registry.execute_with_fallback(
            "self_reflection",
            {"outputs": [{"text": "test"}, {"text": "test"}]}
        )
        assert result.success is True


class TestPipeline:
    """Test pipeline"""

    @pytest.mark.asyncio
    async def test_pipeline_runner(self):
        """Test full pipeline execution"""
        runner = PipelineRunner()
        job_id = uuid4()

        result = await runner.run(job_id, "What is Python?")

        assert "final_answer" in result


class TestEvaluation:
    """Test evaluation"""

    def test_test_cases_loaded(self):
        """Test that test cases are loaded"""
        pipeline = evaluation_pipeline
        assert len(pipeline.test_cases) == 15

    def test_categories(self):
        """Test test case categories"""
        pipeline = evaluation_pipeline
        categories = {tc.category for tc in pipeline.test_cases}
        assert "baseline" in categories
        assert "ambiguous" in categories
        assert "adversarial" in categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])