from config.settings import settings
from ..utils.pipeline import pipeline_runner
from ..db.database import db
from ..eval.pipeline import evaluation_pipeline
from ..eval.meta import meta_agent
from uuid import UUID


# Check if Redis is available
_celery_app = None
if settings.REDIS_URL:
    try:
        from celery import Celery
        _celery_app = Celery(
            "agenticflow",
            broker=settings.REDIS_URL,
            backend=settings.REDIS_URL
        )
        _celery_app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            enable_utc=True
        )
    except Exception as e:
        print(f"Celery not available: {e}")
        _celery_app = None


def process_query(job_id: str, query: str):
    """Process a query through the pipeline (synchronous wrapper)"""
    import asyncio

    async def run():
        job_uuid = UUID(job_id)
        await db.update_job(job_uuid, "processing")
        result = await pipeline_runner.run(job_uuid, query)
        await db.update_job(job_uuid, "completed", result)
        return result

    return asyncio.run(run())


def run_evaluation():
    """Run evaluation pipeline"""
    import asyncio

    async def run():
        results = await evaluation_pipeline.run_evaluation(pipeline_runner)
        run_id = UUID(int=0)
        await db.save_eval_run(
            run_id,
            results["test_results"],
            results["summary"],
            meta_agent.current_prompts
        )
        analysis = meta_agent.analyze_failures(results["test_results"])
        if analysis.get("worst_agent"):
            rewrite = meta_agent.propose_rewrite(analysis["worst_agent"], analysis)
            await db.save_prompt_rewrite(rewrite)
        return results

    return asyncio.run(run())


def re_eval_failed_cases(test_case_ids: list, approved_prompts: dict):
    """Re-evaluate previously failed cases"""
    import asyncio

    async def run():
        meta_agent.current_prompts.update(approved_prompts)
        results = await evaluation_pipeline.run_evaluation(pipeline_runner)
        return {"results": results, "delta": calculate_delta(results)}

    return asyncio.run(run())


def calculate_delta(new_results: dict) -> dict:
    return {"improvement": "To be calculated based on comparison with previous run"}


# Celery task decorators (only if Redis is available)
if _celery_app:
    process_query_task = _celery_app.task(name="process_query")(process_query)
    run_evaluation_task = _celery_app.task(name="run_evaluation")(run_evaluation)
    re_eval_task = _celery_app.task(name="re_eval_failed_cases")(re_eval_failed_cases)
else:
    # Fallback: direct function calls
    process_query_task = process_query
    run_evaluation_task = run_evaluation
    re_eval_task = re_eval_failed_cases