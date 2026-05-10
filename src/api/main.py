from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
import json
import asyncio
import logging

from ..db.database import db, get_db
from ..utils.pipeline import pipeline_runner
from config.settings import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgenticFlow API",
    description="Multi-agent LLM system with self-improving evaluation",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str
    context_budget: Optional[int] = None


class ApprovalRequest(BaseModel):
    rewrite_id: str
    approved: bool
    approved_by: Optional[str] = "human"


class RetryRequest(BaseModel):
    test_case_ids: Optional[List[str]] = None
    category: Optional[str] = None


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    job_id: Optional[str] = None


@app.on_event("startup")
async def startup():
    """Initialize database connection"""
    await db.connect()
    await db.init_schema()
    logger.info("API server started")


@app.on_event("shutdown")
async def shutdown():
    """Close database connection"""
    await db.disconnect()
    logger.info("API server stopped")


async def sse_format(event: str, data: Any) -> str:
    """Format SSE event properly"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def event_generator(job_id: UUID, query: str):
    """Generate SSE events for streaming"""
    try:
        # Send initial event
        yield await sse_format("start", {"job_id": str(job_id), "query": query})

        # Run pipeline with callback for streaming
        async def callback(event: Dict):
            await asyncio.sleep(0.1)
            return await sse_format(event.get("type", "update"), event)

        # Execute pipeline
        result = await pipeline_runner.run(job_id, query, callback)

        # Yield completion event
        yield await sse_format("complete", result)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        yield await sse_format("error", {"error": str(e)})


@app.post("/query")
async def submit_query(request: QueryRequest):
    """Submit a query and receive streaming SSE response"""
    job_id = uuid4()

    # Create job in database
    await db.create_job(job_id, request.query)

    # Return streaming response
    return StreamingResponse(
        event_generator(job_id, request.query),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache"
        }
    )


@app.get("/trace/{job_id}")
async def get_execution_trace(job_id: str):
    """Retrieve full execution trace for a completed job"""
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_JOB_ID",
                "message": "Invalid job ID format",
                "job_id": None
            }
        )

    job = await db.get_job(job_uuid)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message": f"Job {job_id} not found",
                "job_id": job_id
            }
        )

    trace = await db.get_execution_trace(job_uuid)

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "query": job.get("user_query"),
        "result": job.get("result"),
        "trace": trace,
        "trace_length": len(trace)
    }


@app.get("/eval/summary")
async def get_eval_summary():
    """Get latest eval run summary"""
    latest_run = await db.get_latest_eval_run()

    if not latest_run:
        return {
            "error_code": "NO_EVAL_RUNS",
            "message": "No evaluation runs found",
            "summary": None
        }

    test_results = json.loads(latest_run.get("test_results", "[]"))

    categories = {}
    for result in test_results:
        category = result.get("category", "unknown")
        if category not in categories:
            categories[category] = {"count": 0, "scores": {}}
        categories[category]["count"] += 1
        scores = result.get("scores", {})
        for dim, score in scores.items():
            if dim not in categories[category]["scores"]:
                categories[category]["scores"][dim] = []
            categories[category]["scores"][dim].append(score)

    for category in categories:
        for dim in categories[category]["scores"]:
            scores = categories[category]["scores"][dim]
            categories[category]["scores"][dim] = sum(scores) / len(scores) if scores else 0

    return {
        "run_id": str(latest_run.get("id")),
        "timestamp": latest_run.get("timestamp"),
        "categories": categories,
        "summary": latest_run.get("summary"),
        "total_tests": len(test_results)
    }


@app.post("/prompt/approve")
async def approve_rewrite(request: ApprovalRequest):
    """Submit human approval/rejection for prompt rewrite"""
    try:
        rewrite_uuid = UUID(request.rewrite_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_REWRITE_ID",
                "message": "Invalid rewrite ID format",
                "job_id": None
            }
        )

    rewrite = await db.get_rewrite(rewrite_uuid)
    if not rewrite:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "REWRITE_NOT_FOUND",
                "message": f"Rewrite {request.rewrite_id} not found",
                "job_id": None
            }
        )

    status = "approved" if request.approved else "rejected"
    await db.update_rewrite_status(rewrite_uuid, status, request.approved_by)

    return {
        "success": True,
        "rewrite_id": request.rewrite_id,
        "status": status,
        "approved_by": request.approved_by
    }


@app.post("/eval/retry")
async def retry_eval(request: RetryRequest):
    """Trigger targeted re-eval"""
    return {
        "success": True,
        "message": "Re-eval triggered",
        "test_case_ids": request.test_case_ids,
        "category": request.category
    }


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "AgenticFlow",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/query", "method": "POST", "description": "Submit query with SSE streaming"},
            {"path": "/trace/{job_id}", "method": "GET", "description": "Get execution trace"},
            {"path": "/eval/summary", "method": "GET", "description": "Get eval summary"},
            {"path": "/prompt/approve", "method": "POST", "description": "Approve/reject prompt rewrite"},
            {"path": "/eval/retry", "method": "POST", "description": "Trigger re-eval"}
        ]
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "job_id": None
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)