from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class AgentContext(BaseModel):
    """Shared context object for inter-agent communication"""
    version: str = "1.0"
    job_id: UUID
    user_query: str
    current_agent: Optional[str] = None
    decomposed_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_results: List[Dict[str, Any]] = Field(default_factory=list)
    critique_results: List[Dict[str, Any]] = Field(default_factory=list)
    synthesis_output: Optional[Dict[str, Any]] = None
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecomposedTask(BaseModel):
    """A decomposed sub-task with dependency graph"""
    task_id: str
    task_type: str  # "retrieval", "code_execution", "data_lookup", "reflection"
    description: str
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskDependencyGraph(BaseModel):
    """Dependency graph for tasks"""
    tasks: List[DecomposedTask]

    def get_ready_tasks(self) -> List[DecomposedTask]:
        """Get tasks that can be executed (dependencies resolved)"""
        ready = []
        for task in self.tasks:
            if task.status != "pending":
                continue
            deps = set(task.dependencies)
            completed = {t.task_id for t in self.tasks if t.status == "completed"}
            if deps.issubset(completed):
                ready.append(task)
        return ready


class RoutingDecision(BaseModel):
    """Orchestrator's routing decision"""
    reasoning: str
    agents_to_invoke: List[str]
    context_budget: Dict[str, int]
    tool_assignments: Dict[str, List[str]] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Tool call record"""
    tool_name: str
    input: Dict[str, Any]
    output: Optional[Any] = None
    latency_ms: float = 0.0
    accepted: bool = True
    retry_count: int = 0
    error: Optional[str] = None


class ToolResult(BaseModel):
    """Tool result with metadata"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


class CritiqueResult(BaseModel):
    """Critique agent output"""
    confidence_scores: Dict[str, float] = Field(default_factory=dict)  # claim -> confidence
    disagreements: List[Dict[str, Any]] = Field(default_factory=list)  # span, reason, alternative
    overall_confidence: float = 0.0


class ProvenanceMap(BaseModel):
    """Provenance mapping for synthesis output"""
    sentence: str
    source_agent: str
    source_chunk: Optional[str] = None
    confidence: float = 1.0


class SynthesisOutput(BaseModel):
    """Synthesis agent output"""
    final_answer: str
    provenance: List[ProvenanceMap]
    resolved_contradictions: List[Dict[str, Any]] = Field(default_factory=list)


class ContextBudget(BaseModel):
    """Context budget tracking"""
    agent_name: str
    max_budget: int
    current_usage: int = 0
    remaining: int

    def check_available(self, needed: int) -> bool:
        return self.remaining >= needed

    def consume(self, tokens: int):
        self.current_usage += tokens
        self.remaining = self.max_budget - self.current_usage


class ExecutionEvent(BaseModel):
    """Structured logging event"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    event_type: str  # start, end, tool_call, tool_result, routing_decision, handoff, error
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    latency_ms: float = 0.0
    token_count: int = 0
    policy_violations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Evaluation result for a test case"""
    test_case_id: str
    category: str  # baseline, ambiguous, adversarial
    scores: Dict[str, float] = Field(default_factory=dict)
    justifications: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PromptRewrite(BaseModel):
    """Prompt rewrite proposal"""
    rewrite_id: UUID = Field(default_factory=uuid4)
    agent_name: str
    original_prompt: str
    proposed_prompt: str
    justification: str
    diff: str
    status: str = "pending"  # pending, approved, rejected
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None


class EvalRun(BaseModel):
    """Evaluation run record"""
    run_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    test_cases: List[EvalResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    prompts_used: Dict[str, str] = Field(default_factory=dict)