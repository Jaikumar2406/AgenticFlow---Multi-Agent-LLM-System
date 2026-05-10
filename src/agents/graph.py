from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from ..utils.llm_client import llm_client
from ..tools import tool_registry
import json


class AgentState(TypedDict):
    """State passed between nodes in the agent graph"""
    job_id: str
    user_query: str
    current_agent: str
    decomposed_tasks: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    critique_results: List[Dict[str, Any]]
    synthesis_output: Optional[Dict[str, Any]]
    tool_outputs: Dict[str, Any]
    metadata: Dict[str, Any]
    routing_plan: Dict[str, Any]


async def orchestrator_node(state: AgentState) -> AgentState:
    """Orchestrator - LLM-driven intent classification and routing"""
    query = state["user_query"]

    classification_prompt = f"""Analyze this user query and determine the appropriate routing.

User Query: "{query}"

Analyze:
1. What is the user's intent?
2. Does this require factual information/evidence?
3. Is this conversational/greeting?
4. Is this potentially adversarial?
5. Does this require multiple steps/decomposition?
6. Is this an analytical/comparison task?

Respond in JSON format - be precise about which agents are needed:
{{
  "intent_analysis": "brief analysis",
  "category": "greeting|casual|identity|acknowledgement|factual|analytical|decomposition|adversarial|coding",
  "needs_facts": true|false,
  "needs_decomposition": true|false,
  "needs_retrieval": true|false,
  "needs_critique": true|false,
  "risk_level": "low|medium|high",
  "agents_needed": ["agent1", "agent2"],
  "reasoning": "why these agents"
}}"""

    result = await llm_client.generate(
        classification_prompt,
        system_prompt="You are an orchestration classifier. Determine needed agents precisely. If needs_critique is true, include critique in agents_needed."
    )

    if result.get("success"):
        try:
            classification = json.loads(result["content"])
        except:
            classification = _fallback_classification(query)
    else:
        classification = _fallback_classification(query)

    # FORCE synchronization: if needs_critique is true, must include critique in agents_needed
    if classification.get("needs_critique") and "critique" not in classification.get("agents_needed", []):
        classification["agents_needed"].append("critique")

    # Build routing plan - MUST match classification
    routing_plan = {
        "order": classification.get("agents_needed", ["synthesis"]),
        "category": classification.get("category", "unknown"),
        "intent_analysis": classification.get("intent_analysis", ""),
        "reasoning": classification.get("reasoning", ""),
        "needs_facts": classification.get("needs_facts", False),
        "risk_level": classification.get("risk_level", "low"),
        "needs_critique": classification.get("needs_critique", False)
    }

    first_agent = routing_plan["order"][0] if routing_plan["order"] else "synthesis"

    state["current_agent"] = first_agent
    state["routing_plan"] = routing_plan
    state["metadata"]["current_state"] = f"{first_agent}_invoked"
    state["metadata"]["classification"] = classification

    return state


def _fallback_classification(query: str) -> Dict[str, Any]:
    """Fallback when LLM fails"""
    query_lower = query.lower().strip()

    if any(x in query_lower for x in ['hi', 'hello', 'hey', 'hii', 'who are you']):
        return {
            "intent_analysis": "User is greeting or asking about identity",
            "category": "greeting",
            "needs_facts": False,
            "needs_decomposition": False,
            "needs_retrieval": False,
            "needs_critique": False,
            "risk_level": "low",
            "agents_needed": ["synthesis"],
            "reasoning": "Greeting - no facts needed"
        }

    if any(x in query_lower for x in ['thanks', 'thank you', 'ok', 'okay', 'nice']):
        return {
            "intent_analysis": "User is acknowledging",
            "category": "acknowledgement",
            "needs_facts": False,
            "needs_decomposition": False,
            "needs_retrieval": False,
            "needs_critique": False,
            "risk_level": "low",
            "agents_needed": ["synthesis"],
            "reasoning": "Acknowledgement"
        }

    # Default for unknown - retrieval + critique + synthesis
    return {
        "intent_analysis": "Query needs factual information",
        "category": "factual",
        "needs_facts": True,
        "needs_decomposition": False,
        "needs_retrieval": True,
        "needs_critique": True,
        "risk_level": "medium",
        "agents_needed": ["retrieval", "critique", "synthesis"],
        "reasoning": "Factual query - retrieve, critique, then synthesize"
    }


async def decomposition_node(state: AgentState) -> AgentState:
    """Decomposition - only runs if needed"""
    query = state["user_query"]
    classification = state.get("metadata", {}).get("classification", {})

    if not classification.get("needs_decomposition"):
        state["decomposed_tasks"] = []
        return state

    decomp_prompt = f"""Break this query into specific tasks.

Query: {query}

Respond in JSON:
{{
  "tasks": [
    {{"task_id": "1", "task_type": "retrieval", "description": "what to find", "dependencies": []}}
  ]
}}"""

    result = await llm_client.generate(decomp_prompt, system_prompt="You decompose complex queries into steps.")

    if result.get("success"):
        try:
            data = json.loads(result["content"])
            state["decomposed_tasks"] = data.get("tasks", [])
        except:
            state["decomposed_tasks"] = []
    else:
        state["decomposed_tasks"] = []

    return state


async def retrieval_node(state: AgentState) -> AgentState:
    """Retrieval - with proper tool execution and normalization"""
    query = state["user_query"]
    classification = state.get("metadata", {}).get("classification", {})

    if not classification.get("needs_retrieval", False):
        state["retrieval_results"] = []
        state["tool_outputs"] = {"retrieval": {"status": "skipped", "reason": "not_needed"}}
        return state

    normalized_results = []
    tool_outputs = {}

    # Execute web search
    try:
        web_result = await tool_registry.execute_with_fallback("web_search", {"query": query})
        tool_outputs["web_search"] = {"success": web_result.success, "latency": web_result.latency_ms}

        if web_result.success and web_result.data:
            # Extract results from various possible formats
            data = web_result.data
            results = []

            # Try different data formats
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], dict):
                    results = data["data"].get("results", [])
                elif "results" in data:
                    results = data.get("results", [])

            for r in results:
                # Extract meaningful content - use full snippet as claim, clean evidence
                snippet = r.get("snippet", "") or r.get("content", "") or r.get("title", "")
                title = r.get("title", "")
                url = r.get("url", "web")
                relevance = float(r.get("relevance", r.get("score", 0.7)))

                if snippet:
                    # Clean evidence - just the factual content, no metadata
                    evidence = snippet[:200] if len(snippet) > 200 else snippet

                    normalized_results.append({
                        "claim": snippet[:350],  # Full content
                        "evidence": evidence,  # Clean factual evidence
                        "source": url,
                        "confidence": relevance,
                        "chunk_id": f"chunk_{len(normalized_results)}"
                    })
    except Exception as e:
        tool_outputs["web_search"] = {"success": False, "error": str(e)}

    # If web search empty, try data lookup
    if not normalized_results:
        try:
            data_result = await tool_registry.execute_with_fallback("data_lookup", {"query": query})
            tool_outputs["data_lookup"] = {"success": data_result.success, "latency": data_result.latency_ms}

            if data_result.success and data_result.data:
                data = data_result.data.get("data", {})
                results = data.get("results", [])
                row_count = data.get("row_count", 0)

                if results:
                    normalized_results.append({
                        "claim": f"Database query returned {row_count} results",
                        "evidence": str(results)[:250],
                        "source": "database_lookup",
                        "confidence": 0.8,
                        "chunk_id": f"chunk_{len(normalized_results)}"
                    })
        except Exception as e:
            tool_outputs["data_lookup"] = {"success": False, "error": str(e)}

    # If still no results, add a note
    if not normalized_results:
        normalized_results.append({
            "claim": f"Search performed for: {query}",
            "evidence": "No specific results found",
            "source": "web_search",
            "confidence": 0.5,
            "chunk_id": "chunk_0"
        })

    state["retrieval_results"] = normalized_results
    state["tool_outputs"] = tool_outputs

    return state


async def critique_node(state: AgentState) -> AgentState:
    """Critique - with proper evidence handling"""
    retrieval_results = state["retrieval_results"]
    query = state["user_query"]
    classification = state.get("metadata", {}).get("classification", {})

    if not classification.get("needs_critique", False) or not retrieval_results:
        state["critique_results"] = [{
            "claims": [],
            "confidence_scores": {},
            "disagreements": [],
            "overall_confidence": 1.0
        }]
        return state

    # Build detailed claims for critique
    claims_list = []
    for i, r in enumerate(retrieval_results):
        claim_text = r.get("claim", "")
        source = r.get("source", "unknown")
        conf = r.get("confidence", 0.5)
        claims_list.append(f"Claim {i+1} (source: {source}, base_confidence: {conf}): {claim_text}")

    claims_text = "\n".join(claims_list)

    critique_prompt = f"""Analyze each claim and assign a confidence score (0.0 to 1.0).

Query: {query}

{claims_text}

For each claim, evaluate:
- Is the claim factual and accurate?
- Is the source credible?
- Does the evidence support the claim?

Respond in EXACTLY this JSON format:
{{
  "confidence_scores": {{
    "full_claim_text_1": 0.85,
    "full_claim_text_2": 0.72
  }},
  "disagreements": [
    {{"span": "specific problematic text", "reason": "why it's questionable", "alternative": "better phrasing"}}
  ],
  "overall_confidence": 0.78
}}

IMPORTANT: Include EVERY claim in confidence_scores with a score. Use the full claim text as the key."""

    result = await llm_client.generate(critique_prompt, system_prompt="You critically evaluate factual claims and assign precise confidence scores.", max_tokens=3000)

    try:
        critique_data = json.loads(result.get("content", "{}"))
        # Validate it has required fields
        if not isinstance(critique_data.get("confidence_scores"), dict):
            critique_data["confidence_scores"] = {}
    except:
        critique_data = {"confidence_scores": {}, "disagreements": [], "overall_confidence": 0.7}

    # Build confidence scores - use critique scores where available, fallback to retrieval confidence
    confidence_scores = {}
    for r in retrieval_results:
        claim = r.get("claim", "")
        ret_conf = r.get("confidence", 0.7)

        # Try to find this claim in critique scores
        found = False
        for crit_key, crit_conf in critique_data.get("confidence_scores", {}).items():
            if claim[:50] in crit_key or crit_key[:50] in claim:
                confidence_scores[claim[:100]] = crit_conf
                found = True
                break

        if not found:
            # Use retrieval confidence as fallback
            confidence_scores[claim[:100]] = ret_conf

    # Calculate overall confidence from retrieval if critique didn't provide one
    if not critique_data.get("overall_confidence"):
        if retrieval_results:
            avg_conf = sum(r.get("confidence", 0.7) for r in retrieval_results) / len(retrieval_results)
            critique_data["overall_confidence"] = round(avg_conf, 2)

    state["critique_results"] = [{
        "claims": [r.get("claim", "") for r in retrieval_results],
        "confidence_scores": confidence_scores,
        "disagreements": critique_data.get("disagreements", []),
        "overall_confidence": critique_data.get("overall_confidence", 0.7)
    }]
    return state


async def synthesis_node(state: AgentState) -> AgentState:
    """Synthesis - with evidence-based response generation"""
    query = state["user_query"]
    retrieval_results = state.get("retrieval_results", [])
    critique_results = state.get("critique_results", [])
    classification = state.get("metadata", {}).get("classification", {})

    # Build context for LLM - prioritize evidence
    context_parts = []

    # Include retrieval evidence FIRST and prominently
    if retrieval_results and len(retrieval_results) > 0:
        evidence_lines = []
        for r in retrieval_results:
            claim = r.get("claim", "")
            source = r.get("source", "unknown")
            conf = r.get("confidence", 0.5)
            if claim and claim != f"Search performed for: {query}":
                evidence_lines.append(f"Source: {source}, Info: {claim} (relevance: {conf})")

        if evidence_lines:
            context_parts.append("FACTS RETRIEVED:")
            context_parts.append("\n".join(evidence_lines))
            context_parts.append("")
        else:
            context_parts.append("Some information was retrieved but needs synthesis.")
    else:
        context_parts.append("No specific evidence retrieved.")

    # Add category info
    context_parts.append(f"Query type: {classification.get('category', 'factual')}")
    context_parts.append(f"User asked: {query}")

    # Build a direct synthesis prompt that forces using the evidence
    synthesis_prompt = f"""Based on the retrieved information below, provide a direct answer to the user's question.

USER QUESTION: {query}

RETRIEVED INFORMATION:
{chr(10).join(context_parts)}

DIRECTIONS:
- Use the retrieved facts above to answer the question
- If you have retrieved information, answer using that information
- Do not say "I don't have information" if you just retrieved information
- Provide a direct, informative answer
- Be concise but complete

YOUR ANSWER:"""

    result = await llm_client.generate(synthesis_prompt, system_prompt="You are a helpful assistant that provides direct answers based on retrieved information.")

    # Check result properly
    if result.get("success") and result.get("content"):
        final_answer = result["content"]
    elif retrieval_results and len(retrieval_results) > 0:
        # Fallback: construct answer from retrieval results directly
        claims = [r.get("claim", "") for r in retrieval_results if r.get("claim")]
        final_answer = " ".join(claims[:2]) if claims else "Here is what I found about that."
    else:
        final_answer = "I'd be happy to help with that. Could you provide more details?"

    # Build provenance - keep full sentences
    provenance = []
    for r in retrieval_results:
        if r.get("claim"):
            # Get full claim, truncate only at sentence end if too long
            sentence = r.get("claim", "")
            if len(sentence) > 200:
                # Try to end at a sentence boundary
                for end_mark in ['.', '!', '?']:
                    last_idx = sentence[:200].rfind(end_mark)
                    if last_idx > 50:
                        sentence = sentence[:last_idx+1]
                        break
                else:
                    sentence = sentence[:200] + "..."

            provenance.append({
                "sentence": sentence,
                "source_agent": "retrieval_agent",
                "source_chunk": r.get("chunk_id", "unknown"),
                "confidence": r.get("confidence", 0.5)
            })

    # Resolve contradictions
    resolved = []
    if critique_results:
        for cr in critique_results:
            for d in cr.get("disagreements", []):
                resolved.append({
                    "original": d.get("span", ""),
                    "resolution": d.get("reason", "")
                })

    state["synthesis_output"] = {
        "final_answer": final_answer,
        "provenance": provenance,
        "resolved_contradictions": resolved
    }
    state["metadata"]["current_state"] = "completed"

    return state


def create_agent_graph() -> StateGraph:
    """Create LangGraph with synchronized routing"""
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("decomposition", decomposition_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("critique", critique_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("orchestrator")

    # Route based on routing plan from orchestrator
    def route_from_orchestrator(state: AgentState) -> str:
        order = state.get("routing_plan", {}).get("order", ["synthesis"])
        return order[0] if order else "synthesis"

    def route_after_retrieval(state: AgentState) -> str:
        classification = state.get("metadata", {}).get("classification", {})
        if classification.get("needs_critique", False):
            return "critique"
        return "synthesis"

    graph.add_conditional_edges("orchestrator", route_from_orchestrator, {
        "decomposition": "decomposition",
        "retrieval": "retrieval",
        "synthesis": "synthesis"
    })

    graph.add_edge("decomposition", "retrieval")
    graph.add_conditional_edges("retrieval", route_after_retrieval, {
        "critique": "critique",
        "synthesis": "synthesis"
    })
    graph.add_edge("critique", "synthesis")
    graph.add_edge("synthesis", END)

    return graph


agent_graph = create_agent_graph().compile()


async def run_agent_pipeline(job_id: str, query: str) -> Dict[str, Any]:
    """Run pipeline with proper retrieval"""
    initial_state: AgentState = {
        "job_id": job_id,
        "user_query": query,
        "current_agent": "orchestrator",
        "decomposed_tasks": [],
        "retrieval_results": [],
        "critique_results": [],
        "synthesis_output": None,
        "tool_outputs": {},
        "metadata": {"current_state": "initial"},
        "routing_plan": {}
    }

    final_state = await agent_graph.ainvoke(initial_state)

    return {
        "success": True,
        "final_answer": final_state.get("synthesis_output", {}).get("final_answer", "No answer"),
        "provenance": final_state.get("synthesis_output", {}).get("provenance", []),
        "resolved_contradictions": final_state.get("synthesis_output", {}).get("resolved_contradictions", []),
        "routing_plan": final_state.get("routing_plan", {}),
        "retrieval_results": final_state.get("retrieval_results", []),
        "classification": final_state.get("metadata", {}).get("classification", {}),
        "context": final_state
    }