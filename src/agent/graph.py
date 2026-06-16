"""
LangGraph ReAct fraud investigation agent.

Graph structure:
    START → agent → (tool_call?) → tools → agent → ... → END

The LLM decides which tools to call and in what order based on what
it finds. It stops when it has enough information to write the report.
"""

import sys
import os

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from src.agent.tools import (
    run_fraud_model,
    explain_prediction,
    check_account_velocity,
    retrieve_regulations,
)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert fraud investigator at a Canadian financial institution.

You have four tools available:
- run_fraud_model: always call this first to get the fraud risk score
- explain_prediction: call this when risk_score >= 0.4 to get SHAP feature attributions
- check_account_velocity: call this to get the account's recent transaction frequency
- retrieve_regulations: call this to retrieve relevant OSFI/FINTRAC regulatory guidance

Investigation protocol:
1. Always call run_fraud_model first
2. If risk_score >= 0.4, call explain_prediction to understand why
3. Always call check_account_velocity to assess behavioural patterns
4. Call retrieve_regulations with a query relevant to what you found (e.g. "balance drain suspicious transaction reporting", "account velocity escalation threshold")
5. After all tool calls, write a structured investigator report

Your final report must include:
- FRAUD INVESTIGATION REPORT header with transaction details
- Risk Score and Risk Label
- Key findings from the ML model and SHAP explainability
- Behavioural context from velocity analysis
- Relevant regulatory obligations cited from the documents retrieved
- A clear RECOMMENDATION: ESCALATE, MONITOR, or CLEAR with justification
"""

# ── Wrap tools with @tool decorator so LangGraph can bind them to the LLM ─────

@tool
def tool_run_fraud_model(transaction: dict) -> dict:
    """Run the XGBoost fraud model on a transaction. Always call this first.
    Input: dict with keys step, type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest, velocity_cumcount, velocity_1hr,
    velocity_3hr, velocity_24hr."""
    return run_fraud_model(transaction)


@tool
def tool_explain_prediction(transaction: dict) -> dict:
    """Get SHAP feature attributions explaining the fraud score.
    Call this when risk_score >= 0.4.
    Input: same transaction dict as run_fraud_model."""
    return explain_prediction(transaction)


@tool
def tool_check_account_velocity(account_id: str, step: int) -> dict:
    """Get recent transaction velocity stats for an account.
    Input: account_id (string like C1234567890), step (integer hour 1-743)."""
    return check_account_velocity(account_id, step)


@tool
def tool_retrieve_regulations(query: str) -> str:
    """Search OSFI/FINTRAC compliance documents for relevant regulatory guidance.
    Input: plain English query describing what regulation you need.
    Example: 'suspicious transaction reporting deadline' or 'velocity escalation threshold'"""
    return retrieve_regulations(query)


# ── Build the agent ───────────────────────────────────────────────────────────

def build_agent():
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    tools = [
        tool_run_fraud_model,
        tool_explain_prediction,
        tool_check_account_velocity,
        tool_retrieve_regulations,
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent


# ── Module-level agent instance ───────────────────────────────────────────────
# Built once at import time. graph.py is only imported after FastAPI startup
# so state.model is already populated when this runs.

agent = build_agent()


# ── Public interface ──────────────────────────────────────────────────────────

def invoke(transaction: dict) -> str:
    """Run the full investigation and return the final report string."""
    result = agent.invoke({"messages": [str(transaction)]})
    return result["messages"][-1].content


def stream(transaction: dict):
    """
    Stream investigation steps as they happen.
    Yields dicts describing each event — tool calls, tool results, final report.
    Each yielded dict has keys: type, name (optional), content.
    """
    for event in agent.stream({"messages": [str(transaction)]}, stream_mode="updates"):
        for node_name, node_output in event.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                msg_type = type(msg).__name__

                if msg_type == "AIMessage":
                    # LLM reasoning step — may contain tool call requests or final answer
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield {"type": "tool_call", "name": tc["name"], "input": tc["args"]}
                    elif msg.content:
                        yield {"type": "report", "content": msg.content}

                elif msg_type == "ToolMessage":
                    # Result returned from a tool execution
                    yield {"type": "tool_result", "name": msg.name, "content": str(msg.content)[:500]}
