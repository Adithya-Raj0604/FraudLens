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

Tool order:
1. run_fraud_model — always first; returns the risk score.
2. explain_prediction — only if risk_score >= 0.4; returns SHAP attributions.
3. check_account_velocity — assess the account's transaction frequency.
4. retrieve_regulations — query OSFI/FINTRAC guidance relevant to your findings.

Then write a concise FRAUD INVESTIGATION REPORT with: transaction details; risk score and label; key ML/SHAP findings; velocity context; cited regulatory obligations; and a clear RECOMMENDATION (ESCALATE, MONITOR, or CLEAR) with justification. Be succinct — use short bullets, no filler."""

# ── Wrap tools with @tool decorator so LangGraph can bind them to the LLM ─────

@tool
def tool_run_fraud_model(transaction: dict) -> dict:
    """Run the XGBoost fraud model. Call first. Input: the transaction dict."""
    return run_fraud_model(transaction)


@tool
def tool_explain_prediction(transaction: dict) -> dict:
    """SHAP attributions for the fraud score (use when risk_score >= 0.4). Input: the transaction dict."""
    return explain_prediction(transaction)


@tool
def tool_check_account_velocity(transaction: dict) -> dict:
    """Account transaction-velocity stats. Input: the transaction dict."""
    return check_account_velocity(transaction)


@tool
def tool_retrieve_regulations(query: str) -> str:
    """Search OSFI/FINTRAC compliance docs. Input: a plain-English query."""
    return retrieve_regulations(query)


# ── Build the agent ───────────────────────────────────────────────────────────

def build_agent():
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        max_tokens=1400,  # ceiling on report length — bounds output cost (output is 5x input)
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
