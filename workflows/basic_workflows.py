"""Basic multi-agent workflows.

Uses simple sequential orchestration; swap with LangGraph graphs when ready.
"""

from typing import Dict, List

from agents.market_search_agent import run_market_research_agent
from agents.investment_strategy_agent import run_investment_strategy_agent
from agents.portfolio_manager_agent import run_portfolio_manager_agent
from agents.execution_agent import run_execute_agent


def _investment_advice(user_id: int, query: str, intent: Dict) -> Dict:
    research = run_market_research_agent(user_message=query, user_id=user_id)
    strategy_query = (
        f"User asked: {query}\n\nMarket context:\n{research}\n"
        "Provide a concise investment recommendation based on this analysis."
    )
    strategy = run_investment_strategy_agent(user_message=strategy_query, user_id=user_id)
    response = _synthesize_advice(query, research, strategy)
    return {
        "response": response,
        "workflow": "investment_advice",
        "steps": ["market_research", "investment_strategy", "synthesize"],
        "metadata": {
            "research": research,
            "strategy": strategy,
            "agents": ["market_research_agent", "investment_strategy_agent"],
        },
    }


def _rebalancing(user_id: int, query: str, intent: Dict) -> Dict:
    portfolio = run_portfolio_manager_agent(user_message="Analyze my portfolio", user_id=user_id)
    strategy = run_investment_strategy_agent(
        user_message=f"Current portfolio: {portfolio}\nSuggest rebalance plan.",
        user_id=user_id,
    )
    research = run_market_research_agent(user_message="Current macro and sector risks", user_id=user_id)
    execution_plan = run_execute_agent(
        user_message=f"Draft execution plan for: {strategy}\nUse context: {research}",
        user_id=user_id,
    )
    response = execution_plan or strategy
    return {
        "response": response,
        "workflow": "rebalancing",
        "steps": ["portfolio", "strategy", "research", "execution"],
        "metadata": {
            "portfolio": portfolio,
            "strategy": strategy,
            "research": research,
            "execution_plan": execution_plan,
            "agents": [
                "portfolio_manager_agent",
                "investment_strategy_agent",
                "market_research_agent",
                "execution_agent",
            ],
        },
    }


def _synthesize_advice(query: str, research: str, strategy: str) -> str:
    return (
        f"Question: {query}\n\nMarket view:\n{research}\n\nStrategy view:\n{strategy}\n\n"
        "Combined: Focus on fit to risk profile, size positions modestly, and review monthly."
    )


WORKFLOW_CATALOG = {
    "investment_advice": _investment_advice,
    "rebalancing": _rebalancing,
}


def execute_workflow(workflow_name: str, user_id: int, query: str, intent: Dict) -> Dict:
    fn = WORKFLOW_CATALOG.get(workflow_name)
    if not fn:
        return {"error": f"workflow {workflow_name} not found", "response": ""}
    return fn(user_id, query, intent)
