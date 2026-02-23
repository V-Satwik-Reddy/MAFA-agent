"""TRUE MCP Server for Investment Strategy.

Exposes investment recommendations, risk assessment, and rebalancing via official MCP stdio protocol.
Run: python mcp_servers/strategy_server.py
"""

from __future__ import annotations

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# Import existing tools
from tools.investment_strategy_tools import (
    assess_risk_tolerance,
    analyze_portfolio_alignment,
    generate_personalized_strategy,
    calculate_optimal_allocation,
    track_strategy_adherence,
)
from tools.profile_tools import get_user_holdings, get_current_stock_price, get_user_balance
from tools.market_research_tools import predict

# Create TRUE MCP Server
mcp = FastMCP("strategy-server")

# Valid tickers for predictions
VALID_TICKERS = ["AAPL", "AMZN", "ADBE", "GOOGL", "IBM", "JPM", "META", "MSFT", "NVDA", "ORCL", "TSLA"]


@mcp.tool()
def generate_investment_recommendation(symbol: str, user_id: int) -> str:
    """Generate a complete investment recommendation for a symbol.
    
    Combines risk profile, portfolio fit, and price prediction to produce
    a buy/sell/hold recommendation with confidence and reasoning.
    
    Args:
        symbol: Stock ticker symbol
        user_id: User ID for personalization
        
    Returns:
        JSON string with recommendation details
    """
    symbol = symbol.upper().strip()
    
    recommendation = {
        "symbol": symbol,
        "user_id": user_id,
        "action": "hold",
        "confidence": 0.5,
        "position_size_pct": 0.0,
        "reasoning": [],
        "success": True
    }
    
    try:
        # 1. Assess user risk tolerance
        risk_result = assess_risk_tolerance(user_id)
        risk_data = json.loads(risk_result) if isinstance(risk_result, str) else risk_result
        risk_level = risk_data.get("risk_level", "moderate")
        recommendation["risk_profile"] = risk_level
        recommendation["reasoning"].append(f"Risk profile: {risk_level}")
        
        # 2. Check portfolio alignment
        alignment_result = analyze_portfolio_alignment(user_id, None)
        alignment_data = json.loads(alignment_result) if isinstance(alignment_result, str) else alignment_result
        recommendation["portfolio_alignment"] = alignment_data
        
        # 3. Get price prediction if available
        if symbol in VALID_TICKERS:
            try:
                predicted_price = predict(symbol)
                current_price = get_current_stock_price(symbol)
                price_change_pct = ((predicted_price - current_price) / current_price) * 100
                
                recommendation["prediction"] = {
                    "current_price": float(current_price),
                    "predicted_price": float(predicted_price),
                    "expected_change_pct": round(price_change_pct, 2)
                }
                
                # Generate recommendation based on prediction
                if price_change_pct > 3:
                    recommendation["action"] = "buy"
                    recommendation["confidence"] = min(0.85, 0.6 + price_change_pct / 20)
                    recommendation["reasoning"].append(
                        f"Bullish: Predicted {price_change_pct:.1f}% upside"
                    )
                elif price_change_pct < -3:
                    recommendation["action"] = "sell"
                    recommendation["confidence"] = min(0.85, 0.6 + abs(price_change_pct) / 20)
                    recommendation["reasoning"].append(
                        f"Bearish: Predicted {price_change_pct:.1f}% downside"
                    )
                else:
                    recommendation["action"] = "hold"
                    recommendation["confidence"] = 0.6
                    recommendation["reasoning"].append(
                        f"Neutral: Predicted {price_change_pct:.1f}% change"
                    )
                    
            except Exception as exc:
                recommendation["reasoning"].append(f"Prediction unavailable: {str(exc)}")
        else:
            recommendation["reasoning"].append(f"No LSTM model available for {symbol}")
        
        # 4. Calculate position size based on risk
        risk_multiplier = {"conservative": 0.03, "moderate": 0.05, "aggressive": 0.08}.get(risk_level, 0.05)
        recommendation["position_size_pct"] = risk_multiplier * 100
        recommendation["reasoning"].append(
            f"Suggested position size: {recommendation['position_size_pct']:.1f}% of portfolio"
        )
        
    except Exception as exc:
        recommendation["error"] = str(exc)
        recommendation["success"] = False
    
    return json.dumps(recommendation)


@mcp.tool()
def get_risk_profile(user_id: int) -> str:
    """Get user's risk tolerance profile.
    
    Args:
        user_id: User ID
        
    Returns:
        JSON string with risk profile details
    """
    try:
        result = assess_risk_tolerance(user_id)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "user_id": user_id,
            "success": False
        })


@mcp.tool()
def portfolio_rebalancing_proposal(user_id: int) -> str:
    """Generate a complete portfolio rebalancing proposal.
    
    Analyzes current allocation vs target and suggests trades.
    
    Args:
        user_id: User ID
        
    Returns:
        JSON string with rebalancing trades and reasoning
    """
    try:
        # Get current portfolio
        holdings = get_user_holdings()
        balance = get_user_balance()
        
        # Calculate current values
        positions = []
        total_invested = 0.0
        
        for symbol, quantity in holdings.items():
            try:
                price = get_current_stock_price(symbol)
                value = price * quantity
                positions.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "value": float(value),
                    "price": float(price)
                })
                total_invested += value
            except Exception:
                pass
        
        total_portfolio = total_invested + float(balance)
        
        # Get target allocation
        allocation_result = calculate_optimal_allocation(user_id, "balanced")
        allocation_data = json.loads(allocation_result) if isinstance(allocation_result, str) else allocation_result
        
        # Generate rebalancing proposal
        proposal = {
            "user_id": user_id,
            "current_portfolio_value": total_portfolio,
            "cash_balance": float(balance),
            "current_positions": positions,
            "suggested_trades": [],
            "reasoning": [],
            "success": True
        }
        
        # Analyze each position
        for pos in positions:
            current_pct = (pos["value"] / total_portfolio * 100) if total_portfolio > 0 else 0
            
            # Flag overweight positions
            if current_pct > 20:
                trim_pct = current_pct - 15
                trim_value = total_portfolio * (trim_pct / 100)
                trim_shares = int(trim_value / pos["price"])
                
                if trim_shares > 0:
                    proposal["suggested_trades"].append({
                        "action": "sell",
                        "symbol": pos["symbol"],
                        "quantity": trim_shares,
                        "reason": f"Reduce concentration from {current_pct:.1f}% to ~15%"
                    })
                    proposal["reasoning"].append(
                        f"{pos['symbol']} is overweight at {current_pct:.1f}%"
                    )
        
        # Check for underinvested cash
        cash_pct = (float(balance) / total_portfolio * 100) if total_portfolio > 0 else 100
        if cash_pct > 30:
            proposal["reasoning"].append(
                f"High cash allocation ({cash_pct:.1f}%). Consider deploying capital."
            )
        
        return json.dumps(proposal)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "user_id": user_id,
            "success": False
        })


@mcp.tool()
def get_strategy_for_goal(user_id: int, goal: str, time_horizon: str) -> str:
    """Generate personalized investment strategy based on goal.
    
    Args:
        user_id: User ID
        goal: Investment goal (e.g., "retirement", "growth", "income")
        time_horizon: Time horizon (e.g., "short", "medium", "long")
        
    Returns:
        JSON string with strategy recommendation
    """
    try:
        result = generate_personalized_strategy(user_id, goal, time_horizon)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "user_id": user_id,
            "goal": goal,
            "time_horizon": time_horizon,
            "success": False
        })


@mcp.tool()
def check_strategy_adherence(user_id: int) -> str:
    """Check how well user is following their investment strategy.
    
    Args:
        user_id: User ID
        
    Returns:
        JSON string with adherence metrics
    """
    try:
        result = track_strategy_adherence(user_id)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "user_id": user_id,
            "success": False
        })


if __name__ == "__main__":
    # Run as TRUE MCP server via stdio transport
    mcp.run(transport="stdio")
