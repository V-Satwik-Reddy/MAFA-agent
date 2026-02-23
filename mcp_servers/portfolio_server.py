"""TRUE MCP Server for Portfolio Management.

Exposes portfolio analysis, holdings tracking, and metrics via official MCP stdio protocol.
Run: python mcp_servers/portfolio_server.py
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
from tools.profile_tools import get_user_holdings, get_current_stock_price, get_user_balance

# Create TRUE MCP Server
mcp = FastMCP("portfolio-server")


@mcp.tool()
def get_portfolio_snapshot() -> str:
    """Get complete portfolio snapshot with current values.
    
    Returns:
        JSON string with portfolio holdings, values, and totals
    """
    try:
        holdings = get_user_holdings()
        balance = get_user_balance()
        
        portfolio = {
            "holdings": [],
            "cash_balance": float(balance),
            "total_invested": 0.0,
            "total_portfolio_value": float(balance),
            "success": True
        }
        
        for symbol, quantity in holdings.items():
            try:
                price = get_current_stock_price(symbol)
                value = price * quantity
                portfolio["holdings"].append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "current_price": float(price),
                    "market_value": float(value)
                })
                portfolio["total_invested"] += value
                portfolio["total_portfolio_value"] += value
            except Exception:
                portfolio["holdings"].append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "current_price": None,
                    "market_value": None,
                    "error": "Could not fetch price"
                })
        
        return json.dumps(portfolio)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def get_portfolio_allocation() -> str:
    """Get portfolio allocation percentages by holding.
    
    Returns:
        JSON string with allocation breakdown
    """
    try:
        holdings = get_user_holdings()
        balance = get_user_balance()
        
        allocations = []
        total_value = float(balance)
        
        # Calculate total portfolio value
        for symbol, quantity in holdings.items():
            try:
                price = get_current_stock_price(symbol)
                value = price * quantity
                total_value += value
                allocations.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "value": float(value)
                })
            except Exception:
                pass
        
        # Calculate percentages
        result = {
            "allocations": [],
            "cash_percentage": (float(balance) / total_value * 100) if total_value > 0 else 100,
            "total_value": total_value,
            "success": True
        }
        
        for alloc in allocations:
            percentage = (alloc["value"] / total_value * 100) if total_value > 0 else 0
            result["allocations"].append({
                "symbol": alloc["symbol"],
                "quantity": alloc["quantity"],
                "value": alloc["value"],
                "percentage": round(percentage, 2)
            })
        
        return json.dumps(result)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def analyze_portfolio_risk() -> str:
    """Analyze portfolio risk metrics and concentration.
    
    Returns:
        JSON string with risk analysis
    """
    try:
        holdings = get_user_holdings()
        balance = get_user_balance()
        
        # Calculate values and concentration
        positions = []
        total_invested = 0.0
        
        for symbol, quantity in holdings.items():
            try:
                price = get_current_stock_price(symbol)
                value = price * quantity
                positions.append({
                    "symbol": symbol,
                    "value": float(value)
                })
                total_invested += value
            except Exception:
                pass
        
        total_portfolio = total_invested + float(balance)
        
        # Risk metrics
        risk_analysis = {
            "total_portfolio_value": total_portfolio,
            "cash_percentage": (float(balance) / total_portfolio * 100) if total_portfolio > 0 else 100,
            "invested_percentage": (total_invested / total_portfolio * 100) if total_portfolio > 0 else 0,
            "position_count": len(positions),
            "concentration_risk": "low",
            "warnings": [],
            "success": True
        }
        
        # Check concentration risk
        for pos in positions:
            pct = (pos["value"] / total_portfolio * 100) if total_portfolio > 0 else 0
            if pct > 25:
                risk_analysis["concentration_risk"] = "high"
                risk_analysis["warnings"].append(
                    f"{pos['symbol']} represents {pct:.1f}% of portfolio (>25%)"
                )
            elif pct > 15:
                if risk_analysis["concentration_risk"] != "high":
                    risk_analysis["concentration_risk"] = "medium"
                risk_analysis["warnings"].append(
                    f"{pos['symbol']} represents {pct:.1f}% of portfolio (>15%)"
                )
        
        # Check cash level
        cash_pct = risk_analysis["cash_percentage"]
        if cash_pct < 5:
            risk_analysis["warnings"].append(
                f"Low cash reserves ({cash_pct:.1f}%). Consider maintaining 5-10%."
            )
        elif cash_pct > 50:
            risk_analysis["warnings"].append(
                f"High cash allocation ({cash_pct:.1f}%). Consider investing."
            )
        
        return json.dumps(risk_analysis)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def check_position(symbol: str) -> str:
    """Check if user has a position in a specific stock.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        JSON string with position details
    """
    symbol = symbol.upper().strip()
    
    try:
        holdings = get_user_holdings()
        quantity = holdings.get(symbol, 0)
        
        result = {
            "symbol": symbol,
            "has_position": quantity > 0,
            "quantity": quantity,
            "success": True
        }
        
        if quantity > 0:
            try:
                price = get_current_stock_price(symbol)
                result["current_price"] = float(price)
                result["market_value"] = float(price * quantity)
            except Exception:
                pass
        
        return json.dumps(result)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "symbol": symbol,
            "success": False
        })


if __name__ == "__main__":
    # Run as TRUE MCP server via stdio transport
    mcp.run(transport="stdio")
