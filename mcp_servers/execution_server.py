"""TRUE MCP Server for Trade Execution.

Exposes trade execution, balance checking, and holdings via official MCP stdio protocol.
Run: python mcp_servers/execution_server.py
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
from tools.execute_trade_tools import execute_trade_order
from tools.profile_tools import get_user_balance, get_user_holdings, get_current_stock_price

# Create TRUE MCP Server
mcp = FastMCP("execution-server")


@mcp.tool()
def execute_trade(symbol: str, quantity: int, action: str) -> str:
    """Execute a trade order (buy/sell) with validation.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)
        quantity: Number of shares to trade (must be positive integer)
        action: Trade action - either "buy" or "sell"
        
    Returns:
        JSON string with execution result or error message
    """
    action = action.lower().strip()
    symbol = symbol.upper().strip()
    
    if action not in ["buy", "sell"]:
        return json.dumps({
            "error": f"Invalid action '{action}'. Must be 'buy' or 'sell'.",
            "success": False
        })
    
    if quantity <= 0:
        return json.dumps({
            "error": "Quantity must be a positive integer.",
            "success": False
        })
    
    try:
        order_data = {
            "symbol": symbol,
            "quantity": quantity
        }
        result = execute_trade_order(order_data, action)
        return json.dumps({
            "order_id": result,
            "symbol": symbol,
            "quantity": quantity,
            "action": action,
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "symbol": symbol,
            "quantity": quantity,
            "action": action,
            "success": False
        })


@mcp.tool()
def check_balance() -> str:
    """Check user's available trading balance.
    
    Returns:
        JSON string with current balance or error message
    """
    try:
        balance = get_user_balance()
        return json.dumps({
            "balance": float(balance),
            "currency": "USD",
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def check_holdings() -> str:
    """Get user's current stock holdings.
    
    Returns:
        JSON string with holdings dictionary or error message
    """
    try:
        holdings = get_user_holdings()
        return json.dumps({
            "holdings": holdings,
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)
        
    Returns:
        JSON string with current price or error message
    """
    symbol = symbol.upper().strip()
    
    try:
        price = get_current_stock_price(symbol)
        return json.dumps({
            "symbol": symbol,
            "price": float(price),
            "currency": "USD",
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "symbol": symbol,
            "success": False
        })


@mcp.tool()
def validate_trade(symbol: str, quantity: int, action: str) -> str:
    """Validate a trade before execution (check balance/holdings).
    
    Args:
        symbol: Stock ticker symbol
        quantity: Number of shares
        action: Trade action - "buy" or "sell"
        
    Returns:
        JSON string with validation result
    """
    action = action.lower().strip()
    symbol = symbol.upper().strip()
    
    validation = {
        "symbol": symbol,
        "quantity": quantity,
        "action": action,
        "valid": True,
        "issues": []
    }
    
    try:
        if action == "buy":
            # Check balance
            balance = get_user_balance()
            price = get_current_stock_price(symbol)
            total_cost = price * quantity
            
            if total_cost > balance:
                validation["valid"] = False
                validation["issues"].append(
                    f"Insufficient balance. Need ${total_cost:.2f}, have ${balance:.2f}"
                )
            else:
                validation["estimated_cost"] = total_cost
                validation["remaining_balance"] = balance - total_cost
                
        elif action == "sell":
            # Check holdings
            holdings = get_user_holdings()
            current_qty = holdings.get(symbol, 0)
            
            if current_qty < quantity:
                validation["valid"] = False
                validation["issues"].append(
                    f"Insufficient shares. Own {current_qty}, trying to sell {quantity}"
                )
            else:
                price = get_current_stock_price(symbol)
                validation["estimated_proceeds"] = price * quantity
        else:
            validation["valid"] = False
            validation["issues"].append(f"Invalid action: {action}")
            
    except Exception as exc:
        validation["valid"] = False
        validation["issues"].append(f"Validation error: {str(exc)}")
    
    validation["success"] = True
    return json.dumps(validation)


if __name__ == "__main__":
    # Run as TRUE MCP server via stdio transport
    mcp.run(transport="stdio")
