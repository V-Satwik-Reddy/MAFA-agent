"""TRUE MCP Server for Market Research.

Exposes LSTM predictions and live news search via official MCP stdio protocol.
Run: python mcp_servers/market_research_server.py
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
from tools.market_research_tools import predict, search_live_news

# Create TRUE MCP Server
mcp = FastMCP("market-research-server")

# Valid ticker symbols supported by our LSTM models
VALID_TICKERS = ["AAPL", "AMZN", "ADBE", "GOOGL", "IBM", "JPM", "META", "MSFT", "NVDA", "ORCL", "TSLA"]


@mcp.tool()
def predict_next_day(symbol: str) -> str:
    """Predict next day's closing price using LSTM model.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA, GOOGL)
        
    Returns:
        JSON string with prediction result or error message
    """
    symbol = symbol.upper().strip()
    
    if symbol not in VALID_TICKERS:
        return json.dumps({
            "error": f"Ticker '{symbol}' not supported. Valid tickers: {VALID_TICKERS}",
            "success": False
        })
    
    try:
        prediction = predict(symbol)
        return json.dumps({
            "symbol": symbol,
            "predicted_close": float(prediction),
            "model": "LSTM",
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "symbol": symbol,
            "success": False
        })


@mcp.tool()
def get_live_news(query: str, num_results: int = 5) -> str:
    """Search for live financial news related to a stock or topic.
    
    Args:
        query: Search query (e.g., "AAPL earnings", "Tesla stock news")
        num_results: Number of news results to return (default: 5)
        
    Returns:
        JSON string with news articles or error message
    """
    try:
        news = search_live_news(query)
        return json.dumps({
            "query": query,
            "results": news,
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "query": query,
            "success": False
        })


@mcp.tool()
def get_market_analysis(symbol: str) -> str:
    """Get combined market analysis: LSTM prediction + recent news.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        JSON string with comprehensive market analysis
    """
    symbol = symbol.upper().strip()
    
    if symbol not in VALID_TICKERS:
        return json.dumps({
            "error": f"Ticker '{symbol}' not supported. Valid tickers: {VALID_TICKERS}",
            "success": False
        })
    
    analysis = {
        "symbol": symbol,
        "prediction": None,
        "news": None,
        "success": True
    }
    
    # Get LSTM prediction
    try:
        prediction = predict(symbol)
        analysis["prediction"] = {
            "predicted_close": float(prediction),
            "model": "LSTM"
        }
    except Exception as exc:
        analysis["prediction"] = {"error": str(exc)}
    
    # Get live news
    try:
        news = search_live_news(f"{symbol} stock")
        analysis["news"] = news
    except Exception as exc:
        analysis["news"] = {"error": str(exc)}
    
    return json.dumps(analysis)


if __name__ == "__main__":
    # Run as TRUE MCP server via stdio transport
    mcp.run(transport="stdio")
