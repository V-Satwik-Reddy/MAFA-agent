import requests
import asyncio
import websockets
import json
import time

BASE_URL = "http://localhost:5001"
WS_URL = "ws://localhost:5001/ws/mcp-stream"
TOKEN_HEADER = {"Authorization": "Bearer test-token-123"}

def test_health():
    print("\n--- Testing Health Check ---")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(f"✅ Health Check Passed: {response.json()}")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_mcp_servers():
    print("\n--- Testing MCP Server List ---")
    try:
        response = requests.get(f"{BASE_URL}/mcp/servers")
        if response.status_code == 200:
            data = response.json()
            servers = data.get("servers", {})
            if all(k in servers for k in ["market", "execution", "portfolio", "strategy"]):
                print(f"✅ MCP Servers Listed: {list(servers.keys())}")
                return True
            else:
                print(f"❌ Missing MCP Servers: found {list(servers.keys())}")
                return False
        else:
            print(f"❌ Server List Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server List Error: {e}")
        return False

def test_direct_mcp_market():
    print("\n--- Testing Direct MCP Market Tool (Predict) ---")
    try:
        # Note: This might fail if LSTM models aren't trained, but we check specific error handling
        response = requests.post(
            f"{BASE_URL}/mcp/market/predict", 
            params={"symbol": "AAPL"},
            headers=TOKEN_HEADER
        )
        if response.status_code == 200:
            print(f"✅ Market Predict Response: {response.json()}")
            return True
        else:
            print(f"❌ Market Predict Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Market Predict Error: {e}")
        return False

def test_mcp_orchestrator():
    print("\n--- Testing MCP Orchestrator (/mcp/query) ---")
    payload = {
        "query": "What is the price of AAPL and should I buy it?",
        "userId": 1,
        "sessionId": "test-session-001"
    }
    try:
        response = requests.post(f"{BASE_URL}/mcp/query", json=payload, headers=TOKEN_HEADER)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Orchestrator Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Orchestrator Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Orchestrator Error: {e}")
        return False

async def test_websocket():
    print("\n--- Testing WebSocket Event Stream ---")
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✅ WebSocket Connected")
            
            # Send a ping/ack test
            await websocket.send("test-ping")
            response = await websocket.recv()
            print(f"✅ WebSocket Received: {response}")
            
            # Wait a bit to see if any broadcast events come through (unlikely without triggering one)
            # But the connection itself proves the endpoint works
            
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")

def run_tests():
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    if not test_health():
        print("Aborting tests due to health check failure.")
        return

    test_mcp_servers()
    test_direct_mcp_market()
    test_mcp_orchestrator()
    
    # Run async test
    asyncio.run(test_websocket())

if __name__ == "__main__":
    run_tests()
