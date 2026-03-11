"""MCP Event Bus - Redis Pub/Sub for Agent Communication.

Provides async event publishing and subscription for the MCP system.
All agents and the orchestrator use this for inter-component messaging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event Topics
# ---------------------------------------------------------------------------

class MCPTopics:
    """Standard MCP event topics for financial workflows."""
    
    # Market data events
    MARKET_RAW = "market.raw"
    MARKET_PREDICTIONS = "market.predictions"
    MARKET_NEWS = "market.news"
    
    # Strategy events
    STRATEGY_RECOMMENDATIONS = "strategy.recommendations"
    STRATEGY_ALERTS = "strategy.alerts"
    
    # Portfolio events
    PORTFOLIO_SNAPSHOTS = "portfolio.snapshots"
    PORTFOLIO_UPDATES = "portfolio.updates"
    
    # Execution events
    EXECUTION_ORDERS = "execution.orders"
    EXECUTION_RESULTS = "execution.results"
    EXECUTION_ERRORS = "execution.errors"
    
    # MCP orchestration events
    MCP_QUERY = "mcp.query"
    MCP_RESULTS = "mcp.results"
    MCP_ERRORS = "mcp.errors"
    
    @classmethod
    def all_topics(cls) -> List[str]:
        """Return all defined topics."""
        return [
            cls.MARKET_RAW, cls.MARKET_PREDICTIONS, cls.MARKET_NEWS,
            cls.STRATEGY_RECOMMENDATIONS, cls.STRATEGY_ALERTS,
            cls.PORTFOLIO_SNAPSHOTS, cls.PORTFOLIO_UPDATES,
            cls.EXECUTION_ORDERS, cls.EXECUTION_RESULTS, cls.EXECUTION_ERRORS,
            cls.MCP_QUERY, cls.MCP_RESULTS, cls.MCP_ERRORS,
        ]


# ---------------------------------------------------------------------------
# Event Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MCPEvent:
    """Standard MCP event structure."""
    
    topic: str
    user_id: int
    payload: Dict[str, Any]
    timestamp: float
    event_id: Optional[str] = None
    source: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPEvent":
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "MCPEvent":
        return cls.from_dict(json.loads(json_str))


# NOTE: The specialised event subclasses (MarketPredictionEvent,
# TradeExecutionEvent, StrategyRecommendationEvent) were removed during
# cleanup — they were never instantiated by any module.  If you need typed
# events in the future, create them in the module that publishes them.


# ---------------------------------------------------------------------------
# Event Bus Implementation
# ---------------------------------------------------------------------------

class MCPEventBus:
    """Redis-backed event bus for MCP agent communication."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
    ):
        self.redis_url = redis_url
        self.db = db
        self._redis: Optional[redis.Redis] = None
        self._pubsub = None
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._listener_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> None:
        """Connect to Redis and verify connectivity with PING."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                db=self.db,
                decode_responses=True,
            )
        # Verify real connectivity (lazy client won't fail until first command)
        await self._redis.ping()
        logger.info(f"Connected to Redis at {self.redis_url}")
        if self._pubsub is None:
            self._pubsub = self._redis.pubsub()
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")
    
    async def publish(self, event: MCPEvent) -> int:
        """Publish an event to a topic."""
        await self.connect()
        message = event.to_json()
        # redis.asyncio publish returns number of clients that received the message
        if self._redis is None:
            raise RuntimeError("Redis connection not established")
        count = await self._redis.publish(event.topic, message)
        logger.debug(f"Published to {event.topic}: {count} subscribers")
        return count
    
    async def publish_raw(self, topic: str, user_id: int, payload: Dict[str, Any]) -> int:
        """Publish a raw event (convenience method).
        
        Args:
            topic: Event topic
            user_id: User ID
            payload: Event payload
            
        Returns:
            Number of subscribers
        """
        event = MCPEvent(
            topic=topic,
            user_id=user_id,
            payload=payload,
            timestamp=time.time(),
        )
        return await self.publish(event)
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[MCPEvent], Any],
    ) -> None:
        """Subscribe to a topic with a callback."""
        await self.connect()
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        self._subscriptions[topic].append(callback)
        if self._pubsub is None:
            raise RuntimeError("Redis pubsub not initialized")
        await self._pubsub.subscribe(topic)
        logger.info(f"Subscribed to {topic}")
        # Start listener if not running
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())
    
    async def subscribe_many(
        self,
        topics: List[str],
        callback: Callable[[MCPEvent], Any],
    ) -> None:
        """Subscribe to multiple topics with the same callback."""
        for topic in topics:
            await self.subscribe(topic, callback)
    
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic."""
        if self._pubsub:
            await self._pubsub.unsubscribe(topic)
            self._subscriptions.pop(topic, None)
            logger.info(f"Unsubscribed from {topic}")
    
    async def _listen(self) -> None:
        """Internal listener loop with automatic reconnection on failure.

        On transient errors the loop waits with exponential back-off
        (1 s → 2 s → 4 s … capped at 30 s) and re-subscribes.
        """
        backoff = 1.0
        max_backoff = 30.0
        while True:
            try:
                if self._pubsub is None:
                    raise RuntimeError("Redis pubsub not initialized")
                async for message in self._pubsub.listen():
                    backoff = 1.0  # reset on every successful message
                    if message["type"] == "message":
                        topic = message["channel"]
                        data = message["data"]
                        try:
                            event = MCPEvent.from_json(data)
                            callbacks = self._subscriptions.get(topic, [])
                            for callback in callbacks:
                                try:
                                    result = callback(event)
                                    if asyncio.iscoroutine(result):
                                        await result
                                except Exception as exc:
                                    logger.error(f"Callback error for {topic}: {exc}")
                        except json.JSONDecodeError as exc:
                            logger.error(f"Invalid JSON on {topic}: {exc}")
            except asyncio.CancelledError:
                logger.info("Event listener cancelled")
                raise
            except Exception as exc:
                logger.warning(f"Event listener error (reconnecting in {backoff:.0f}s): {exc}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                # Attempt to re-establish pubsub and re-subscribe
                try:
                    if self._redis:
                        self._pubsub = self._redis.pubsub()
                        for topic in self._subscriptions:
                            await self._pubsub.subscribe(topic)
                        logger.info("Re-subscribed to %d topics after reconnect", len(self._subscriptions))
                except Exception as re_exc:
                    logger.error(f"Reconnection failed: {re_exc}")


# ---------------------------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------------------------

_event_bus: Optional[MCPEventBus] = None


def get_event_bus() -> MCPEventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = MCPEventBus()
    return _event_bus


async def shutdown_event_bus() -> None:
    """Shutdown the global event bus."""
    global _event_bus
    if _event_bus:
        await _event_bus.disconnect()
        _event_bus = None
