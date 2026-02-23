# MAFA-B Services Status

> Tracks which MAFA-B services MAFA-agents depends on and their implementation status.
> Last updated: June 2025

---

## ✅ IMPLEMENTED & WIRED

### 1. Sector-per-Stock Mapping
- **Endpoints**: `GET /companies/{symbol}` (cached), `POST /companies/by-symbols` (bulk)
- **MAFA-agents usage**: `profile_tools.get_company_by_symbol()`, `profile_tools.get_companies_by_symbols()`
- **Consumed by**: Investment Strategy Agent (sector concentration), Portfolio Manager Agent (sector allocation), Market Research Agent (company info), Execution Agent (ticker verification)

### 2. Portfolio Value History
- **Endpoint**: `GET /portfolio/history?period=X&interval=Y`
- **Params**: period: `LAST_24_HOURS|LAST_7_DAYS|LAST_30_DAYS|LAST_90_DAYS|LAST_1_YEAR|ALL`, interval: `DAILY|WEEKLY|MONTHLY|QUARTERLY|YEARLY`
- **Response**: `ApiResponse { data: List<PortfolioDailySnapshotDTO> }` `{date, totalValue, cashBalance, investedValue}`
- **MAFA-agents usage**: `profile_tools.get_portfolio_history()`
- **Consumed by**: Portfolio Manager Agent (trend analysis), Investment Strategy Agent (adherence tracking, monthly trend)

### 3. Watchlist CRUD
- **Endpoints**: `GET /watchlist`, `POST /watchlist`, `DELETE /watchlist/{symbol}`
- **Response shape**: `WatchlistDto { company: CompanyDto, addedAt }`
- **MAFA-agents usage**: `profile_tools.get_watchlist()`, `profile_tools.add_to_watchlist()`, `profile_tools.remove_from_watchlist()`
- **Consumed by**: General Agent, Portfolio Manager Agent

### 4. Strategy Persistence
- **Endpoints**: `GET /strategy` (active; 404 if none), `GET /strategy/history`, `POST /strategy`, `PUT /strategy/{id}`
- **StrategyRequestDto**: `{strategyType, goal, timeHorizonMonths, riskProfile(CONSERVATIVE|MODERATE|AGGRESSIVE), targetAllocation(Map<String,Integer>), sectorLimits(Map<String,Integer>), rebalancingFrequency(MONTHLY|QUARTERLY|ANNUALLY|NONE)}`
- **StrategyDto**: `{id, strategyType, goal, timeHorizonMonths, riskProfile, targetAllocation, sectorLimits, rebalancingFrequency, active(boolean), createdAt, updatedAt}`
- **MAFA-agents usage**: `strategy_tools.get_active_strategy()`, `strategy_tools.get_strategy_history()`, `strategy_tools.save_strategy()`, `strategy_tools.update_strategy()`
- **Consumed by**: Investment Strategy Agent (CRUD + adherence analysis via `track_strategy_adherence()`)

### 7. Alerts / Notifications
- **Endpoints**: `POST /alerts`, `GET /alerts?status=X`, `DELETE /alerts/{id}` (soft-delete → CANCELLED)
- **AlertRequestDto**: `{symbol, condition(ABOVE|BELOW), targetPrice, channel(IN_APP|USER)}`
- **AlertResponseDto**: `{id, symbol, condition, targetPrice, status(ACTIVE|TRIGGERED|CANCELLED), channel, createdAt}`
- **MAFA-agents usage**: `alert_tools.create_alert()`, `alert_tools.get_alerts()`, `alert_tools.delete_alert()`
- **Consumed by**: Execution Agent, General Agent, Portfolio Manager Agent

### 9. Bulk Stock Prices
- **Endpoint**: `POST /bulkstockprice` body: `{symbols: [...]}`
- **Response**: `ApiResponse { data: List<StockPriceDto> }` where `StockPriceDto: {symbol, close, date, open, high, low, volume}`
- **MAFA-agents usage**: `profile_tools.get_bulk_stock_prices()`
- **Consumed by**: General Agent, Portfolio Manager Agent, Market Research Agent, Investment Strategy tools (internal helpers)

---

## ⚠️ AVAILABLE BUT NOT CONSUMED

### 8. Chat History
- **Endpoint**: `GET /chats?limit=Y&page=Z`
- **Note**: Only supports `limit` and `page` params. Does NOT support filtering by agent type.
- **Status**: Available in MAFA-B. Not consumed by MAFA-agents tools (chat history is managed via Supabase vector memory currently).

---

## ⏳ NOT YET IMPLEMENTED IN MAFA-B

### 5. Portfolio Analytics Endpoint `GET /portfolio/analytics`

**Why it matters:** Risk analytics (beta, Sharpe, VaR, max drawdown) are currently approximated by MAFA-agents in Python. A server-side endpoint with access to full price history and position data would be more accurate and performant.

**What is needed:**
```
GET /portfolio/analytics
→ {
    totalValue, cashBalance,
    beta, sharpe, volatility30d,
    maxDrawdown, var95,
    diversificationScore,
    topHolding: { symbol, pct }
  }
```

**Current workaround:** `investment_strategy_tools.py` computes risk metrics in-process using dashboard + transaction data.

### 6. Order Status / Pending Orders `GET /orders`

**Why it matters:** MAFA-B's `POST /execute/buy` and `POST /execute/sell` execute instantly and return a `TransactionDto`, but there is no concept of pending/limit orders, order status, or order cancellation.

**What is needed (if limit orders are planned):**
```
GET  /orders               → List<OrderDto>
GET  /orders/{id}          → OrderDto
POST /orders/{id}/cancel
```

**Current workaround:** `GET /transactions` provides completed trade history. Only market orders are supported.

---

## Summary Table

| # | Service | Status | MAFA-agents Tool |
|---|---------|--------|-----------------|
| 1 | Sector-per-stock mapping | ✅ Done | `get_company_by_symbol`, `get_companies_by_symbols` |
| 2 | Portfolio value history | ✅ Done | `get_portfolio_history` |
| 3 | Watchlist CRUD | ✅ Done | `get_watchlist`, `add_to_watchlist`, `remove_from_watchlist` |
| 4 | Strategy persistence | ✅ Done | `get_active_strategy`, `save_strategy`, `update_strategy`, `get_strategy_history` |
| 5 | Portfolio analytics | ⏳ Pending | *approximated in investment_strategy_tools* |
| 6 | Order status / pending | ⏳ Pending | *GET /transactions used as workaround* |
| 7 | Alerts / notifications | ✅ Done | `create_alert`, `get_alerts`, `delete_alert` |
| 8 | Chat history | ⚠️ Available | *not consumed (uses Supabase memory instead)* |
| 9 | Bulk stock prices | ✅ Done | `get_bulk_stock_prices` |
