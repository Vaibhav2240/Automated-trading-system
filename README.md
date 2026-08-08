# 🏛️ Institutional V5 Quant Trading Engine (XAUUSD Framework)

A high-frequency algorithmic backtesting and regime-adaptive execution framework built for **XAUUSD (Gold)**. Incorporating **Marcos López de Prado's** quantitative architecture, **Gaussian Hidden Markov Models (HMM)**, dynamic risk budgeting, and out-of-sample Walk-Forward Optimization.

---

## 📌 Executive Summary

Unlike conventional rule-based backtesters that overfit historical market data, this engine dynamically shifts position parameters based on market microstructures and hidden states. By continuously training an out-of-sample Gaussian HMM and enforcing strict Trade Quality Scores (TQS), the system filters out market noise and optimizes for **Sharpe, Sortino, and Calmar Ratios** rather than naive nominal returns.

---

## ⚡ Key Architectural Features

### 1. Marcos López de Prado Framework Integration
* **Walk-Forward Out-Of-Sample Retraining:** Retrains market regime detectors on rolling 90-day execution blocks to systematically defeat **Look-Ahead Bias** and data leakage.
* **Triple Barrier Method Principles:** Tracks dynamic stop-losses ($1.0 \times \text{ATR}$), profit targets ($2.0 \times \text{RR}$ and $4.0 \times \text{RR}$ scale-out), and time-decay expiries.

### 2. Adaptive Risk Budgeting & Dynamic Sizing
Constant position sizing is replaced with a 3-tier risk allocation formula:
$$\text{Risk \%} = \text{Kelly Base } (0.25) \times \text{Volatility Scaler } \left(\frac{1.0}{\text{v\_ratio}}\right) \times \text{Regime Multiplier}$$

* **Kelly Criterion Fractioning:** Enforces a $0.25$ Quarter-Kelly baseline capped within strict institutional boundaries ($0.5\%$ to $2.0\%$ max risk per trade).
* **Volatility Scaling:** Automatically downscales risk during ATR volatility spikes and scales up during low-volatility compression setups.
* **Regime Budget Allocation:** Shifts capital dynamically based on state classification ($100\%$ for high-performing states, $60\%$ for transition states, $20\%$ for high-noise states).

### 3. Composite Trade Quality Scoring (TQS)
Before firing limit orders during core London/NY killzones, every signal must pass a composite scoring gate ($\text{TQS} \ge 70$ threshold):
* **Trend Alignment (EMA 20):** $30\%$
* **Momentum Vector (Log Returns):** $25\%$
* **Volatility Normalization (ATR Ratio):** $20\%$
* **Liquidity Proxy (True Range):** $15\%$
* **Regime Confidence Score:** $10\%$

---

## 📊 Performance Analytics Engine

The engine generates deep mathematical, risk-adjusted performance reports including:
* **Sharpe, Sortino, and Calmar Ratios**
* **Max Historical Drawdown (Peak-to-Trough)**
* **Compound Annual Growth Rate (CAGR)**
* **Profit Factor & System Expectancy**
* **Rolling 50-Trade Performance Tracking**
* **Regime-Wise Performance Breakdown Matrix**

---

## 🛠️ System Architecture & Execution Flow

```text
[ Raw Tick/OHLC Data ] 
          │
          ▼
[ Indicator Calculations (ATR, EMA, Log-Returns) ]
          │
          ▼
[ Walk-Forward 90-Day Rolling Window Fit (Gaussian HMM) ]
          │
          ▼
[ Regime Detection & Probability Confidence Gate (≥ 80%) ]
          │
          ▼
[ Liquidity Sweep Signal + TQS Composite Gate (≥ 70) ]
          │
          ▼
[ Volatility-Scaled / Kelly-Budgeted Order Execution ]
          │
          ▼
[ State Machine Tracking (TP1 Scale-Out -> BE -> TP2 / SL) ]
          │
          ▼
[ Advanced Institutional Metrics Output & Equity Curve Plotting ]
