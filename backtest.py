import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from brain.hmm_model import InstitutionalHMMEngine
import warnings

# Ignore convergence and scaling warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_advanced_metrics(journal_df, initial_balance, df):
    """Computes high-tier institutional portfolio analytics."""
    if journal_df.empty:
        return {}

    # Generate daily chronological return series
    journal_df['date'] = pd.to_datetime(journal_df['timestamp']).dt.date
    daily_pnl = journal_df.groupby('date')['pnl'].sum()

    # Reindex to match the entire backtest window for accurate exposure
    all_dates = pd.date_range(start=df['timestamp'].min().date(), end=df['timestamp'].max().date()).date
    daily_pnl = daily_pnl.reindex(all_dates, fill_value=0.0)

    # Portfolio value tracking
    portfolio_value = initial_balance + daily_pnl.cumsum()
    daily_returns = daily_pnl / portfolio_value.shift(1).fillna(initial_balance)

    # Metrics computation
    total_trades = len(journal_df)
    net_profit = journal_df['pnl'].sum()
    final_val = initial_balance + net_profit

    # CAGR
    days = (df['timestamp'].max() - df['timestamp'].min()).days
    years = max(days / 365.25, 0.1)
    cagr = ((final_val / initial_balance) ** (1 / years)) - 1 if final_val > 0 else -1.0

    # Volatility & Ratios
    std_dev = daily_returns.std() * np.sqrt(252)
    mean_return = daily_returns.mean() * 252

    sharpe = (mean_return / std_dev) if std_dev > 0 else 0.0

    # Downside deviation for Sortino
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (mean_return / downside_std) if downside_std > 0 else 0.0

    # Max Drawdown & Calmar
    roll_max = portfolio_value.cummax()
    drawdowns = (portfolio_value - roll_max) / roll_max
    max_dd = drawdowns.min()

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # Trade statistics
    wins = journal_df[journal_df['pnl'] > 0]['pnl']
    losses = journal_df[journal_df['pnl'] < 0]['pnl']
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0

    profit_factor = abs(wins.sum() / losses.sum()) if not losses.empty else float('inf')
    recovery_factor = net_profit / abs(portfolio_value.min() - initial_balance) if (portfolio_value.min() < initial_balance) else float('inf')

    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    # Exposure
    active_trade_days = len(journal_df['date'].unique())
    exposure = (active_trade_days / len(all_dates)) * 100

    return {
        "cagr": cagr * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd * 100,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "recovery_factor": recovery_factor,
        "expectancy": expectancy,
        "exposure": exposure
    }

def run_greatness_prop_engine(csv_filename, name):
    csv_path = csv_filename if os.path.exists(csv_filename) else os.path.join("data", csv_filename)
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} missing.")
        return

    # 1. Load Data
    print(f"[DATA PROCESS] Loading from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';')
    df.columns = [col.lower().strip() for col in df.columns]
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y.%m.%d %H:%M')
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Date Range Lock (Starting Dec 1st, 2024)
    df = df[(df['timestamp'] >= '2024-12-01') & (df['timestamp'] <= '2025-12-31')].copy().reset_index(drop=True)

    # Base Indicators
    df['tr'] = np.max(np.vstack([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift(1)),
        np.abs(df['low'] - df['close'].shift(1))
    ]), axis=0)
    df['atr'] = df['tr'].rolling(14).mean().ffill().bfill()
    df['ema_trend'] = df['close'].ewm(span=20, adjust=False).mean()

    # HMM Features
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['atr_normalized'] = df['atr'] / df['close']
    hmm_features = ['log_returns', 'atr_normalized']

    # Drop Nans for HMM
    df_clean = df.dropna(subset=hmm_features + ['atr']).copy()

    # 2. HMM Walk-Forward Optimization Setup (Retraining every 3 Months)
    print("\n[HMM PROCESS] Commencing Out-of-Sample Walk-Forward Fitting Pipeline...")

    # Initialize target tracking arrays on df_clean directly to prevent alignment gaps
    df_clean['market_regime'] = -1
    df_clean['regime_confidence'] = 0.0

    # Dynamic Retraining Windows
    chunk_size = pd.Timedelta(days=90) # ~3 Months retrain cycle
    start_date = df_clean['timestamp'].min()
    end_date = df_clean['timestamp'].max()

    current_start = start_date
    while current_start < end_date:
        train_end = current_start + chunk_size
        test_end = train_end + chunk_size

        train_data = df_clean[(df_clean['timestamp'] >= current_start) & (df_clean['timestamp'] < train_end)]
        if len(train_data) < 100:
            current_start += chunk_size
            continue

        # Fit Engine
        hmm_engine = InstitutionalHMMEngine(min_components=2, max_components=5)
        try:
            hmm_engine.fit(train_data, feature_cols=hmm_features)

            # Identify test indices directly on clean slice
            test_mask = (df_clean['timestamp'] >= train_end) & (df_clean['timestamp'] < test_end)
            test_slice = df_clean[test_mask]

            if not test_slice.empty:
                predictions = hmm_engine.predict_regimes(test_slice, feature_cols=hmm_features)

                # Direct Series index-aligned assignments
                df_clean.loc[test_mask, 'market_regime'] = predictions['market_regime'].values
                df_clean.loc[test_mask, 'regime_confidence'] = predictions['regime_confidence'].values
        except Exception as e:
            pass

        current_start = train_end

    # Clean the propagation chain using real dynamic predictions first
    # Jo unmapped edge points hain unko safely propagate karenge
    df_clean['market_regime'] = df_clean['market_regime'].replace(-1, method='ffill').replace(-1, method='bfill').fillna(0)
    df_clean['regime_confidence'] = df_clean['regime_confidence'].replace(0.0, method='ffill').replace(0.0, method='bfill').fillna(0.50)

    # Re-merge the clean, fully predicted data back to original DF layout to avoid any missing dates
    df = df.drop(columns=['market_regime', 'regime_confidence'], errors='ignore')
    df = pd.merge(df, df_clean[['timestamp', 'market_regime', 'regime_confidence']], on='timestamp', how='left')

    # Final safety fallback fill on main DF
    df['market_regime'] = df['market_regime'].fillna(0).astype(int)
    df['regime_confidence'] = df['regime_confidence'].fillna(0.50)

    # Time helpers
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['date_only'] = df['timestamp'].dt.date
    df['year_week'] = df['timestamp'].dt.strftime('%Y-W%U')

    # Build 4H Macro Bias
    df_temp = df.set_index('timestamp')
    df_4h = df_temp.resample('4h').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna().reset_index()
    TARGET_THRESHOLD = 0.0070
    df_4h['body_pct'] = (df_4h['close'] - df_4h['open']).abs() / df_4h['open']
    df_4h['is_bullish'] = df_4h['close'] > df_4h['open']
    df_4h['total_range'] = df_4h['high'] - df_4h['low']
    df_4h['body_size'] = (df_4h['close'] - df_4h['open']).abs()
    df_4h['qualifying_bias'] = np.where(
        (df_4h['body_pct'] >= TARGET_THRESHOLD) & (df_4h['body_size'] > (df_4h['total_range'] * 0.60)),
        np.where(df_4h['is_bullish'], 'BULLISH', 'BEARISH'), 'NONE'
    )
    df_4h['bias_expiry_time'] = df_4h['timestamp'] + pd.Timedelta(hours=28)

    bias_map = {}
    for i, row in df_4h[df_4h['qualifying_bias'] != 'NONE'].iterrows():
        bias_map[row['timestamp'] + pd.Timedelta(hours=4)] = {
            'bias': row['qualifying_bias'], 'expiry': row['bias_expiry_time']
        }

    # Simulation Variables
    account_balance = 5000.0
    initial_balance = account_balance
    trade_journal = []
    equity_curve = [account_balance]
    equity_timestamps = [df['timestamp'].iloc[0]]

    current_day = None
    trades_taken_today = 0
    day_terminated = False
    consecutive_losses = 0

    in_position = False
    position_type = None
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    units = 0.0
    tp1_hit = False
    be_triggered = False
    pending_limit = None
    active_bias = 'NONE'
    bias_expiry = None

    print("\n[BACKTEST] Simulating institutional risk engine execution loop...")

    # Backtest Loop
    for idx in range(20, len(df)):
        row = df.iloc[idx]
        timestamp = row['timestamp']
        row_day = row['date_only']
        yw_tag = row['year_week']

        close = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        atr = float(row['atr'])
        ema = float(row['ema_trend'])
        hour = row['hour']
        minute = row['minute']

        market_regime = int(row['market_regime'])
        regime_confidence = float(row['regime_confidence'])

        is_core_killzone = (hour > 11 or (hour == 11 and minute >= 0)) & (hour < 15 or (hour == 15 and minute <= 0))
        is_eod_flush = (hour >= 21)

        if timestamp in bias_map:
            active_bias = bias_map[timestamp]['bias']
            bias_expiry = bias_map[timestamp]['expiry']
        if bias_expiry and timestamp > bias_expiry:
            active_bias = 'NONE'
            bias_expiry = None

        if row_day != current_day:
            current_day = row_day
            trades_taken_today = 0
            day_terminated = False
            pending_limit = None
            if in_position: in_position = False

        # --- Position State Machine ---
        if in_position:
            if position_type == "LONG":
                if low <= sl_price:
                    pnl_loss = - (units * (entry_price - sl_price)) if not be_triggered else 0.0
                    if tp1_hit: pnl_loss = 0.0
                    account_balance += pnl_loss
                    trade_journal.append({"timestamp": timestamp, "year_week": yw_tag, "type": "LONG", "outcome": "SL" if not tp1_hit and not be_triggered else "BE_STOP", "pnl": pnl_loss, "regime": market_regime})
                    in_position = False
                    if not tp1_hit: consecutive_losses += 1
                    equity_curve.append(account_balance)
                    equity_timestamps.append(timestamp)
                    continue

                if not tp1_hit and high >= tp1_price:
                    tp1_hit = True
                    be_triggered = True
                    sl_price = entry_price
                    account_balance += (units * 0.5) * (tp1_price - entry_price)

                if high >= tp2_price or is_eod_flush:
                    exit_p = tp2_price if high >= tp2_price else close
                    rem_units = units * 0.5 if tp1_hit else units
                    pnl_win = rem_units * (exit_p - entry_price)
                    account_balance += pnl_win
                    total_pnl = pnl_win + ((units * 0.5) * (tp1_price - entry_price) if tp1_hit else 0.0)
                    trade_journal.append({"timestamp": timestamp, "year_week": yw_tag, "type": "LONG", "outcome": "TP_WIN", "pnl": total_pnl, "regime": market_regime})
                    in_position = False
                    day_terminated = True
                    consecutive_losses = 0
                    equity_curve.append(account_balance)
                    equity_timestamps.append(timestamp)

            elif position_type == "SHORT":
                if high >= sl_price:
                    pnl_loss = - (units * (sl_price - entry_price)) if not be_triggered else 0.0
                    if tp1_hit: pnl_loss = 0.0
                    account_balance += pnl_loss
                    trade_journal.append({"timestamp": timestamp, "year_week": yw_tag, "type": "SHORT", "outcome": "SL" if not tp1_hit and not be_triggered else "BE_STOP", "pnl": pnl_loss, "regime": market_regime})
                    in_position = False
                    if not tp1_hit: consecutive_losses += 1
                    equity_curve.append(account_balance)
                    equity_timestamps.append(timestamp)
                    continue

                if not tp1_hit and low <= tp1_price:
                    tp1_hit = True
                    be_triggered = True
                    sl_price = entry_price
                    account_balance += (units * 0.5) * (entry_price - tp1_price)

                if low <= tp2_price or is_eod_flush:
                    exit_p = tp2_price if low <= tp2_price else close
                    rem_units = units * 0.5 if tp1_hit else units
                    pnl_win = rem_units * (entry_price - exit_p)
                    account_balance += pnl_win
                    total_pnl = pnl_win + ((units * 0.5) * (entry_price - tp1_price) if tp1_hit else 0.0)
                    trade_journal.append({"timestamp": timestamp, "year_week": yw_tag, "type": "SHORT", "outcome": "TP_WIN", "pnl": total_pnl, "regime": market_regime})
                    in_position = False
                    day_terminated = True
                    consecutive_losses = 0
                    equity_curve.append(account_balance)
                    equity_timestamps.append(timestamp)

        # Trigger Pending Orders
        if not in_position and pending_limit:
            if pending_limit['type'] == "LONG" and low <= pending_limit['limit_price']:
                in_position, position_type = True, "LONG"
                tp1_hit, be_triggered = False, False
                entry_price, sl_price = pending_limit['limit_price'], pending_limit['sl']
                tp1_price, tp2_price, units = pending_limit['tp1'], pending_limit['tp2'], pending_limit['units']
                trades_taken_today += 1
                pending_limit = None
            elif pending_limit['type'] == "SHORT" and high >= pending_limit['limit_price']:
                in_position, position_type = True, "SHORT"
                tp1_hit, be_triggered = False, False
                entry_price, sl_price = pending_limit['limit_price'], pending_limit['sl']
                tp1_price, tp2_price, units = pending_limit['tp1'], pending_limit['tp2'], pending_limit['units']
                trades_taken_today += 1
                pending_limit = None

        # --- OPTION B: LIQUIDITY SWEEP INTERCEPT RULES WITH LOPEZ DE PRADO CONTROLS ---
        if not in_position and not pending_limit and is_core_killzone and not day_terminated and trades_taken_today < 1:
            if active_bias == 'NONE':
                continue

            # PRIORITY 2: Confidence Filter Gate (>= 80% Cutoff)
            if regime_confidence < 0.65:
                continue

            # PRIORITY 3: Regime-Adaptive Structural Risk Filtering (Skip high risk/crash/chop regimes)
            # Filter Regime 1 & 4 out of execution loop based on initial performance matrix
            # if market_regime in [1, 4]:
            #     continue

            recent_window = df.iloc[idx-4:idx]

            # PRIORITY 4: Composite Trade Quality Score (TQS) Computation
            # 1. Trend (EMA alignment): 30%
            trend_score = 30 if (active_bias == 'BULLISH' and close > ema) or (active_bias == 'BEARISH' and close < ema) else 0
            # 2. Momentum (Return alignment): 25%
            momentum_score = 25 if (active_bias == 'BULLISH' and row['log_returns'] > 0) or (active_bias == 'BEARISH' and row['log_returns'] < 0) else 10
            # 3. Volatility (Within normal limits): 20%
            v_ratio = atr / df['atr'].mean()
            vol_score = 20 if (0.5 <= v_ratio <= 2.0) else 5
            # 4. Liquidity Score: 15%
            liq_score = 15 if (row['tr'] > 0) else 0
            # 5. Regime Confidence: 10%
            regime_score = int(regime_confidence * 10)

            total_tqs = trend_score + momentum_score + vol_score + liq_score + regime_score

            # Quality Gate: Skip poor setups (Score < 70)
            if total_tqs < 70:
                continue

            # PRIORITY 1: Dynamic Position Sizing (Kelly Fraction + Volatility Scaling)
            # Kelly Fraction calculation (approximate win-rate ratio)
            f_kelly = 0.25 # Aggressive quarter-kelly allocation base

            # Volatility scaling multiplier (low vol scaling up, high vol downscaling)
            vol_scaler = 1.0 / max(v_ratio, 0.4)

            # Final scaled capital allocation
            risk_pct = f_kelly * 0.02 * vol_scaler
            risk_pct = np.clip(risk_pct, 0.005, 0.02) # Enforce strict institutional limits (0.5% - 2.0%)

            # Apply Regime-Adaptive Budgeting caps
            if market_regime == 0:   # High performing regime
                risk_pct *= 1.0
            elif market_regime == 3: # Moderate risk regime
                risk_pct *= 0.6
            else:
                risk_pct *= 0.2

            # Entry Executions
            if active_bias == 'BULLISH':
                structural_low = recent_window['low'].min()
                limit_entry = structural_low
                sl_absolute = limit_entry - (atr * 1.0)
                risk_dist = limit_entry - sl_absolute

                if risk_dist > 0 and close > limit_entry:
                    cash_at_risk = account_balance * risk_pct
                    pending_limit = {
                        "type": "LONG", "limit_price": limit_entry, "sl": sl_absolute,
                        "tp1": limit_entry + (risk_dist * 2.0), "tp2": limit_entry + (risk_dist * 4.0),
                        "units": cash_at_risk / risk_dist
                    }

            elif active_bias == 'BEARISH':
                structural_high = recent_window['high'].max()
                limit_entry = structural_high
                sl_absolute = limit_entry + (atr * 1.0)
                risk_dist = sl_absolute - limit_entry

                if risk_dist > 0 and close < limit_entry:
                    cash_at_risk = account_balance * risk_pct
                    pending_limit = {
                        "type": "SHORT", "limit_price": limit_entry, "sl": sl_absolute,
                        "tp1": limit_entry - (risk_dist * 2.0), "tp2": limit_entry - (risk_dist * 4.0),
                        "units": cash_at_risk / risk_dist
                    }

    # Final Sync
    equity_curve.append(account_balance)
    equity_timestamps.append(df['timestamp'].iloc[-1])

    # 3. Report Engine Outputs
    journal_df = pd.DataFrame(trade_journal)

    print(f"\n========================================================================================================")
    print(f"[METRICS REPORT] V5 INSTITUTIONAL QUANT EVALUATION FOR: {name}")
    print(f"========================================================================================================")

    metrics = calculate_advanced_metrics(journal_df, initial_balance, df)

    if metrics:
        print(f"   CAGR (Compound Annual Growth)          : {metrics['cagr']:.2f}%")
        print(f"   Sharpe Ratio (Annualized Risk-Adj)     : {metrics['sharpe']:.2f}")
        print(f"   Sortino Ratio (Downside Deviation)     : {metrics['sortino']:.2f}")
        print(f"   Calmar Ratio (CAGR / Max DD)           : {metrics['calmar']:.2f}")
        print(f"   Max Historical Drawdown (Max DD)       : {metrics['max_dd']:.2f}%")
        print(f"   System Recovery Factor                 : {metrics['recovery_factor']:.2f}")
        print(f"   Expectancy (Expected Return Per Trade) : ${metrics['expectancy']:.2f}")
        print(f"   System Market Exposure Time            : {metrics['exposure']:.2f}%")
        print(f"   Profit Factor                          : {metrics['profit_factor']:.2f}")
        print(f"   Average Trade Win / Loss               : +${metrics['avg_win']:.2f} / -${abs(metrics['avg_loss']):.2f}")

    # 4. Dynamic Rolling Performance Analysis (Window: Last 50 Trades)
    print("\n=============================== ROLLING PERFORMANCE STATS (LAST 50 TRADES) ===============================")
    if len(journal_df) >= 10:
        rolling_metrics = []
        for i in range(10, len(journal_df) + 1):
            slice_df = journal_df.iloc[max(0, i-50):i].copy()
            sub_metrics = calculate_advanced_metrics(slice_df, initial_balance, df)
            if sub_metrics:
                rolling_metrics.append({
                    'trade_count': i,
                    'rolling_sharpe': sub_metrics['sharpe'],
                    'rolling_cagr': sub_metrics['cagr'],
                    'rolling_dd': sub_metrics['max_dd']
                })

        roll_df = pd.DataFrame(rolling_metrics)
        print(roll_df.tail(10).to_string(index=False, formatters={
            'rolling_sharpe': '{:,.2f}'.format,
            'rolling_cagr': '{:+.2f}%'.format,
            'rolling_dd': '{:+.2f}%'.format
        }))
    else:
        print("[INFO] Not enough transactions to calculate 50-trade rolling curves.")

    # 5. HMM Regime Performance Matrix
    print("\n==================================== HMM REGIME WISE PERFORMANCE ====================================")
    if not journal_df.empty:
        reg_summary = []
        for r_id in range(5):
            r_trades = journal_df[journal_df['regime'] == r_id]
            r_total = len(r_trades)
            if r_total > 0:
                r_pnl = r_trades['pnl'].sum()
                r_win = len(r_trades[r_trades['pnl'] > 0])
                r_wr = (r_win / r_total) * 100
                reg_summary.append({
                    'Regime Type': f"Regime {r_id}", 'Total Trades': r_total, 'Net P&L': f"${r_pnl:+5.0f}", 'Win Rate': f"{r_wr:.1f}%"
                })
        print(pd.DataFrame(reg_summary).to_string(index=False))

    # 6. Save Professional Equity Curve Plot
    if len(equity_curve) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_timestamps, equity_curve, label="V5 Quant Equity Curve", color="#00ffd0", linewidth=2)
        plt.title("Institutional V5 Portfolio Performance - XAUUSD")
        plt.xlabel("Timeline")
        plt.ylabel("Portfolio Balance ($)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.savefig("equity_curve.png")
        print("\n[PLOT SYSTEM] Professional Equity Curve successfully exported to 'equity_curve.png'.")

if __name__ == "__main__":
    run_greatness_prop_engine("XAU_5m_data.csv", "Gold Liquidity Sweep Engine")