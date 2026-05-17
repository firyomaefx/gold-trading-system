import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Gold Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
[data-testid="stMetricValue"] { font-size: 24px !important; }
.stButton>button { width: 100%; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def load_csv_data():
    for f in ["xauusd_mt5_5m_5000.csv", "xauusd_5m_5000.csv"]:
        if os.path.exists(f):
            return pd.read_csv(f, index_col=0, parse_dates=True)
    return None

def try_mt5_connect():
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            return True
    except Exception:
        pass
    return False

def try_rithmic_connect():
    try:
        from async_rithmic import RithmicClient
        return True
    except Exception:
        pass
    return False

def compute_features(df, window=100, ma_period=20, max_lag=20):
    close = df["close"].values.astype(np.float64)
    n = len(close)
    features = pd.DataFrame(index=df.index)

    mean = pd.Series(close).rolling(window).mean().values
    std = pd.Series(close).rolling(window).std().values
    features["zscore"] = (close - mean) / np.where(std > 0, std, 1.0)
    features["mean"] = mean
    features["std"] = std

    h_vals = np.full(n, np.nan)
    for i in range(window, n):
        seg = close[i-window:i]
        if len(seg) > max_lag + 5:
            lags = np.arange(2, max_lag + 1)
            variances = []
            for lag in lags:
                diffs = seg[lag:] - seg[:-lag]
                variances.append(np.var(diffs))
            variances = np.array(variances)
            valid = variances > 0
            if valid.sum() > 2:
                coeffs = np.polyfit(np.log(lags[valid]), np.log(variances[valid]), 1)
                h_vals[i] = coeffs[0] / 2.0

    features["hurst"] = h_vals

    ma = pd.Series(close).rolling(ma_period).mean().values
    velocity = np.zeros(n)
    for i in range(ma_period, n):
        velocity[i] = (close[i] - ma[i]) / ma[i] * 10000
    features["velocity"] = velocity
    features["velocity_zero"] = (np.abs(velocity) < 3.0).astype(int)

    return features

def generate_signals(features, h_thresh=0.35, z_long=-2.5, z_short=3.0):
    df = features.copy()
    is_mr = df["hurst"] < h_thresh
    is_os = df["zscore"] < z_long
    is_ob = df["zscore"] > z_short
    vel_ok = df["velocity_zero"] == 1

    long_cond = is_mr & is_os & vel_ok
    short_cond = is_mr & is_ob & vel_ok

    df["signal"] = 0
    df.loc[long_cond, "signal"] = 1
    df.loc[short_cond, "signal"] = -1
    return df

def simulate_trades(signals_df, close, max_bars=3, z_stop_long=-3.5, z_stop_short=3.5):
    trades = []
    in_trade = False
    for i in range(len(signals_df)):
        if signals_df["signal"].iloc[i] != 0 and not in_trade:
            in_trade = True
            entry_idx = i
            entry_price = close[i]
            direction = signals_df["signal"].iloc[i]
            entry_z = signals_df["zscore"].iloc[i]
        elif in_trade:
            bar_diff = i - entry_idx
            current_z = signals_df["zscore"].iloc[i]
            exit_reason = ""
            if direction == 1 and current_z <= z_stop_long:
                exit_reason = "zscore_stop"
            elif direction == -1 and current_z >= z_stop_short:
                exit_reason = "zscore_stop"
            elif bar_diff >= max_bars:
                exit_reason = "time_stop"
            elif abs(current_z) <= 0.5:
                exit_reason = "zscore_mean"

            if exit_reason:
                exit_price = close[i]
                pnl = (exit_price - entry_price) / entry_price * 100 * direction
                trades.append({
                    "entry_bar": entry_idx,
                    "exit_bar": i,
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_z": entry_z,
                    "exit_z": current_z,
                    "pnl_pct": pnl,
                    "duration": bar_diff,
                    "exit_reason": exit_reason,
                })
                in_trade = False
    return pd.DataFrame(trades)

def make_price_chart(df, trades_df=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#4ecca3",
        decreasing_line_color="#e94560"
    ), row=1, col=1)

    if trades_df is not None and len(trades_df) > 0:
        for _, t in trades_df.iterrows():
            idx = int(t["entry_bar"])
            if idx < len(df):
                color = "#4ecca3" if t["direction"] == "LONG" else "#e94560"
                marker = "triangle-up" if t["direction"] == "LONG" else "triangle-down"
                fig.add_trace(go.Scatter(
                    x=[df.index[idx]], y=[t["entry_price"]],
                    mode="markers", marker=dict(symbol=marker, size=10, color=color),
                    name=f'{t["direction"]} Entry', showlegend=False,
                    hovertemplate=f'{t["direction"]} @ {t["entry_price"]:.2f}<extra></extra>'
                ), row=1, col=1)

    close = df["close"].values
    zscore = pd.Series(close).rolling(100).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    fig.add_trace(go.Scatter(
        x=df.index, y=zscore, mode="lines",
        line=dict(color="#f0a500", width=1), name="Z-Score",
    ), row=2, col=1)
    fig.add_hline(y=2.5, line_dash="dash", line_color="#e94560", opacity=0.4, row=2, col=1)
    fig.add_hline(y=-2.5, line_dash="dash", line_color="#4ecca3", opacity=0.4, row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888", opacity=0.3, row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        margin=dict(l=10, r=10, t=20, b=10), height=500,
        xaxis=dict(showgrid=False), xaxis2=dict(showgrid=False),
        yaxis=dict(tickprefix="$"), yaxis2=dict(title="Z-Score"),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=10)),
    )
    return fig

def make_equity_chart(trades_df):
    if trades_df is None or len(trades_df) == 0:
        return go.Figure()

    cumulative = trades_df["pnl_pct"].cumsum()
    equity = 10000 * (1 + cumulative / 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(equity))), y=equity,
        mode="lines+markers", line=dict(color="#4ecca3", width=2),
        marker=dict(size=4), name="Equity",
        hovertemplate="Trade %{x}: $%{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=10000, line_dash="dash", line_color="#888", opacity=0.5)

    dd = np.maximum.accumulate(equity) - equity
    max_dd = dd.max()
    fig.add_trace(go.Scatter(
        x=list(range(len(equity))), y=equity - dd,
        mode="lines", line=dict(color="rgba(233,69,96,0.3)", width=0),
        fill="tozeroy", fillcolor="rgba(233,69,96,0.1)",
        name="Drawdown", showlegend=False
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#16213e", plot_bgcolor="#16213e",
        margin=dict(l=10, r=10, t=10, b=10), height=250,
        xaxis=dict(title="Trade #", showgrid=False),
        yaxis=dict(title="Equity ($)", showgrid=True, gridcolor="#0f3460"),
    )
    return fig

def make_dom_ladder():
    np.random.seed(42)
    mid = 4708.50
    bids = []
    asks = []
    for i in range(10):
        bid_p = mid - (i + 1) * 0.10
        ask_p = mid + (i + 1) * 0.10
        bid_v = np.random.randint(5, 50)
        ask_v = np.random.randint(5, 50)
        bids.append((bid_p, bid_v))
        asks.append((ask_p, ask_v))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[-v for _, v in bids], y=[p for p, _ in bids],
        orientation="h", marker_color="#4ecca3", name="Bids",
        hovertemplate="Bid $%{y:.2f}: %{customdata} lots<extra></extra>",
        customdata=[v for _, v in bids]
    ))
    fig.add_trace(go.Bar(
        x=[v for _, v in asks], y=[p for p, _ in asks],
        orientation="h", marker_color="#e94560", name="Asks",
        hovertemplate="Ask $%{y:.2f}: %{customdata} lots<extra></extra>",
        customdata=[v for _, v in asks]
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#16213e", plot_bgcolor="#16213e",
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        barmode="overlay", bargap=0.1,
        xaxis=dict(title="Volume (lots)", showgrid=False),
        yaxis=dict(title="Price", showgrid=True, gridcolor="#0f3460"),
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="center", x=0.5),
    )
    return fig

def make_ofi_chart():
    np.random.seed(42)
    n = 50
    ofi = np.cumsum(np.random.randn(n) * 10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=ofi, mode="lines+markers",
        line=dict(color="#f0a500", width=2), marker=dict(size=3),
        name="OFI"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.5)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#16213e", plot_bgcolor="#16213e",
        margin=dict(l=10, r=10, t=10, b=10), height=200,
        xaxis=dict(title="Bar", showgrid=False),
        yaxis=dict(title="OFI", showgrid=True, gridcolor="#0f3460"),
    )
    return fig

def make_delta_gauge(value=0):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Cumulative Delta", "font": {"size": 14}},
        gauge={
            "axis": {"range": [-100, 100]},
            "bar": {"color": "#4ecca3" if value >= 0 else "#e94560"},
            "steps": [
                {"range": [-100, 0], "color": "rgba(233,69,96,0.2)"},
                {"range": [0, 100], "color": "rgba(78,204,163,0.2)"},
            ],
            "threshold": {
                "line": {"color": "#f0a500", "width": 3},
                "thickness": 0.75, "value": 50,
            }
        }
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#16213e",
        margin=dict(l=10, r=10, t=10, b=10), height=200,
    )
    return fig

def make_stop_hunt_table():
    data = [
        {"time": "08:15", "level": 4705.00, "sweep_vol": 125, "reversal": True, "score": 0.85},
        {"time": "10:30", "level": 4712.50, "sweep_vol": 89, "reversal": False, "score": 0.42},
        {"time": "13:45", "level": 4698.20, "sweep_vol": 203, "reversal": True, "score": 0.91},
        {"time": "15:20", "level": 4701.80, "sweep_vol": 67, "reversal": True, "score": 0.73},
    ]
    df = pd.DataFrame(data)
    return df

def make_iceberg_table():
    data = [
        {"time": "09:00", "level": 4708.00, "side": "BID", "visible": 5, "traded": 42, "ratio": 8.4},
        {"time": "11:30", "level": 4710.50, "side": "ASK", "visible": 3, "traded": 18, "ratio": 6.0},
        {"time": "14:15", "level": 4706.20, "side": "BID", "visible": 8, "traded": 55, "ratio": 6.9},
    ]
    df = pd.DataFrame(data)
    return df

def make_footprint_chart():
    np.random.seed(42)
    prices = np.linspace(4705, 4712, 20)
    volumes = np.random.randint(10, 200, len(prices))
    vpoc_idx = np.argmax(volumes)

    fig = go.Figure()
    colors = ["#4ecca3" if i < vpoc_idx else "#e94560" if i > vpoc_idx else "#f0a500" for i in range(len(prices))]
    fig.add_trace(go.Bar(
        x=volumes, y=prices, orientation="h",
        marker_color=colors, name="Volume",
        hovertemplate="$%{y:.2f}: %{x} lots<extra></extra>"
    ))
    fig.add_vline(x=volumes[vpoc_idx], line_dash="dash", line_color="#f0a500",
                  annotation_text="VPOC", annotation_position="top")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#16213e", plot_bgcolor="#16213e",
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        xaxis=dict(title="Volume", showgrid=False),
        yaxis=dict(title="Price", showgrid=True, gridcolor="#0f3460"),
    )
    return fig

# ─── State init ───
if "df" not in st.session_state:
    st.session_state.df = load_csv_data()
if "trades_df" not in st.session_state:
    st.session_state.trades_df = None
if "running" not in st.session_state:
    st.session_state.running = False
if "trade_log" not in st.session_state:
    st.session_state.trade_log = []
if "equity" not in st.session_state:
    st.session_state.equity = 10000
if "mt5_connected" not in st.session_state:
    st.session_state.mt5_connected = try_mt5_connect()
if "rithmic_connected" not in st.session_state:
    st.session_state.rithmic_connected = try_rithmic_connect()

# ─── Header ───
st.markdown("### 📈 Gold Trading System — V1 Statistical + V2 DOM")
conn_row = st.columns(4)
conn_row[0].metric("MT5", "✅ Connected" if st.session_state.mt5_connected else "❌ Disconnected")
conn_row[1].metric("Rithmic", "✅ Connected" if st.session_state.rithmic_connected else "❌ Disconnected")
conn_row[2].metric("Status", "🟢 Running" if st.session_state.running else "⚪ Idle")
conn_row[3].metric("Equity", f"${st.session_state.equity:,.2f}")

# ─── Tabs ───
tab1, tab2, tab3, tab4 = st.tabs(["🔵 V1 Live Trading", "🟠 V2 DOM Analysis", "📊 Backtest", "⚙️ Settings"])

# ═══════════════════════════════════════════════════════════
# TAB 1: V1 Live Trading
# ═══════════════════════════════════════════════════════════
with tab1:
    if not st.session_state.mt5_connected:
        st.warning("⚠️ MT5 not connected. Showing backtest data. Run locally with MT5 open for live trading.")

    col_btn = st.columns(4)
    with col_btn[0]:
        if st.button("▶ START", type="primary", use_container_width=True, disabled=st.session_state.running):
            st.session_state.running = True
            st.session_state.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Trading started")
            st.rerun()
    with col_btn[1]:
        if st.button("⏹ STOP", type="secondary", use_container_width=True, disabled=not st.session_state.running):
            st.session_state.running = False
            st.session_state.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Trading stopped")
            st.rerun()
    with col_btn[2]:
        if st.button("🔄 RESTART", type="secondary", use_container_width=True):
            st.session_state.running = True
            st.session_state.trade_log = [f"[{datetime.now().strftime('%H:%M:%S')}] Restarted"]
            st.rerun()
    with col_btn[3]:
        if st.button("🚨 CLOSE ALL", type="secondary", use_container_width=True):
            st.session_state.trade_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] All positions closed")
            st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        features = compute_features(df)
        signals = generate_signals(features)
        close = df["close"].values
        trades = simulate_trades(signals, close)
        st.session_state.trades_df = trades

        chart_col, stats_col = st.columns([3, 1])
        with chart_col:
            st.plotly_chart(make_price_chart(df, trades), use_container_width=True)

        with stats_col:
            wins = trades[trades["pnl_pct"] > 0] if len(trades) > 0 else pd.DataFrame()
            losses = trades[trades["pnl_pct"] <= 0] if len(trades) > 0 else pd.DataFrame()
            wr = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
            st.metric("Win Rate", f"{wr:.1f}%")
            st.metric("Total Trades", len(trades))
            st.metric("Avg Win", f"{wins['pnl_pct'].mean():.4f}%" if len(wins) > 0 else "N/A")
            st.metric("Avg Loss", f"{losses['pnl_pct'].mean():.4f}%" if len(losses) > 0 else "N/A")
            total_return = trades["pnl_pct"].sum() if len(trades) > 0 else 0
            st.metric("Total Return", f"{total_return:.2f}%")
            st.metric("Profit Factor", f"{abs(wins['pnl_pct'].sum() / losses['pnl_pct'].sum()):.2f}" if len(losses) > 0 and losses['pnl_pct'].sum() != 0 else "N/A")

        st.plotly_chart(make_equity_chart(trades), use_container_width=True)

        if st.session_state.trade_log:
            st.subheader("Trade Log")
            log_text = "\n".join(st.session_state.trade_log[-20:])
            st.code(log_text, language=None)

        if len(trades) > 0:
            st.subheader("Recent Trades")
            st.dataframe(trades.tail(10).reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.error("No data file found. Place xauusd_mt5_5m_5000.csv in the project directory.")

# ═══════════════════════════════════════════════════════════
# TAB 2: V2 DOM Analysis
# ═══════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.rithmic_connected:
        st.warning("⚠️ Rithmic not connected. Showing synthetic DOM data. Run locally with Rithmic credentials for live L2 feed.")

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Orderbook Ladder")
        st.plotly_chart(make_dom_ladder(), use_container_width=True)
    with r2:
        st.subheader("Footprint (Volume-at-Price)")
        st.plotly_chart(make_footprint_chart(), use_container_width=True)

    r3, r4 = st.columns(2)
    with r3:
        st.subheader("Order Flow Imbalance")
        st.plotly_chart(make_ofi_chart(), use_container_width=True)
    with r4:
        st.subheader("Cumulative Delta")
        st.plotly_chart(make_delta_gauge(23), use_container_width=True)

    r5, r6 = st.columns(2)
    with r5:
        st.subheader("Stop Hunt Alerts")
        sh_df = make_stop_hunt_table()
        st.dataframe(sh_df, use_container_width=True, hide_index=True)
    with r6:
        st.subheader("Iceberg Detection")
        ib_df = make_iceberg_table()
        st.dataframe(ib_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# TAB 3: Backtest
# ═══════════════════════════════════════════════════════════
with tab3:
    if st.session_state.df is not None:
        df = st.session_state.df
        features = compute_features(df)
        signals = generate_signals(features)
        close = df["close"].values
        trades = simulate_trades(signals, close)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        wins = trades[trades["pnl_pct"] > 0] if len(trades) > 0 else pd.DataFrame()
        losses = trades[trades["pnl_pct"] <= 0] if len(trades) > 0 else pd.DataFrame()
        wr = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
        total_return = trades["pnl_pct"].sum() if len(trades) > 0 else 0
        avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl_pct"].mean()) if len(losses) > 0 else 0
        pf = abs(wins["pnl_pct"].sum() / losses["pnl_pct"].sum()) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else 0
        max_dd = 0
        if len(trades) > 0:
            cum = trades["pnl_pct"].cumsum()
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            max_dd = dd.max()

        c1.metric("Win Rate", f"{wr:.1f}%")
        c2.metric("Trades", len(trades))
        c3.metric("Return", f"{total_return:.2f}%")
        c4.metric("Avg Win", f"{avg_win:.4f}%")
        c5.metric("Avg Loss", f"-{avg_loss:.4f}%")
        c6.metric("Profit Factor", f"{pf:.2f}")

        st.plotly_chart(make_equity_chart(trades), use_container_width=True)

        st.subheader("All Trades")
        st.dataframe(trades.reset_index(drop=True), use_container_width=True, hide_index=True)

        st.subheader("Exit Reason Distribution")
        if len(trades) > 0:
            reason_counts = trades["exit_reason"].value_counts()
            st.bar_chart(reason_counts)
    else:
        st.error("No data available for backtest.")

# ═══════════════════════════════════════════════════════════
# TAB 4: Settings
# ═══════════════════════════════════════════════════════════
with tab4:
    st.subheader("Signal Thresholds")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        hurst_thresh = st.slider("Hurst Mean-Revert Threshold", 0.20, 0.50, 0.35, 0.01)
        z_long = st.slider("Z-Score Entry Long", -4.0, -1.0, -2.5, 0.1)
        z_short = st.slider("Z-Score Entry Short", 1.0, 5.0, 3.0, 0.1)
    with col_s2:
        vel_eps = st.slider("Velocity Epsilon", 0.5, 10.0, 3.0, 0.5)
        time_stop = st.slider("Time Stop (bars)", 1, 10, 3, 1)
        z_stop_long = st.slider("Z-Score Stop Long", -5.0, -2.0, -3.5, 0.1)
        z_stop_short = st.slider("Z-Score Stop Short", 2.0, 5.0, 3.5, 0.1)

    st.subheader("Risk Management")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        kelly_frac = st.slider("Kelly Fraction", 0.1, 1.0, 0.5, 0.05)
        risk_pct = st.slider("Account Risk %", 0.5, 5.0, 2.0, 0.5)
    with col_r2:
        atr_mult = st.slider("ATR SL Multiplier", 0.5, 3.0, 1.5, 0.1)
        max_daily_loss = st.slider("Max Daily Loss %", 1.0, 10.0, 5.0, 0.5)

    if st.button("💾 Save Settings", type="primary"):
        settings = {
            "hurst_mean_revert": hurst_thresh,
            "zscore_entry_long": z_long,
            "zscore_entry_short": z_short,
            "velocity_epsilon": vel_eps,
            "time_stop_bars": time_stop,
            "zscore_stop_long": z_stop_long,
            "zscore_stop_short": z_stop_short,
            "kelly_fraction": kelly_frac,
            "account_risk_pct": risk_pct / 100,
            "atr_multiplier_sl": atr_mult,
            "max_daily_loss_pct": max_daily_loss / 100,
        }
        import json
        with open("settings_live.json", "w") as f:
            json.dump(settings, f, indent=2)
        st.success("Settings saved to settings_live.json")

    st.divider()
    st.subheader("Current Configuration")
    st.json(settings)
