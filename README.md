# 📈 Gold Trading System

Statistical mean-reversion trading system for XAU/USD (GOLD-Pro) with DOM-validated entries via Rithmic L2 order book.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gold Trading System                       │
├──────────────────────┬──────────────────────────────────────┤
│  V1: Statistical     │  V2: DOM Validation                   │
│  ──────────────────  │  ──────────────────                   │
│  • Hurst Exponent    │  • Rithmic L2 Orderbook               │
│  • Rolling Z-Score   │  • Order Flow Imbalance (OFI)         │
│  • Velocity Filter   │  • Cumulative Delta                   │
│  • HMM Regime Detect │  • Absorption Detection               │
│  • Kelly Sizing      │  • Iceberg Detection                  │
│  • ATR Stop Loss     │  • Stop Hunt Scoring                  │
│                      │  • SL Zone Mapping                    │
│                      │  • Footprint / VPOC                   │
├──────────────────────┴──────────────────────────────────────┤
│  Dash Dashboard (V1)  │  Streamlit App (V1+V2)              │
│  • Price + Z-Score    │  • 4-tab interface                   │
│  • START/STOP buttons │  • Live trading controls             │
│  • Equity curve       │  • DOM ladder + footprint            │
│  • Telegram alerts    │  • Backtest explorer                 │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Local (Full Functionality)

```powershell
# Install dependencies
pip install -r requirements.txt

# Run V1 live trading with Dash dashboard
python main.py dashboard

# Run V1 live trading (CLI)
python main.py live --tf 5 --hmm-gate 0.0

# Run Streamlit app (V1 + V2)
streamlit run streamlit_app.py
```

### Streamlit Cloud (Partial)

1. Fork this repo
2. Deploy at [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in Streamlit Cloud dashboard:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

**Cloud limitations**: MT5 and Rithmic require Windows desktop. Cloud deployment supports backtest viewing and settings management only.

## V1: Statistical Engine

### Indicators
| Indicator | Method | Purpose |
|---|---|---|
| Hurst Exponent | Variance-time | Detect mean-reverting regime (H < 0.35) |
| Rolling Z-Score | 100-bar window | Statistical anomaly detection |
| Adaptive Z-Score | Rolling kurtosis, 200-bar | Tighten entry in fat tails, loosen in thin tails |
| Velocity | MA deviation | Confirm price has flattened |
| HMM | 2-state Gaussian | Regime classification (disabled by default) |
| Multi-TF Hurst | 15m confirmation | Optional filter: only enter when 15m is mean-reverting |

### Signal Logic
```
LONG  = Hurst<0.35 AND Z_adaptive<-2.5 AND Velocity≈0 AND Session(London/NY) AND ATR_ratio<1.6 [AND 15m_Hurst<0.45]
SHORT = Hurst<0.35 AND Z_adaptive>+3.0 AND Velocity≈0 AND Session(London/NY) AND ATR_ratio<1.6 [AND 15m_Hurst<0.45]
```

### Adaptive Z-Score
The entry threshold is dynamically scaled by recent return kurtosis:
- **Fat tails (kurtosis > 4.0)** → divide z-score by 1.18 → requires stronger signal
- **Thin tails (kurtosis < 1.5)** → divide z-score by 0.90 → accepts weaker signals
- Configurable via `kurtosis_tighten_factor`, `kurtosis_loosen_factor`

### Exit Rules
| Rule | Condition |
|---|---|
| Z-Score Stop (Long) | Z ≤ -3.5 |
| Z-Score Stop (Short) | Z ≥ +3.5 |
| Z-Trail | Best z retraces 30% from peak |
| Time Stop | 5 bars elapsed (configurable) |
| Partial Profit | Z returns to ±0.5 |
| Mean Reversion | Z returns to 0 |
| Hurst Regime Flip | H > 0.55 while in trade |

### Performance (5000 bars, GOLD-Pro 5m, V1.1 with adaptive Z)
| Metric | Baseline | Adaptive Z |
|---|---|---|
| Win Rate | 75.0% | 64.3% |
| Total Trades | 16 | 14 |
| Total Return | +0.29% | **+1.44%** |
| Sharpe Ratio | 1.11 | **4.28** |
| Profit Factor | 1.21 | **2.72** |
| Max Drawdown | — | 0.88% |

Higher return, much higher Sharpe, smaller drawdown. Adaptive Z filters out weak signals in fat-tailed regimes where mean reversion tends to fail.

## V2: DOM Validation

V2 wraps V1 signals with orderflow confirmation. **DOM gates are OFF by default** — V2 = V1 until you enable them.

### Orderflow Indicators
| Indicator | Description |
|---|---|
| DOM Ladder | Top-5 bid/ask depth imbalance |
| OFI | Order Flow Imbalance with SMA smoothing |
| Cumulative Delta | Running buy/sell volume delta |
| Absorption | Large resting limit order detection |
| Iceberg | Hidden order detection (traded/visible ratio > 2.0) |
| Stop Hunt | SL sweep + reversal scoring (≥ 0.7 = high confidence) |
| SL Zones | Swing fractal + round number cluster map |
| Footprint | Volume-at-price with VPOC identification |

### Signal Flow
```
V1 Statistical Signal → DOM Gate Validation → Execute
                              ↓
                    (gates OFF = passthrough)
```

## Configuration

Edit thresholds in `config/settings.py` or use the Streamlit Settings tab:

```python
threshold.hurst_mean_revert = 0.35    # Mean-reversion threshold
threshold.zscore_entry_long = -2.5    # Long entry Z-score
threshold.zscore_entry_short = 3.0    # Short entry Z-score
threshold.velocity_epsilon = 3.0      # Velocity flatness threshold
threshold.time_stop_bars = 3          # Max bars in trade
threshold.zscore_stop_long = -3.5     # Long stop loss Z-score
threshold.zscore_stop_short = 3.5     # Short stop loss Z-score
```

## Telegram Alerts

Receive real-time alerts on entry, exit, hourly snapshots, and errors.

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather)
2. Add token and chat ID to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
3. Start trading — alerts are automatic

## Project Structure

```
├── config/           # Settings (V1 + V2)
├── stats/            # Hurst, Z-score, velocity, HMM
├── signals/          # Signal generator
├── risk/             # Kelly sizing, exits, stops
├── backtest/         # vectorbt backtest engine
├── data/             # Data fetchers (MT5, synthetic)
├── live/             # MT5 adapter, live trader
├── orderflow/        # V2 DOM indicators
├── dashboard/        # V1 Dash dashboard
├── notifications/    # Telegram notifier
├── main.py           # V1 CLI
├── main_v2.py        # V2 CLI
├── streamlit_app.py  # Streamlit app (V1 + V2)
└── requirements.txt
```

## Releases

| Version | Description |
|---|---|
| [v1](../../releases/tag/v1) | Statistical mean-reversion engine, Dash dashboard, Telegram alerts |
| [v2](../../releases/tag/v2) | DOM-validated entries, Rithmic L2, 8 orderflow indicators, 6-panel dashboard |

## Disclaimer

This is a research project. Past performance does not guarantee future results. Trade at your own risk.
