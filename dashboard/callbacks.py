import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import numpy as np
import threading
import time
from datetime import datetime

from dashboard.layout import POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR, BUY_COLOR, SELL_COLOR
from notifications.telegram import TelegramNotifier


class TradingController:
    def __init__(self, provider):
        self.provider = provider
        self._running = False
        self._thread = None
        self._trader = None
        self._lock = threading.Lock()
        self.live_log = []
        self.trade_results = []
        self._prev_position = None
        self._tg = TelegramNotifier.get()
        self._status_text = "IDLE"
        self._last_error = ""
        self._stats = {"trades": 0, "wins": 0, "pnl": 0.0}

    @property
    def is_running(self):
        if self._running and self._thread and not self._thread.is_alive():
            self._running = False
            self._status_text = "CRASHED"
        return self._running

    def get_status(self):
        if self.is_running:
            return "RUNNING"
        if self._status_text == "CRASHED":
            return "CRASHED"
        if self._status_text == "STOPPED":
            return "STOPPED"
        return "IDLE"

    def force_stop(self):
        self._running = False
        self._thread = None
        if self._trader:
            try:
                self._trader.stop()
            except Exception:
                pass
        self._trader = None
        self._status_text = "FORCE STOPPED"

    def start(self):
        if self._running:
            return "ALREADY RUNNING"
        if not self.provider.connected:
            return "MT5 NOT CONNECTED"

        from live.trader import LiveTrader
        self._trader = LiveTrader()

        # Passive connect: will attach to whatever account is currently logged in MT5
        if not self._trader.connect():
            return "MT5 CONNECTION FAILED"

        try:
            self._trader.load_history(timeframe=5, bars=2000)
            self._trader.calibrate_hmm()
            self._trader.config.threshold.hmm_ranging_prob = 0.0
        except Exception as e:
            return f"INIT ERROR: {e}"

        self._running = True
        self._status_text = "RUNNING"
        self._prev_position = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._tg.send_startup()
        return "STARTED"

    def _detect_and_alert(self, signal, zscore):
        current_pos = None
        if self._trader and self._trader._open_position:
            current_pos = dict(self._trader._open_position)

        if self._prev_position is None and current_pos is not None:
            direction = "BUY" if current_pos["direction"] == 1 else "SELL"
            self._tg.send_entry_alert(
                direction=direction,
                price=current_pos["entry_price"],
                zscore=current_pos["entry_zscore"],
                hurst=0.0,
                volume=current_pos["volume"],
                sl=current_pos["sl"],
                reason=f"Z-score: {current_pos['entry_zscore']:.2f}",
            )

        elif self._prev_position is not None and current_pos is None:
            prev = self._prev_position
            direction = "BUY" if prev["direction"] == 1 else "SELL"
            bid, ask = self._trader.mt5.get_current_price()
            exit_price = ask if prev["direction"] == 1 else bid
            pnl = 0.0
            exit_reason = "signal_exit"
            if self._trader._trade_history:
                last = self._trader._trade_history[-1]
                pnl = last.get("pnl", 0.0)
                exit_reason = last.get("reason", "exit")
                self._stats["trades"] += 1
                if pnl > 0:
                    self._stats["wins"] += 1
                self._stats["pnl"] += pnl
            self._tg.send_exit_alert(
                direction=direction,
                entry_price=prev["entry_price"],
                exit_price=exit_price,
                pnl=pnl,
                reason=exit_reason,
            )

        self._prev_position = current_pos

    def _loop(self):
        while self._running:
            try:
                signal, zscore = self._trader.run_once()
                self._detect_and_alert(signal, zscore)

                status = "LONG" if signal == 1 else "SHORT" if signal == -1 else "WAIT"
                pos = "IN TRADE" if self._trader._open_position else "FLAT"
                entry = ""
                if self._trader._open_position:
                    p = self._trader._open_position
                    entry = f" | Entry: {p['entry_price']:.2f} SL: {p['sl']:.2f}"

                msg = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                       f"Signal: {status} | Z: {zscore:.2f} | {pos}{entry}")

                with self._lock:
                    self.live_log.append(msg)
                    if len(self.live_log) > 200:
                        self.live_log = self.live_log[-200:]

                if self._tg.should_send_hourly():
                    try:
                        data = self.provider.refresh()
                        self._tg.send_dashboard_snapshot(data)
                    except Exception:
                        pass

            except Exception as e:
                with self._lock:
                    self.live_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {e}")
                self._tg.send_error(str(e))

            for _ in range(60):
                if not self._running:
                    break
                time.sleep(1)

    def stop(self):
        was_running = self._running
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        summary_text = ""
        if self._trader:
            try:
                summary_text = self._trader.trade_summary()
                with self._lock:
                    self.trade_results.append(summary_text)
                self._trader.stop()
            except Exception:
                pass
        self._trader = None
        self._thread = None
        self._status_text = "STOPPED"
        if was_running:
            self._tg.send_shutdown(summary_text)
        return "STOPPED"

    def close_all(self):
        closed_msg = f"[{datetime.now().strftime('%H:%M:%S')}] CLOSED ALL POSITIONS"
        if self._trader and self._trader.mt5:
            try:
                self._trader.mt5.close_all_positions()
                with self._lock:
                    self.live_log.append(closed_msg)
            except Exception as e:
                with self._lock:
                    self.live_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] CLOSE ERROR: {e}")
        elif self.provider and self.provider.mt5:
            try:
                self.provider.mt5.close_all_positions()
                with self._lock:
                    self.live_log.append(closed_msg)
            except Exception as e:
                with self._lock:
                    self.live_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] CLOSE ERROR: {e}")
        self._tg.send_exit_alert("CLOSE ALL", 0, 0, 0, "manual close all")

    def get_log(self):
        with self._lock:
            return list(self.live_log)

    def get_summary(self):
        with self._lock:
            if self.trade_results:
                return self.trade_results[-1]
        return None


def register_callbacks(app, provider):
    trading_ctrl = TradingController(provider)

    @app.callback(
        [
            Output("btn-start", "disabled"),
            Output("btn-stop", "disabled"),
            Output("btn-restart", "disabled"),
            Output("trade-status-badge", "children"),
            Output("trade-status-badge", "style"),
        ],
        [
            Input("btn-start", "n_clicks"),
            Input("btn-stop", "n_clicks"),
            Input("btn-restart", "n_clicks"),
            Input("interval-refresh", "n_intervals"),
        ],
    )
    def toggle_trading(start_clicks, stop_clicks, restart_clicks, n_intervals):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False, True, False, "IDLE", {
                "fontSize": "13px", "fontWeight": "bold", "color": "#888",
            }

        triggered = ctx.triggered[0]["prop_id"]

        if triggered == "interval-refresh.n_intervals":
            running = trading_ctrl.is_running
            if running:
                s = trading_ctrl._stats
                info = f"ACTIVE | Trades: {s['trades']} Wins: {s['wins']} P&L: ${s['pnl']:.2f}"
                return True, False, False, info, {
                    "fontSize": "13px", "fontWeight": "bold", "color": POSITIVE_COLOR,
                }
            status = trading_ctrl.get_status()
            if status == "CRASHED":
                return False, True, False, "CRASHED - click RESTART or START", {
                    "fontSize": "13px", "fontWeight": "bold", "color": NEGATIVE_COLOR,
                }
            return False, True, False, "IDLE", {
                "fontSize": "13px", "fontWeight": "bold", "color": "#888",
            }

        if triggered == "btn-start.n_clicks":
            result = trading_ctrl.start()
            if result == "STARTED":
                return True, False, False, "TRADING ACTIVE", {
                    "fontSize": "13px", "fontWeight": "bold", "color": POSITIVE_COLOR,
                }
            else:
                return False, True, False, f"FAILED: {result}", {
                    "fontSize": "13px", "fontWeight": "bold", "color": NEGATIVE_COLOR,
                }

        elif triggered == "btn-stop.n_clicks":
            trading_ctrl.stop()
            return False, True, False, "STOPPED", {
                "fontSize": "13px", "fontWeight": "bold", "color": NEUTRAL_COLOR,
            }

        elif triggered == "btn-restart.n_clicks":
            trading_ctrl.force_stop()
            time.sleep(2)
            result = trading_ctrl.start()
            if result == "STARTED":
                return True, False, False, "RESTARTED - TRADING ACTIVE", {
                    "fontSize": "13px", "fontWeight": "bold", "color": POSITIVE_COLOR,
                }
            else:
                return False, True, False, f"RESTART FAILED: {result}", {
                    "fontSize": "13px", "fontWeight": "bold", "color": NEGATIVE_COLOR,
                }

        raise PreventUpdate

    @app.callback(
        Output("obsidian-status", "children"),
        [
            Input("btn-obsidian-dashboard", "n_clicks"),
            Input("btn-obsidian-trades", "n_clicks"),
            Input("btn-obsidian-signals", "n_clicks"),
            Input("btn-obsidian-backtests", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def open_in_obsidian(dash_clicks, trades_clicks, sig_clicks, bt_clicks):
        from obsidian_sync import get_writer
        from config.settings import GOLD_CONFIG
        cfg = getattr(GOLD_CONFIG, "obsidian", None)
        if cfg is None:
            return "Obsidian config missing"
        if not cfg.enabled:
            return "Obsidian disabled — set OBSIDIAN_ENABLED=true in .env"
        w = get_writer(cfg)
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        triggered = ctx.triggered[0]["prop_id"]
        try:
            if triggered == "btn-obsidian-dashboard.n_clicks":
                uri = w.open_in_obsidian("20-Research/GOLD-Trading/Dashboard.md")
            elif triggered == "btn-obsidian-trades.n_clicks":
                uri = w.open_in_obsidian("20-Research/GOLD-Trading/Trades")
            elif triggered == "btn-obsidian-signals.n_clicks":
                uri = w.open_in_obsidian("20-Research/GOLD-Trading/Signals")
            elif triggered == "btn-obsidian-backtests.n_clicks":
                uri = w.open_in_obsidian("20-Research/GOLD-Trading/Backtests")
            else:
                raise PreventUpdate
            return html.A(
                f"Opened: {uri}",
                href=uri,
                target="_blank",
                style={"color": "#7C3AED", "textDecoration": "underline"},
            )
        except Exception as e:
            return f"Error: {e}"

    @app.callback(
        Output("btn-close-all", "children"),
        Input("btn-close-all", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_all_positions(n):
        trading_ctrl.close_all()

        def reset_btn():
            time.sleep(3)
            return "CLOSE ALL"

        return "CLOSED - check log"

    @app.callback(
        [
            Output("conn-status", "children"),
            Output("conn-status", "style"),
            Output("time-display", "children"),
            Output("sig-display", "children"),
            Output("sig-display", "style"),
            Output("zscore-display", "children"),
            Output("zscore-display", "style"),
            Output("hurst-display", "children"),
            Output("hurst-display", "style"),
            Output("regime-display", "children"),
            Output("regime-display", "style"),
            Output("bid-display", "children"),
            Output("spread-display", "children"),
            Output("pnl-display", "children"),
            Output("pnl-display", "style"),
            Output("equity-display", "children"),
            Output("position-display", "children"),
            Output("position-display", "style"),
            Output("pospnl-display", "children"),
            Output("pospnl-display", "style"),
            Output("atr-display", "children"),
        ],
        [Input("interval-refresh", "n_intervals")],
    )
    def update_stats(n):
        data = provider.refresh()

        if not data.get("connected"):
            conn_style = {"fontSize": "12px", "color": NEGATIVE_COLOR}
            return (
                "MT5 DISCONNECTED", conn_style,
                data.get("timestamp", ""),
                "N/A", {"fontSize": "18px", "fontWeight": "bold", "color": NEUTRAL_COLOR},
                "N/A", {"fontSize": "18px", "fontWeight": "bold", "color": "#e0e0e0"},
                "N/A", {"fontSize": "18px", "fontWeight": "bold", "color": "#e0e0e0"},
                "N/A", {"fontSize": "18px", "fontWeight": "bold", "color": "#e0e0e0"},
                "N/A", {"fontSize": "18px", "fontWeight": "bold", "color": "#e0e0e0"},
                "N/A",
                "N/A", "N/A",
                "N/A", "N/A",
                "N/A", {"fontSize": "18px", "fontWeight": "bold", "color": NEUTRAL_COLOR},
            )

        signal = data["signal"]
        sig_color = BUY_COLOR if signal == 1 else SELL_COLOR if signal == -1 else NEUTRAL_COLOR
        sig_style = {"fontSize": "28px", "fontWeight": "bold", "color": sig_color}

        zscore = data["zscore"]
        zscore_color = BUY_COLOR if zscore < -2.5 else SELL_COLOR if zscore > 2.5 else "#e0e0e0"
        zscore_style = {"fontSize": "28px", "fontWeight": "bold", "color": zscore_color}

        hurst = data["hurst"]
        hurst_color = POSITIVE_COLOR if hurst < 0.45 else NEUTRAL_COLOR if hurst < 0.55 else NEGATIVE_COLOR
        hurst_style = {"fontSize": "28px", "fontWeight": "bold", "color": hurst_color}

        regime = data["hurst_regime"]
        regime_color = POSITIVE_COLOR if regime == "M-REVERT" else NEGATIVE_COLOR if regime == "TRENDING" else NEUTRAL_COLOR
        regime_style = {"fontSize": "18px", "fontWeight": "bold", "color": regime_color}

        pnl = data["total_pnl"]
        pnl_color = POSITIVE_COLOR if pnl >= 0 else NEGATIVE_COLOR
        pnl_style = {"fontSize": "18px", "fontWeight": "bold", "color": pnl_color}

        pos_text = f"{data['position_type'].upper()} {data['position_volume']} lots" if data["has_position"] else "NO POSITION"
        pos_color = BUY_COLOR if data.get("position_type") == "buy" else SELL_COLOR if data.get("position_type") == "sell" else NEUTRAL_COLOR
        pos_style = {"fontSize": "18px", "fontWeight": "bold", "color": pos_color}

        pos_pnl = data["position_pnl"]
        pospnl_color = POSITIVE_COLOR if pos_pnl >= 0 else NEGATIVE_COLOR
        pospnl_style = {"fontSize": "18px", "fontWeight": "bold", "color": pospnl_color}

        conn_status = f"CONNECTED | {data['symbol']} | {data['timeframe']}min | {data['bar_count']} bars"

        bid_str = f"{data['bid']:.{data['digits']}f}" if data['digits'] > 0 else f"{data['bid']:.2f}"
        zscore_str = f"{zscore:.2f}" if not np.isnan(zscore) else "N/A"
        hurst_str = f"{hurst:.3f}" if not np.isnan(hurst) else "N/A"
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        pospnl_str = f"+{pos_pnl:.2f}" if pos_pnl >= 0 else f"{pos_pnl:.2f}" if pos_pnl else "0.00"
        atr_str = f"{data['atr']:.3f}" if data['atr'] else "N/A"

        return (
            conn_status, {"fontSize": "12px", "color": POSITIVE_COLOR},
            data["timestamp"],
            data["signal_text"], sig_style,
            zscore_str, zscore_style,
            hurst_str, hurst_style,
            regime, regime_style,
            bid_str,
            f"{data['spread']} pts",
            pnl_str, pnl_style,
            f"{data['account_equity']:.2f}",
            pos_text, pos_style,
            pospnl_str, pospnl_style,
            atr_str,
        )

    @app.callback(
        Output("price-chart", "figure"),
        [Input("interval-refresh", "n_intervals")],
    )
    def update_price_chart(n):
        data = provider.refresh()
        if not data.get("connected"):
            return go.Figure()

        fig = make_subplots(specs=[[{"secondary_y": False}]])

        ph = data.get("price_history", {})
        times = ph.get("time", [])
        close = ph.get("close", [])

        if times and close:
            fig.add_trace(go.Scatter(
                x=times, y=close,
                mode="lines",
                line=dict(color="#4ecca3", width=1.5),
                name="Close",
                hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
            ))

            sm = data.get("signal_markers", {})
            buy_times = sm.get("time", [])
            buy_prices = sm.get("long_price", [])
            if buy_times and buy_prices:
                fig.add_trace(go.Scatter(
                    x=buy_times, y=buy_prices,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color=BUY_COLOR,
                                line=dict(width=1, color="#fff")),
                    name="BUY Signal",
                    hovertemplate="BUY @ $%{y:.2f}<extra></extra>",
                ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#16213e",
            plot_bgcolor="#16213e",
            margin=dict(l=10, r=10, t=5, b=5),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01,
                        font=dict(size=10, color="#888")),
            xaxis=dict(showgrid=False, color="#888", tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="#0f3460", color="#888",
                       tickfont=dict(size=9), tickprefix="$"),
            hovermode="x unified",
        )

        return fig

    @app.callback(
        Output("stats-chart", "figure"),
        [Input("interval-refresh", "n_intervals")],
    )
    def update_stats_chart(n):
        data = provider.refresh()
        if not data.get("connected"):
            return go.Figure()

        zh = data.get("zscore_history", {})
        times = zh.get("time", [])
        zscores = zh.get("zscore", [])
        hursts = zh.get("hurst", [])

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        if times and zscores:
            fig.add_trace(go.Scatter(
                x=times, y=zscores,
                mode="lines",
                line=dict(color="#f0a500", width=1.5),
                name="Z-Score",
                hovertemplate="Z: %{y:.2f}σ<extra></extra>",
            ), secondary_y=False)

            fig.add_hline(y=2.5, line_dash="dash", line_color=SELL_COLOR, opacity=0.4,
                          secondary_y=False)
            fig.add_hline(y=-2.5, line_dash="dash", line_color=BUY_COLOR, opacity=0.4,
                          secondary_y=False)
            fig.add_hline(y=3.5, line_dash="dot", line_color=SELL_COLOR, opacity=0.2,
                          secondary_y=False)
            fig.add_hline(y=-3.5, line_dash="dot", line_color=BUY_COLOR, opacity=0.2,
                          secondary_y=False)

        if times and hursts:
            fig.add_trace(go.Scatter(
                x=times, y=hursts,
                mode="lines",
                line=dict(color="#e94560", width=1.0, dash="dot"),
                name="Hurst",
                hovertemplate="H: %{y:.3f}<extra></extra>",
            ), secondary_y=True)

            fig.add_hline(y=0.45, line_dash="dash", line_color=POSITIVE_COLOR, opacity=0.4,
                          secondary_y=True)
            fig.add_hline(y=0.55, line_dash="dash", line_color=NEGATIVE_COLOR, opacity=0.4,
                          secondary_y=True)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#16213e",
            plot_bgcolor="#16213e",
            margin=dict(l=10, r=10, t=5, b=5),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01,
                        font=dict(size=10, color="#888")),
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=False, color="#888", tickfont=dict(size=9))
        fig.update_yaxes(title_text="Z-Score (σ)", showgrid=True, gridcolor="#0f3460",
                         color="#f0a500", tickfont=dict(size=9), secondary_y=False)
        fig.update_yaxes(title_text="Hurst (H)", showgrid=False,
                         color="#e94560", tickfont=dict(size=9), secondary_y=True)

        return fig

    @app.callback(
        Output("equity-chart", "figure"),
        [Input("interval-refresh", "n_intervals")],
    )
    def update_equity_chart(n):
        data = provider.refresh()
        if not data.get("connected"):
            return go.Figure()

        eh = data.get("equity_history", {})
        times = eh.get("time", [])
        equity = eh.get("equity", [])

        if not times or not equity:
            return go.Figure()

        initial = equity[0] if equity else 10000
        pct = [(v / initial - 1) * 100 for v in equity]

        latest = pct[-1] if pct else 0
        fill_color = "rgba(78, 204, 163, 0.15)" if latest >= 0 else "rgba(233, 69, 96, 0.15)"
        line_color = POSITIVE_COLOR if latest >= 0 else NEGATIVE_COLOR

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=pct,
            mode="lines",
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            name="Equity",
            hovertemplate="%{x}<br>%{y:.3f}%<extra></extra>",
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.5)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#16213e",
            plot_bgcolor="#16213e",
            margin=dict(l=10, r=10, t=5, b=5),
            showlegend=False,
            hovermode="x",
            xaxis=dict(showgrid=False, color="#888", tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="#0f3460", color="#888",
                       tickfont=dict(size=9), ticksuffix="%",
                       title="Return %"),
        )

        return fig

    @app.callback(
        Output("trade-log", "children"),
        [Input("interval-refresh", "n_intervals")],
    )
    def update_trade_log(n):
        data = provider.refresh()
        lines = []

        if trading_ctrl.is_running:
            log_entries = trading_ctrl.get_log()
            for entry in log_entries[-20:]:
                entry_color = NEGATIVE_COLOR if "ERROR" in entry else "#e0e0e0"
                lines.append(html.Div(entry, style={"padding": "2px 0", "color": entry_color,
                                                     "fontFamily": "monospace", "fontSize": "11px"}))

            summary = trading_ctrl.get_summary()
            if summary:
                lines.append(html.Div("─" * 40, style={"color": "#555", "padding": "4px 0"}))
                for sline in summary.split("\n"):
                    lines.append(html.Div(sline, style={"color": NEUTRAL_COLOR, "fontFamily": "monospace",
                                                         "fontSize": "11px", "padding": "1px 0"}))
        elif not data.get("connected"):
            return html.Div("No connection to MT5.", style={"color": NEGATIVE_COLOR})
        else:
            if data.get("has_position"):
                pos_type = data["position_type"].upper()
                pos_color = BUY_COLOR if pos_type == "BUY" else SELL_COLOR
                pos_pnl = data["position_pnl"]
                pnl_color = POSITIVE_COLOR if pos_pnl >= 0 else NEGATIVE_COLOR
                lines.append(html.Div([
                    html.Span("[ACTIVE] ", style={"color": NEUTRAL_COLOR, "fontWeight": "bold"}),
                    html.Span(pos_type, style={"color": pos_color, "fontWeight": "bold"}),
                    html.Span(f" @ {data['position_open_price']:.2f}  "),
                    html.Span(f"P&L: ${pos_pnl:+.2f}", style={"color": pnl_color, "fontWeight": "bold"}),
                    html.Span(f"  Vol: {data['position_volume']} lot(s)"),
                    html.Span(f"  SL: {data.get('position_sl', 'N/A')}"),
                ], style={"padding": "3px 0"}))

            status = html.Div([
                html.Span(f"Z-Score: {data['zscore']:.2f} | ", style={"color": "#f0a500"}),
                html.Span(f"Hurst: {data['hurst']:.3f} | ", style={"color": "#e94560"}),
                html.Span(f"Velocity: {data['velocity']:.3f} | ", style={"color": "#888"}),
                html.Span(f"Signal: {data['signal_text']}", style={
                    "color": BUY_COLOR if data['signal'] == 1 else SELL_COLOR if data['signal'] == -1 else NEUTRAL_COLOR}),
            ], style={"padding": "3px 0"})
            lines.append(status)

            lines.append(html.Div([
                html.Span(f"ATR: {data['atr']:.3f} | ", style={"color": "#888"}),
                html.Span(f"Spread: {data['spread']}pts | ", style={"color": "#888"}),
                html.Span(f"Bars: {data['bar_count']}"),
            ], style={"padding": "3px 0"}))

        return html.Div(lines)

    @app.callback(
        Output("debug-store", "children"),
        [Input("interval-refresh", "n_intervals")],
    )
    def update_debug(n):
        data = provider.refresh()
        return f"tick={n}"

    @app.callback(
        [
            Output("snapshot-status", "children"),
            Output("snapshot-status", "style"),
            Output("snapshot-list", "children"),
        ],
        [
            Input("btn-subscribe", "n_clicks"),
            Input("btn-unsubscribe", "n_clicks"),
            Input("snapshot-time-dropdown", "value"),
            Input("interval-refresh", "n_intervals"),
        ],
    )
    def manage_snapshot_subscription(sub_clicks, unsub_clicks, time_value, n_intervals):
        from dashboard.subscribe import get_scheduler
        from notifications.telegram import TelegramNotifier
        
        scheduler = get_scheduler()
        tg = TelegramNotifier.get()
        chat_id = tg.chat_id
        
        ctx = dash.callback_context
        if not ctx.triggered:
            # Just show current subscriptions
            return "", {"color": "#4ecca3"}, scheduler.list_subscriptions(chat_id)
        
        triggered = ctx.triggered[0]["prop_id"]
        
        if triggered == "btn-subscribe.n_clicks" and time_value:
            result = scheduler.subscribe(chat_id, time_value)
            return result, {"color": "#4ecca3"}, scheduler.list_subscriptions(chat_id)
        
        elif triggered == "btn-unsubscribe.n_clicks" and time_value:
            result = scheduler.unsubscribe(chat_id, time_value)
            return result, {"color": "#e94560"}, scheduler.list_subscriptions(chat_id)
        
        elif triggered == "btn-unsubscribe.n_clicks" and not time_value:
            # Unsubscribe all
            result = scheduler.unsubscribe(chat_id, None)
            return result, {"color": "#e94560"}, scheduler.list_subscriptions(chat_id)
        
        return "", {"color": "#4ecca3"}, scheduler.list_subscriptions(chat_id)

    return app


def html_div(text, color="#e0e0e0"):
    return html.Div(text, style={"color": color})