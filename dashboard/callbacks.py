import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State
import numpy as np
from datetime import datetime

from dashboard.layout import POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR, BUY_COLOR, SELL_COLOR


def register_callbacks(app, provider):

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
        if not data.get("connected"):
            return html_div("No connection to MT5.", NEGATIVE_COLOR)

        lines = []

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

    return app


def html_div(text, color="#e0e0e0"):
    return html.Div(text, style={"color": color})
