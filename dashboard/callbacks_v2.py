import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, html
import numpy as np

from dashboard.layout_v2 import POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR, BUY_COLOR, SELL_COLOR


def register_callbacks(app, provider):
    @app.callback(
        [
            Output("v2-conn-status", "children"), Output("v2-conn-status", "style"),
            Output("v2-time", "children"),
            Output("v2-sig-display", "children"), Output("v2-sig-display", "style"),
            Output("v2-domok-display", "children"), Output("v2-domok-display", "style"),
            Output("v2-zscore-display", "children"), Output("v2-hurst-display", "children"),
            Output("v2-bid-display", "children"), Output("v2-ask-display", "children"),
            Output("v2-ofi-display", "children"), Output("v2-ofi-display", "style"),
            Output("v2-delta-display", "children"), Output("v2-delta-display", "style"),
            Output("v2-pnl-display", "children"), Output("v2-pnl-display", "style"),
        ],
        [Input("v2-interval", "n_intervals")],
    )
    def update_top_bar(n):
        d = provider.refresh()
        if not d.get("connected_mt5"):
            return ("MT5 DISCONNECTED", {"color": NEGATIVE_COLOR}, d.get("timestamp", ""),
                    "N/A", {"color": NEUTRAL_COLOR}, "N/A", {"color": NEUTRAL_COLOR},
                    "N/A", "N/A", "N/A", "N/A", "N/A", {"color": NEUTRAL_COLOR},
                    "N/A", {"color": NEUTRAL_COLOR}, "N/A", {"color": NEUTRAL_COLOR})

        r_status = "R" if d.get("connected_rithmic") else "S" if d.get("using_synthetic") else "X"
        conn = f"MT5 OK [{r_status}] | {d.get('symbol','')} {d.get('timeframe',5)}m | {d.get('bar_count',0)} bars"
        sig_c = d.get("signal_color", NEUTRAL_COLOR)
        zs = f"{d['zscore']:.2f}" if not np.isnan(d.get("zscore", np.nan)) else "N/A"
        h = f"{d['hurst']:.3f}" if not np.isnan(d.get("hurst", np.nan)) else "N/A"
        bid = f"{d['bid']:.2f}"
        ask = f"{d['ask']:.2f}"
        ofi_c = POSITIVE_COLOR if d.get("ofi", 0) > 0 else NEGATIVE_COLOR if d.get("ofi", 0) < 0 else NEUTRAL_COLOR
        delta_c = POSITIVE_COLOR if d.get("cum_delta", 0) > 0 else NEGATIVE_COLOR
        pnl = d.get("total_pnl", 0)
        pnl_c = POSITIVE_COLOR if pnl >= 0 else NEGATIVE_COLOR
        pnl_s = f"+{pnl:.0f}" if pnl >= 0 else f"{pnl:.0f}"

        return (conn, {"color": POSITIVE_COLOR}, d["timestamp"],
                d["signal_text"], {"color": sig_c, "fontWeight": "bold", "fontSize": "24px"},
                d["dom_validation"], {"color": d["dom_color"], "fontWeight": "bold", "fontSize": "18px"},
                zs, h, bid, ask,
                f"{d.get('ofi', 0):+.1f}", {"color": ofi_c, "fontWeight": "bold", "fontSize": "24px"},
                f"{d.get('cum_delta', 0):+.0f}", {"color": delta_c, "fontWeight": "bold", "fontSize": "24px"},
                pnl_s, {"color": pnl_c, "fontWeight": "bold", "fontSize": "24px"})

    @app.callback(Output("v2-dom-ladder", "figure"), [Input("v2-interval", "n_intervals")])
    def update_dom_ladder(n):
        d = provider.refresh()
        fig = go.Figure()
        if not d.get("connected_mt5"):
            return fig

        dl = d.get("dom_ladder", {})
        asks = dl.get("asks", [])
        bids = dl.get("bids", [])
        all_labels, all_vols, all_colors = [], [], []

        for price, vol in reversed(asks):
            all_labels.append(f"ASK {price:.1f}")
            all_vols.append(-vol)
            all_colors.append(SELL_COLOR)
        for price, vol in bids:
            all_labels.append(f"BID {price:.1f}")
            all_vols.append(vol)
            all_colors.append(BUY_COLOR)

        if all_labels:
            fig.add_trace(go.Bar(y=all_labels, x=all_vols, orientation="h",
                                 marker_color=all_colors, text=[abs(v) for v in all_vols],
                                 textposition="outside", textfont=dict(size=9, color="#ccc")))
            mid_price = d.get("bid", 0)
            fig.add_vline(x=0, line_color="#888", line_width=1)
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#111133", plot_bgcolor="#111133",
                margin=dict(l=5, r=5, t=5, b=5), showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(tickfont=dict(size=9, color="#aaa")),
                title=dict(text=f"Bid/Ask Depth @ {mid_price:.1f}", font=dict(size=10, color="#aaa")),
                height=320, bargap=0.1,
            )
        return fig

    @app.callback(Output("v2-price-chart", "figure"), [Input("v2-interval", "n_intervals")])
    def update_price_chart(n):
        d = provider.refresh()
        fig = go.Figure()
        if not d.get("connected_mt5"):
            return fig

        ph = d.get("price_history", {})
        times, close = ph.get("time", []), ph.get("close", [])
        if times and close:
            fig.add_trace(go.Scatter(x=times, y=close, mode="lines",
                                     line=dict(color=POSITIVE_COLOR, width=1.5), name="Close"))

        sl_above = d.get("sl_zone_above", 0)
        sl_below = d.get("sl_zone_below", 0)
        if sl_above > 0 and times:
            fig.add_hline(y=sl_above, line_dash="dash", line_color=SELL_COLOR, opacity=0.5,
                          annotation_text=f"SL ▲ {sl_above}", annotation_position="top right")
        if sl_below > 0 and times:
            fig.add_hline(y=sl_below, line_dash="dash", line_color=BUY_COLOR, opacity=0.5,
                          annotation_text=f"SL ▼ {sl_below}", annotation_position="bottom right")

        for rn in d.get("round_numbers", [])[:8]:
            fig.add_hline(y=rn, line_dash="dot", line_color="#555", opacity=0.3)

        sm = d.get("signal_markers", {})
        bt = sm.get("buy_time", [])
        bp = sm.get("buy_price", [])
        if bt and bp:
            fig.add_trace(go.Scatter(x=bt, y=bp, mode="markers",
                                     marker=dict(symbol="triangle-up", size=10, color=BUY_COLOR),
                                     name="SIGNAL"))

        if d.get("sweep_detected") and times:
            fig.add_trace(go.Scatter(x=[times[-1]], y=[close[-1]], mode="markers",
                                     marker=dict(symbol="x", size=14, color=NEGATIVE_COLOR), name="SWEEP"))

        fig.update_layout(template="plotly_dark", paper_bgcolor="#111133", plot_bgcolor="#111133",
                          margin=dict(l=5, r=5, t=5, b=5), showlegend=True,
                          legend=dict(orientation="h", y=0.99, x=0.01, font=dict(size=8, color="#888")),
                          xaxis=dict(showgrid=False, tickfont=dict(size=8, color="#888")),
                          yaxis=dict(showgrid=True, gridcolor="#1a1a4a", tickfont=dict(size=8, color="#888"), tickprefix="$"),
                          hovermode="x unified", height=320)
        return fig

    @app.callback(Output("v2-ofi-meter", "figure"), [Input("v2-interval", "n_intervals")])
    def update_ofi_meter(n):
        d = provider.refresh()
        ofi = d.get("ofi", 0)
        fig = go.Figure()
        bar_color = POSITIVE_COLOR if ofi > 0 else NEGATIVE_COLOR if ofi < 0 else NEUTRAL_COLOR
        fig.add_trace(go.Bar(x=["OFI"], y=[ofi], marker_color=bar_color, text=f"{ofi:+.1f}", textposition="outside"))
        fig.add_hline(y=0, line_color="#888", line_width=1)
        title_color = POSITIVE_COLOR if ofi > 0 else NEGATIVE_COLOR if ofi < 0 else NEUTRAL_COLOR
        status = "BUYERS IN CONTROL" if ofi > 0 else "SELLERS IN CONTROL" if ofi < 0 else "NEUTRAL"
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111133", plot_bgcolor="#111133",
                          margin=dict(l=5, r=5, t=25, b=5), showlegend=False, height=150,
                          title=dict(text=status, font=dict(size=11, color=title_color)),
                          yaxis=dict(showgrid=True, gridcolor="#1a1a4a", zeroline=True, title="OFI"),
                          xaxis=dict(showticklabels=False))
        return fig

    @app.callback(Output("v2-delta-chart", "figure"), [Input("v2-interval", "n_intervals")])
    def update_delta_chart(n):
        d = provider.refresh()
        cd = d.get("cum_delta", 0)
        fig = go.Figure()
        bar_color = POSITIVE_COLOR if cd > 0 else NEGATIVE_COLOR
        fig.add_trace(go.Indicator(mode="gauge+number+delta", value=cd,
                                   number=dict(font=dict(color="#ddd", size=36)),
                                   delta=dict(reference=0, increasing_color=POSITIVE_COLOR, decreasing_color=NEGATIVE_COLOR),
                                   gauge=dict(axis=dict(range=[-5000, 5000], tickcolor="#ccc"),
                                              bar=dict(color=bar_color),
                                              bgcolor="#1a1a4a", borderwidth=0),
                                   title=dict(text="CUMULATIVE DELTA", font=dict(size=11, color="#777"))))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111133", plot_bgcolor="#111133",
                          margin=dict(l=5, r=5, t=25, b=5), height=150)
        return fig

    @app.callback(Output("v2-stats-chart", "figure"), [Input("v2-interval", "n_intervals")])
    def update_stats_chart(n):
        d = provider.refresh()
        zh = d.get("zscore_history", {})
        times, zs, hs = zh.get("time", []), zh.get("zscore", []), zh.get("hurst", [])
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        if times and zs:
            fig.add_trace(go.Scatter(x=times, y=zs, mode="lines",
                                     line=dict(color="#f0a500", width=1.5), name="Z"), secondary_y=False)
            fig.add_hline(y=3.0, line_dash="dash", line_color=SELL_COLOR, opacity=0.4, secondary_y=False)
            fig.add_hline(y=-2.5, line_dash="dash", line_color=BUY_COLOR, opacity=0.4, secondary_y=False)

        if times and hs:
            fig.add_trace(go.Scatter(x=times, y=hs, mode="lines",
                                     line=dict(color=NEGATIVE_COLOR, width=1, dash="dot"),
                                     name="H"), secondary_y=True)
            fig.add_hline(y=0.35, line_dash="dash", line_color=POSITIVE_COLOR, opacity=0.4, secondary_y=True)

        if d.get("iceberg_bid_detected") and times:
            fig.add_annotation(x=times[-1], y=0.5, text="ICEBERG BID", showarrow=True,
                               arrowcolor=BUY_COLOR, font=dict(size=9, color=BUY_COLOR), yref="y domain")
        if d.get("iceberg_ask_detected") and times:
            fig.add_annotation(x=times[-1], y=0.85, text="ICEBERG ASK", showarrow=True,
                               arrowcolor=SELL_COLOR, font=dict(size=9, color=SELL_COLOR), yref="y domain")

        fig.update_layout(template="plotly_dark", paper_bgcolor="#111133", plot_bgcolor="#111133",
                          margin=dict(l=5, r=5, t=5, b=5), showlegend=False, height=180,
                          hovermode="x unified",
                          xaxis=dict(showgrid=False, tickfont=dict(size=8, color="#888")),
                          yaxis=dict(showgrid=True, gridcolor="#1a1a4a", tickfont=dict(size=8, color="#f0a500"), secondary_y=False),
                          yaxis2=dict(showgrid=False, tickfont=dict(size=8, color=NEGATIVE_COLOR), secondary_y=True))
        return fig

    @app.callback(Output("v2-equity-chart", "figure"), [Input("v2-interval", "n_intervals")])
    def update_equity_chart(n):
        d = provider.refresh()
        eh = d.get("equity_history", {})
        times, equity = eh.get("time", []), eh.get("equity", [])
        fig = go.Figure()
        if not times or not equity:
            return fig

        init = equity[0] if equity else 10000
        pct = [(v / init - 1) * 100 for v in equity]
        latest = pct[-1] if pct else 0
        c = POSITIVE_COLOR if latest >= 0 else NEGATIVE_COLOR
        fig.add_trace(go.Scatter(x=times, y=pct, mode="lines", fill="tozeroy",
                                 fillcolor=f"rgba({'78,204,163' if latest >= 0 else '233,69,96'},0.15)",
                                 line=dict(color=c, width=2)))
        fig.add_hline(y=0, line_color="#888", line_width=1)

        if d.get("stop_hunt_high_conf") and times:
            fig.add_annotation(x=times[-1], y=pct[-1] if pct else 0,
                               text=f"STOP HUNT {d.get('stop_hunt_score', 0):.2f}",
                               showarrow=True, arrowcolor=NEGATIVE_COLOR,
                               font=dict(size=10, color=NEGATIVE_COLOR))

        fig.update_layout(template="plotly_dark", paper_bgcolor="#111133", plot_bgcolor="#111133",
                          margin=dict(l=5, r=5, t=5, b=5), showlegend=False, height=180,
                          xaxis=dict(showgrid=False, tickfont=dict(size=8, color="#888")),
                          yaxis=dict(showgrid=True, gridcolor="#1a1a4a", tickfont=dict(size=8, color="#888"), ticksuffix="%"))
        return fig

    @app.callback(Output("v2-trade-log", "children"), [Input("v2-interval", "n_intervals")])
    def update_trade_log(n):
        d = provider.refresh()
        lines = []

        if d.get("has_position"):
            p_c = BUY_COLOR if d.get("position_type") == "buy" else SELL_COLOR
            ppnl = d.get("position_pnl", 0)
            pp_c = POSITIVE_COLOR if ppnl >= 0 else NEGATIVE_COLOR
            lines.append(html.Div([
                html.Span("[POS] ", style={"color": NEUTRAL_COLOR, "fontWeight": "bold"}),
                html.Span(d.get("position_type", "").upper(), style={"color": p_c, "fontWeight": "bold"}),
                html.Span(f" Vol:{d.get('position_volume',0)} PnL:${ppnl:+.2f}", style={"color": pp_c}),
            ]))

        status = []
        status.append(html.Span(f"Stat:{d.get('signal_text','')} | ", style={"color": d.get("signal_color", "#ddd")}))
        status.append(html.Span(f"DOM:{d.get('dom_validation','')} | ", style={"color": d.get("dom_color", "#ddd")}))
        status.append(html.Span(f"Z:{d.get('zscore', np.nan):.2f} H:{d.get('hurst', np.nan):.3f} | ", style={"color": "#ddd"}))
        status.append(html.Span(f"OFI:{d.get('ofi',0):+.1f} B/A:{d.get('bid_ask_ratio',0):.2f}", style={"color": "#ddd"}))
        lines.append(html.Div(status))

        sl = []
        sl.append(html.Span(f"SL Below:{d.get('sl_zone_below',0):.0f} (d:{d.get('sl_dist_below',0):.1f}ATR) | ", style={"color": BUY_COLOR}))
        sl.append(html.Span(f"SL Above:{d.get('sl_zone_above',0):.0f} (d:{d.get('sl_dist_above',0):.1f}ATR) | ", style={"color": SELL_COLOR}))
        if d.get("stop_hunt_high_conf"):
            sl.append(html.Span(f"STOP HUNT! score:{d.get('stop_hunt_score',0):.2f}", style={"color": NEGATIVE_COLOR, "fontWeight": "bold"}))
        if d.get("iceberg_bid_detected"):
            sl.append(html.Span(f" | ICE BID p:{d.get('iceberg_persistence',0)} c:{d.get('iceberg_confidence',0):.2f}", style={"color": BUY_COLOR}))
        if d.get("iceberg_ask_detected"):
            sl.append(html.Span(f" | ICE ASK p:{d.get('iceberg_persistence',0)} c:{d.get('iceberg_confidence',0):.2f}", style={"color": SELL_COLOR}))
        lines.append(html.Div(sl))

        return html.Div(lines)
