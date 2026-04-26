import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Math Trading V2 - DOM Monitor",
    update_title=None,
    suppress_callback_exceptions=True,
)

CONTENT_STYLE = {
    "margin-left": "0", "padding": "15px", "color": "#e0e0e0",
    "backgroundColor": "#0a0a1a", "minHeight": "100vh",
}

CARD_STYLE = {
    "backgroundColor": "#111133", "border": "1px solid #222255",
    "borderRadius": "6px", "padding": "8px", "marginBottom": "8px", "color": "#ddd",
}

HEADER_STYLE = {
    "backgroundColor": "#0f0f3a", "padding": "8px 16px",
    "borderBottom": "2px solid #e94560", "display": "flex",
    "justifyContent": "space-between", "alignItems": "center",
}

POSITIVE_COLOR = "#4ecca3"
NEGATIVE_COLOR = "#e94560"
NEUTRAL_COLOR = "#f0a500"
BUY_COLOR = "#4ecca3"
SELL_COLOR = "#e94560"


def stat_card(label, value_id, suffix="", color="#ddd", large=False):
    fs = "24px" if large else "14px"
    return html.Div([
        html.Div(label, style={"fontSize": "9px", "color": "#777", "textTransform": "uppercase"}),
        html.Div([
            html.Span(id=value_id, style={"fontSize": fs, "fontWeight": "bold", "color": color}),
            html.Span(suffix, style={"fontSize": "10px", "color": color}) if suffix else None,
        ]),
    ], style={"textAlign": "center", "padding": "6px 4px", "backgroundColor": "#111133",
              "border": "1px solid #222255", "borderRadius": "6px"})


def build_layout(refresh_interval_ms=5000):
    return html.Div([
        dcc.Interval(id="v2-interval", interval=refresh_interval_ms, n_intervals=0),

        html.Div([
            html.Div([
                html.H3("V2 DOM-VALIDATED TRADING", style={"margin": 0, "color": "#ddd"}),
                html.Span(id="v2-conn-status", style={"fontSize": "11px", "color": NEGATIVE_COLOR}),
            ]),
            html.Div([
                html.Span("=" * 50, style={"fontSize": "10px", "color": "#0f0f3a"}),
                html.Span(" GC/GOLD-Pro | 5m | Stat+D Layer ", style={"fontSize": "12px", "color": "#f0a500"}),
            ]),
            html.Div([
                html.Span(id="v2-time", style={"fontSize": "12px", "color": "#888"}),
            ]),
        ], style=HEADER_STYLE),

        dbc.Row([
            dbc.Col(stat_card("STAT SIGNAL", "v2-sig-display", "", NEUTRAL_COLOR, True), width=2),
            dbc.Col(stat_card("DOM VALID", "v2-domok-display", "", NEUTRAL_COLOR, True), width=2),
            dbc.Col(stat_card("Z-SCORE", "v2-zscore-display", "s", "#ddd", True), width=1),
            dbc.Col(stat_card("HURST", "v2-hurst-display", "", "#ddd", True), width=1),
            dbc.Col(stat_card("BID", "v2-bid-display", "", "#ddd"), width=1),
            dbc.Col(stat_card("ASK", "v2-ask-display", "", "#ddd"), width=1),
            dbc.Col(stat_card("OFI", "v2-ofi-display", "", "#ddd", True), width=1),
            dbc.Col(stat_card("DELTA", "v2-delta-display", "", "#ddd", True), width=1),
            dbc.Col(stat_card("P&L", "v2-pnl-display", "$", POSITIVE_COLOR, True), width=2),
        ], className="g-1", style={"marginBottom": "8px"}),

        dbc.Row([
            dbc.Col([
                html.Div("DOM LADDER (Top 10)", style={"fontSize": "10px", "color": "#777", "marginBottom": "3px"}),
                dcc.Graph(id="v2-dom-ladder", config={"displayModeBar": False},
                          style={"height": "320px", "backgroundColor": "#111133",
                                 "borderRadius": "5px", "border": "1px solid #222255"}),
            ], width=3),
            dbc.Col([
                html.Div("PRICE + SL ZONES + SIGNALS", style={"fontSize": "10px", "color": "#777", "marginBottom": "3px"}),
                dcc.Graph(id="v2-price-chart", config={"displayModeBar": False},
                          style={"height": "320px", "backgroundColor": "#111133",
                                 "borderRadius": "5px", "border": "1px solid #222255"}),
            ], width=5),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Div("OFI METER", style={"fontSize": "10px", "color": "#777", "marginBottom": "3px"}),
                        dcc.Graph(id="v2-ofi-meter", config={"displayModeBar": False},
                                  style={"height": "150px", "backgroundColor": "#111133",
                                         "borderRadius": "5px", "border": "1px solid #222255"}),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div("CUMULATIVE DELTA", style={"fontSize": "10px", "color": "#777", "marginBottom": "3px"}),
                        dcc.Graph(id="v2-delta-chart", config={"displayModeBar": False},
                                  style={"height": "150px", "backgroundColor": "#111133",
                                         "borderRadius": "5px", "border": "1px solid #222255"}),
                    ]),
                ]),
            ], width=4),
        ], className="g-1", style={"marginBottom": "8px"}),

        dbc.Row([
            dbc.Col([
                html.Div("Z-SCORE + HURST + ICEBERG MARKERS", style={"fontSize": "10px", "color": "#777", "marginBottom": "3px"}),
                dcc.Graph(id="v2-stats-chart", config={"displayModeBar": False},
                          style={"height": "180px", "backgroundColor": "#111133",
                                 "borderRadius": "5px", "border": "1px solid #222255"}),
            ], width=6),
            dbc.Col([
                html.Div("EQUITY + STOP HUNT ALERTS", style={"fontSize": "10px", "color": "#777", "marginBottom": "3px"}),
                dcc.Graph(id="v2-equity-chart", config={"displayModeBar": False},
                          style={"height": "180px", "backgroundColor": "#111133",
                                 "borderRadius": "5px", "border": "1px solid #222255"}),
            ], width=6),
        ], className="g-1", style={"marginBottom": "8px"}),

        html.Div([
            html.Div("TRADE LOG + SL ZONE MAP", style={"fontSize": "10px", "color": "#777", "marginBottom": "3px"}),
            html.Div(id="v2-trade-log", style={"fontSize": "10px", "maxHeight": "100px", "overflowY": "auto",
                                                "backgroundColor": "#111133", "padding": "8px",
                                                "borderRadius": "5px", "border": "1px solid #222255"}),
        ]),

        html.Div(id="v2-debug-store", style={"display": "none"}),
    ], style=CONTENT_STYLE)
