import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Math Trading - XAU/USD Monitor",
    update_title=None,
    suppress_callback_exceptions=True,
)

CONTENT_STYLE = {
    "margin-left": "0",
    "padding": "20px",
    "color": "#e0e0e0",
    "backgroundColor": "#1a1a2e",
    "minHeight": "100vh",
}

CARD_STYLE = {
    "backgroundColor": "#16213e",
    "border": "1px solid #0f3460",
    "borderRadius": "8px",
    "padding": "12px",
    "marginBottom": "10px",
    "color": "#e0e0e0",
}

HEADER_STYLE = {
    "backgroundColor": "#0f3460",
    "padding": "10px 20px",
    "borderBottom": "2px solid #e94560",
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
}

STAT_BOX_STYLE = {
    "textAlign": "center",
    "padding": "10px 5px",
    "backgroundColor": "#16213e",
    "border": "1px solid #0f3460",
    "borderRadius": "8px",
}

POSITIVE_COLOR = "#4ecca3"
NEGATIVE_COLOR = "#e94560"
NEUTRAL_COLOR = "#f0a500"
BUY_COLOR = "#4ecca3"
SELL_COLOR = "#e94560"


def make_stat_card(label, value_id, suffix="", color="#e0e0e0", large=False):
    fontSize = "28px" if large else "18px"
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#888", "textTransform": "uppercase"}),
        html.Div([
            html.Span(id=value_id, style={"fontSize": fontSize, "fontWeight": "bold", "color": color}),
            html.Span(suffix, style={"fontSize": "12px", "color": color}) if suffix else None,
        ]),
    ], style=STAT_BOX_STYLE)


def build_layout(refresh_interval_ms: int = 5000):
    return html.Div([
        dcc.Interval(id="interval-refresh", interval=refresh_interval_ms, n_intervals=0),

        html.Div([
            html.Div([
                html.H3("XAU/USD Statistical Trading Dashboard", style={"margin": 0, "color": "#e0e0e0"}),
                html.Span(id="conn-status", style={"fontSize": "12px", "color": NEGATIVE_COLOR}),
            ]),
            html.Div([
                html.Span(id="time-display", style={"fontSize": "14px", "color": "#888"}),
            ]),
        ], style=HEADER_STYLE),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("TRADING CONTROL", style={"fontSize": "11px", "color": "#888", "textTransform": "uppercase", "marginBottom": "6px"}),
                    dbc.ButtonGroup([
                        dbc.Button("START", id="btn-start", color="success", size="sm",
                                   style={"width": "80px", "fontWeight": "bold"}),
                        dbc.Button("STOP", id="btn-stop", color="danger", size="sm",
                                   style={"width": "80px", "fontWeight": "bold"}),
                        dbc.Button("RESTART", id="btn-restart", color="info", size="sm",
                                   style={"width": "90px", "fontWeight": "bold"}),
                        dbc.Button("CLOSE ALL", id="btn-close-all", color="warning", size="sm",
                                   style={"width": "110px", "fontWeight": "bold"}),
                    ], size="sm"),
                    html.Div(id="trade-status-badge", style={"marginTop": "6px", "fontSize": "13px", "fontWeight": "bold"}),
                ], style=CARD_STYLE),
            ], width=12),
        ], className="g-2", style={"marginBottom": "8px"}),

        dbc.Row([
            dbc.Col(make_stat_card("SIGNAL", "sig-display", "", NEUTRAL_COLOR, large=True), width=2),
            dbc.Col(make_stat_card("Z-SCORE", "zscore-display", "σ", "#e0e0e0", large=True), width=2),
            dbc.Col(make_stat_card("HURST", "hurst-display", "", "#e0e0e0", large=True), width=2),
            dbc.Col(make_stat_card("REGIME", "regime-display", "", "#e0e0e0", large=True), width=2),
            dbc.Col(make_stat_card("BID", "bid-display", "", "#e0e0e0", large=True), width=2),
            dbc.Col(make_stat_card("SPREAD", "spread-display", "", "#e0e0e0"), width=2),
        ], className="g-2", style={"marginBottom": "12px"}),

        dbc.Row([
            dbc.Col(make_stat_card("P&L", "pnl-display", "$", POSITIVE_COLOR), width=2),
            dbc.Col(make_stat_card("EQUITY", "equity-display", "$", "#e0e0e0"), width=2),
            dbc.Col(make_stat_card("POSITION", "position-display", "", NEUTRAL_COLOR), width=3),
            dbc.Col(make_stat_card("POS P&L", "pospnl-display", "$", NEUTRAL_COLOR), width=3),
            dbc.Col(make_stat_card("ATR", "atr-display", "", "#e0e0e0"), width=2),
        ], className="g-2", style={"marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([
                html.Div("PRICE CHART", style={"fontSize": "12px", "color": "#888", "marginBottom": "4px"}),
                dcc.Graph(id="price-chart", config={"displayModeBar": False},
                          style={"height": "300px", "backgroundColor": "#16213e",
                                 "borderRadius": "8px", "border": "1px solid #0f3460"}),
            ], width=6),
            dbc.Col([
                html.Div("Z-SCORE & HURST TRACE", style={"fontSize": "12px", "color": "#888", "marginBottom": "4px"}),
                dcc.Graph(id="stats-chart", config={"displayModeBar": False},
                          style={"height": "300px", "backgroundColor": "#16213e",
                                 "borderRadius": "8px", "border": "1px solid #0f3460"}),
            ], width=6),
        ], className="g-2", style={"marginBottom": "12px"}),

        dbc.Row([
            dbc.Col([
                html.Div("EQUITY CURVE", style={"fontSize": "12px", "color": "#888", "marginBottom": "4px"}),
                dcc.Graph(id="equity-chart", config={"displayModeBar": False},
                          style={"height": "220px", "backgroundColor": "#16213e",
                                 "borderRadius": "8px", "border": "1px solid #0f3460"}),
            ], width=12),
        ], className="g-2"),

        html.Div([
            html.Div("SNAPSHOT SUBSCRIPTION", style={"fontSize": "12px", "color": "#888", "marginBottom": "4px", "marginTop": "10px"}),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="snapshot-time-dropdown",
                        options=[{"label": f"{h:02d}:{m:02d}", "value": f"{h:02d}:{m:02d}"} 
                                 for h in range(24) for m in [0, 15, 30, 45]],
                        placeholder="Select time (HH:MM)",
                        style={"width": "150px", "display": "inline-block", "marginRight": "10px"},
                        clearable=False,
                    ),
                    dbc.Button("SUBSCRIBE", id="btn-subscribe", color="primary", size="sm",
                               style={"width": "120px", "fontWeight": "bold", "marginRight": "10px"}),
                    dbc.Button("UNSUBSCRIBE", id="btn-unsubscribe", color="secondary", size="sm",
                               style={"width": "120px", "fontWeight": "bold"}),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
                html.Div(id="snapshot-status", style={"fontSize": "13px", "color": "#4ecca3", "marginBottom": "8px"}),
                html.Div(id="snapshot-list", style={"fontSize": "12px", "color": "#e0e0e0"}),
                html.Div("Max 10 snapshots/day | Separate from hourly snapshots", 
                         style={"fontSize": "10px", "color": "#666", "marginTop": "8px"}),
            ], style={"backgroundColor": "#16213e", "padding": "12px", 
                       "borderRadius": "8px", "border": "1px solid #0f3460"}),
        ]),

        html.Div([
            html.Div("TRADE LOG", style={"fontSize": "12px", "color": "#888", "marginBottom": "4px", "marginTop": "10px"}),
            html.Div(id="trade-log", style={"fontSize": "11px", "maxHeight": "150px", "overflowY": "auto",
                                            "backgroundColor": "#16213e", "padding": "8px",
                                            "borderRadius": "8px", "border": "1px solid #0f3460"}),
        ]),

        html.Div(id="debug-store", style={"display": "none"}),
    ], style=CONTENT_STYLE)
