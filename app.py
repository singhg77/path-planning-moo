"""
Dash Web Application — Multi-Objective Aircraft Path Planning
=============================================================
Browser-based port of the matplotlib interactive visualisation from
weather_path_planning.py.  Run with:

    python app.py

then open http://localhost:8050 in a browser.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

# ── Import domain module ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from weather_path_planning import (
    build_path,
    evaluate_path,
    _best_idx,
    _run_nsga2,
    WEATHER,
    SOURCE,
    DEST,
    N_WP,
    SAFE_AGL,
    DETECT_AGL,
    MAX_AGL,
    FUEL_OPT_AGL,
    _path_along_track,
    _path_agls,
    _path_2d_length,
    _weather_exposure,
    _WEATHER_ORIG,
    TERRAIN,
    X_BOUNDS,
    Y_BOUNDS,
)

# ── Gunicorn entry point ───────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Weather Path Planning",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # for gunicorn

# ── Colours / theme ───────────────────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_PLOT   = "#0d0d23"
BG_PANEL  = "#12122a"
CYAN      = "#00e5ff"
WHITE     = "#ffffff"
LGRAY     = "#cccccc"
GRID_COL  = "rgba(255,255,255,0.12)"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_PLOT,
    font=dict(color=LGRAY, size=11),
    margin=dict(l=50, r=20, t=40, b=40),
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _crowd_sizes(Fm: np.ndarray, smin: int = 6, smax: int = 18) -> np.ndarray:
    """Return per-solution marker sizes based on crowding distance."""
    try:
        from pymoo.operators.survival.rank_and_crowding.metrics import (
            calc_crowding_distance,
        )
        cd = calc_crowding_distance(Fm)
    except Exception:
        return np.full(len(Fm), (smin + smax) / 2)
    cd = np.clip(cd, 0, np.percentile(cd[np.isfinite(cd)], 95) if np.any(np.isfinite(cd)) else 1)
    cd = np.where(np.isfinite(cd), cd, np.nanmax(cd) if np.any(np.isfinite(cd)) else 1)
    lo, hi = cd.min(), cd.max()
    if hi == lo:
        return np.full(len(Fm), (smin + smax) / 2)
    return smin + (smax - smin) * (cd - lo) / (hi - lo)


def _fw(F: np.ndarray, col: int) -> np.ndarray:
    """Display transform: time column → minutes, others raw."""
    return F[:, col] / 60.0 if col == 0 else F[:, col]


def _extreme_idx(Fm: np.ndarray):
    return [int(Fm[:, c].argmin()) for c in range(3)]


# ── Load / compute initial Pareto front ───────────────────────────────────────
_NPZ_PATH = os.path.join(os.path.dirname(__file__), "path_results.npz")

def _load_or_run() -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(_NPZ_PATH):
        d = np.load(_NPZ_PATH)
        return d["F"], d["X"]
    print("path_results.npz not found — running NSGA-II (this may take a minute) …")
    F, X = _run_nsga2(pop=120, ngen=100)
    np.savez_compressed(_NPZ_PATH, F=F, X=X)
    return F, X


_INIT_F, _INIT_X = _load_or_run()

# Pre-build all paths once at startup for fast map rendering
_INIT_PATHS = [build_path(_INIT_X[j].reshape(N_WP, 3)) for j in range(len(_INIT_X))]

# ── Default storm state (serialisable) ────────────────────────────────────────
def _storm_default():
    return [
        {"center": list(s["center"].tolist()), "radius": float(s["radius"]), "name": s["name"]}
        for s in WEATHER
    ]


# ── Figure builders ───────────────────────────────────────────────────────────

def build_map_fig(
    F: np.ndarray,
    X: np.ndarray,
    active_idx: int,
    storms: list,
    paths=None,
) -> go.Figure:
    """2-D top-down map with all paths (faint), active path (bright), storms."""
    if paths is None:
        paths = [build_path(X[j].reshape(N_WP, 3)) for j in range(len(X))]

    active_path = paths[active_idx]

    fig = go.Figure()
    fig.update_layout(**LAYOUT_BASE)
    fig.update_layout(
        title=dict(text="Top-Down Mission Map", font=dict(color=WHITE, size=13)),
        xaxis=dict(
            title="Easting (km)",
            range=list(X_BOUNDS),
            gridcolor=GRID_COL,
            zerolinecolor=GRID_COL,
        ),
        yaxis=dict(
            title="Northing (km)",
            range=list(Y_BOUNDS),
            scaleanchor="x",
            scaleratio=1,
            gridcolor=GRID_COL,
            zerolinecolor=GRID_COL,
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30,30,60,0.85)",
            bordercolor="#445",
            font=dict(size=10, color=LGRAY),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=55, r=15, t=45, b=45),
        uirevision="map-fixed",
    )

    # ── Terrain contour via heatmap ────────────────────────────────────────────
    xg, yg, elev = TERRAIN.grid()
    fig.add_trace(
        go.Heatmap(
            x=xg,
            y=yg,
            z=elev,
            colorscale="Viridis",
            opacity=0.45,
            showscale=False,
            hoverinfo="skip",
            name="Terrain",
        )
    )

    # ── Direct route reference ─────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=[SOURCE[0], DEST[0]],
            y=[SOURCE[1], DEST[1]],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1.2),
            name="Direct route (ref)",
            hoverinfo="skip",
        )
    )

    # ── All Pareto paths (faint, one trace with None separators) ──────────────
    xs_all, ys_all = [], []
    for p in paths:
        xs_all.extend(list(p[:, 0]) + [None])
        ys_all.extend(list(p[:, 1]) + [None])

    fig.add_trace(
        go.Scatter(
            x=xs_all,
            y=ys_all,
            mode="lines",
            line=dict(color="rgba(100,120,200,0.18)", width=1.0),
            name="Pareto paths",
            hoverinfo="skip",
        )
    )

    # ── Active path ───────────────────────────────────────────────────────────
    wps = X[active_idx].reshape(N_WP, 3)
    fig.add_trace(
        go.Scatter(
            x=active_path[:, 0],
            y=active_path[:, 1],
            mode="lines+markers",
            line=dict(color=CYAN, width=2.8),
            marker=dict(size=6, color=CYAN, line=dict(color=WHITE, width=0.7)),
            name="Active path",
        )
    )

    # ── Source / Destination ──────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=[SOURCE[0]],
            y=[SOURCE[1]],
            mode="markers+text",
            marker=dict(symbol="square", size=14, color="lime",
                        line=dict(color="lime", width=1.5)),
            text=[f"SOURCE<br>({SOURCE[2]:.0f} m AGL)"],
            textposition="bottom right",
            textfont=dict(color="lime", size=9),
            name="Source",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[DEST[0]],
            y=[DEST[1]],
            mode="markers+text",
            marker=dict(symbol="star", size=16, color="#ff6666",
                        line=dict(color="#ff8888", width=1.0)),
            text=[f"DEST<br>({DEST[2]:.0f} m AGL)"],
            textposition="bottom left",
            textfont=dict(color="#ff8888", size=9),
            name="Destination",
        )
    )

    # ── Storm circles via shapes ───────────────────────────────────────────────
    shapes = []
    annotations = []
    for storm in storms:
        cx, cy = storm["center"]
        r = storm["radius"]
        shapes.append(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=cx - r,
                y0=cy - r,
                x1=cx + r,
                y1=cy + r,
                fillcolor="rgba(102,136,255,0.22)",
                line=dict(color="#aabbff", width=2.0, dash="dash"),
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=cx,
                y=cy,
                text=f"<b>{storm['name']}</b>",
                showarrow=False,
                font=dict(color=WHITE, size=9),
                bgcolor="rgba(0,0,0,0)",
            )
        )

    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig


def build_alt_fig(path: np.ndarray) -> go.Figure:
    """Altitude AGL profile with band shading."""
    dists = _path_along_track(path)
    agls  = _path_agls(path)
    x_max = float(dists[-1]) * 1.05

    fig = go.Figure()
    fig.update_layout(**LAYOUT_BASE)
    fig.update_layout(
        title=dict(text="Altitude AGL Profile", font=dict(color=WHITE, size=12)),
        xaxis=dict(
            title="Along-Track Distance (km)",
            range=[0, x_max],
            gridcolor=GRID_COL,
        ),
        yaxis=dict(
            title="Altitude AGL (m)",
            range=[0, MAX_AGL + 60],
            gridcolor=GRID_COL,
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30,30,60,0.85)",
            bordercolor="#445",
            font=dict(size=9, color=LGRAY),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=55, r=15, t=40, b=45),
        uirevision="alt-fixed",
    )

    # Terrain risk band
    fig.add_hrect(
        y0=0, y1=SAFE_AGL,
        fillcolor="rgba(255,60,60,0.12)",
        line_width=0,
        annotation_text=f"Terrain risk (<{SAFE_AGL:.0f} m)",
        annotation_font_color="#ff8888",
        annotation_font_size=8,
        annotation_position="right",
    )
    # Safe band
    fig.add_hrect(
        y0=SAFE_AGL, y1=DETECT_AGL,
        fillcolor="rgba(0,200,100,0.07)",
        line_width=0,
        annotation_text=f"Ideal ({SAFE_AGL:.0f}–{DETECT_AGL:.0f} m)",
        annotation_font_color="#66ffaa",
        annotation_font_size=8,
        annotation_position="right",
    )
    # Radar detect band
    fig.add_hrect(
        y0=DETECT_AGL, y1=MAX_AGL + 60,
        fillcolor="rgba(255,165,0,0.08)",
        line_width=0,
        annotation_text=f"Radar detect (>{DETECT_AGL:.0f} m)",
        annotation_font_color="orange",
        annotation_font_size=8,
        annotation_position="right",
    )
    # Reference lines
    fig.add_hline(y=SAFE_AGL,     line=dict(color="rgba(255,80,80,0.7)",   width=1.2, dash="dash"))
    fig.add_hline(y=DETECT_AGL,   line=dict(color="rgba(255,165,0,0.7)",   width=1.2, dash="dash"))
    fig.add_hline(y=FUEL_OPT_AGL, line=dict(color="rgba(255,215,0,0.85)",  width=1.1, dash="dashdot"))

    # Profile fill
    fig.add_trace(
        go.Scatter(
            x=dists,
            y=agls,
            mode="lines+markers",
            line=dict(color=CYAN, width=2.4),
            marker=dict(size=5, color=CYAN, line=dict(color=WHITE, width=0.6)),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.08)",
            name="AGL profile",
        )
    )

    return fig


def build_scatter_fig(F: np.ndarray, active_idx: int) -> go.Figure:
    """3 pairwise scatter subplots with crowding-distance sizes and extreme markers."""
    PW_PAIRS = [
        # (x_col, y_col, c_col, x_label, y_label, c_label, colorscale)
        (0, 1, 2, "Time (min)", "Risk",      "Fuel (kg)", "Plasma"),
        (1, 2, 0, "Risk",       "Fuel (kg)", "Time (min)", "Viridis"),
        (0, 2, 1, "Time (min)", "Fuel (kg)", "Risk",       "RdYlGn"),
    ]
    EXTREME_SYMBOLS = ["triangle-up", "diamond", "square"]
    EXTREME_COLORS  = ["#ff6b6b", "#6bffb8", "#ffd93d"]
    EXTREME_LABELS  = ["Best Time", "Best Risk", "Best Fuel"]

    sizes   = _crowd_sizes(F)
    ext_idx = _extreme_idx(F)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Time × Risk", "Risk × Fuel", "Time × Fuel"],
        horizontal_spacing=0.10,
    )
    fig.update_layout(**LAYOUT_BASE)
    fig.update_layout(
        title=dict(
            text="Pareto Front  ▲=BestTime  ◆=BestRisk  ■=BestFuel  ★=Active  (size ∝ uniqueness)",
            font=dict(color=WHITE, size=11),
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30,30,60,0.85)",
            bordercolor="#445",
            font=dict(size=9, color=LGRAY),
            x=0.01,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            orientation="h",
        ),
        margin=dict(l=50, r=20, t=60, b=80),
        uirevision="scatter-fixed",
    )
    # Dark backgrounds for each subplot
    for i in range(1, 4):
        fig.update_layout(**{f"plot_bgcolor": BG_PLOT})

    for col_i, (xc, yc, cc, xl, yl, cl, cscale) in enumerate(PW_PAIRS):
        xv = _fw(F, xc)
        yv = _fw(F, yc)
        cv = _fw(F, cc)

        # Main Pareto cloud
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=yv,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=cv,
                    colorscale=cscale,
                    showscale=(col_i == 0),
                    colorbar=dict(
                        title=dict(text=cl, font=dict(color=LGRAY, size=9)),
                        tickfont=dict(color=LGRAY, size=8),
                        len=0.6,
                        x=1.02,
                    ) if col_i == 0 else None,
                    opacity=0.72,
                    line=dict(width=0),
                ),
                name="Pareto" if col_i == 0 else None,
                showlegend=(col_i == 0),
                hovertemplate=f"{xl}: %{{x:.2f}}<br>{yl}: %{{y:.2f}}<br>{cl}: %{{marker.color:.2f}}<extra></extra>",
            ),
            row=1,
            col=col_i + 1,
        )

        # Extreme markers
        for ei, (sym, ec, elb) in enumerate(
                zip(EXTREME_SYMBOLS, EXTREME_COLORS, EXTREME_LABELS)):
            eidx = ext_idx[ei]
            fig.add_trace(
                go.Scatter(
                    x=[xv[eidx]],
                    y=[yv[eidx]],
                    mode="markers",
                    marker=dict(
                        symbol=sym,
                        size=13,
                        color=ec,
                        line=dict(color=WHITE, width=0.8),
                    ),
                    name=elb if col_i == 0 else None,
                    showlegend=(col_i == 0),
                    hovertemplate=f"{elb}<br>{xl}: %{{x:.2f}}<br>{yl}: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=col_i + 1,
            )

        # Active star
        fig.add_trace(
            go.Scatter(
                x=[xv[active_idx]],
                y=[yv[active_idx]],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=18,
                    color=CYAN,
                    line=dict(color=WHITE, width=0.9),
                ),
                name="Active ★" if col_i == 0 else None,
                showlegend=(col_i == 0),
                hovertemplate=f"Active<br>{xl}: %{{x:.2f}}<br>{yl}: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=col_i + 1,
        )

        # Axis labels per subplot
        fig.update_xaxes(
            title_text=xl,
            title_font=dict(size=9, color=LGRAY),
            tickfont=dict(size=8, color=LGRAY),
            gridcolor=GRID_COL,
            row=1,
            col=col_i + 1,
        )
        fig.update_yaxes(
            title_text=yl,
            title_font=dict(size=9, color=LGRAY),
            tickfont=dict(size=8, color=LGRAY),
            gridcolor=GRID_COL,
            row=1,
            col=col_i + 1,
        )

    return fig


def build_pcp_fig(F: np.ndarray, active_idx: int) -> go.Figure:
    """Parallel coordinates plot (all 3 objectives, normalised 0-1)."""
    lo = F.min(axis=0)
    hi = F.max(axis=0)
    hi = np.where(hi == lo, lo + 1e-9, hi)
    Fn = (F - lo) / (hi - lo)

    # Display values for tooltips / labels
    time_min = F[:, 0] / 60.0
    risk     = F[:, 1]
    fuel     = F[:, 2]

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=Fn[:, 1],          # colour by normalised risk
                colorscale="Plasma",
                showscale=True,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(text="Risk (norm.)", font=dict(color=LGRAY, size=9)),
                    tickfont=dict(color=LGRAY, size=8),
                    len=0.7,
                ),
            ),
            dimensions=[
                dict(
                    label="Time (min)",
                    values=time_min,
                    range=[float(time_min.min()), float(time_min.max())],
                ),
                dict(
                    label="Survival Risk",
                    values=risk,
                    range=[float(risk.min()), float(risk.max())],
                ),
                dict(
                    label="Fuel (kg)",
                    values=fuel,
                    range=[float(fuel.min()), float(fuel.max())],
                ),
            ],
        )
    )
    fig.update_layout(**LAYOUT_BASE)
    fig.update_layout(
        title=dict(
            text="Parallel Coordinates — each line = one Pareto solution",
            font=dict(color=WHITE, size=11),
        ),
        margin=dict(l=60, r=80, t=50, b=30),
        uirevision="pcp-fixed",
    )
    return fig


# ── Storm card helper ─────────────────────────────────────────────────────────

def _storm_card(i: int, storm: dict) -> html.Div:
    cx, cy = storm["center"]
    return html.Div(
        style={
            "background": "#1e1e3a",
            "border": "1px solid #445",
            "borderRadius": "6px",
            "padding": "8px 12px",
            "flex": "1",
            "minWidth": "180px",
        },
        children=[
            html.Div(
                storm["name"],
                style={"color": "#aabbff", "fontWeight": "bold", "marginBottom": "6px", "fontSize": "11px"},
            ),
            html.Div(
                style={"display": "flex", "gap": "8px", "alignItems": "center"},
                children=[
                    html.Label("X:", style={"color": LGRAY, "fontSize": "10px", "minWidth": "14px"}),
                    dcc.Input(
                        id={"type": "storm-x", "index": i},
                        type="number",
                        value=round(cx, 1),
                        step=0.5,
                        min=float(X_BOUNDS[0]),
                        max=float(X_BOUNDS[1]),
                        style={
                            "width": "70px",
                            "background": "#0d0d23",
                            "color": WHITE,
                            "border": "1px solid #445",
                            "borderRadius": "4px",
                            "padding": "2px 4px",
                            "fontSize": "11px",
                        },
                        debounce=True,
                    ),
                    html.Label("Y:", style={"color": LGRAY, "fontSize": "10px", "minWidth": "14px"}),
                    dcc.Input(
                        id={"type": "storm-y", "index": i},
                        type="number",
                        value=round(cy, 1),
                        step=0.5,
                        min=float(Y_BOUNDS[0]),
                        max=float(Y_BOUNDS[1]),
                        style={
                            "width": "70px",
                            "background": "#0d0d23",
                            "color": WHITE,
                            "border": "1px solid #445",
                            "borderRadius": "4px",
                            "padding": "2px 4px",
                            "fontSize": "11px",
                        },
                        debounce=True,
                    ),
                    html.Span(
                        f"r={storm['radius']} km",
                        style={"color": "#667799", "fontSize": "10px"},
                    ),
                ],
            ),
        ],
    )


# ── Slider factory ────────────────────────────────────────────────────────────

def _slider_row(slider_id: str, label: str, color: str) -> html.Div:
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "10px", "padding": "2px 0"},
        children=[
            html.Span(
                label,
                style={"color": color, "fontWeight": "bold", "fontSize": "12px", "minWidth": "75px"},
            ),
            html.Div(
                dcc.Slider(
                    id=slider_id,
                    min=0,
                    max=1,
                    step=0.01,
                    value=round(1 / 3, 2),
                    marks={0: {"label": "0", "style": {"color": LGRAY}},
                           0.5: {"label": "0.5", "style": {"color": LGRAY}},
                           1: {"label": "1", "style": {"color": LGRAY}}},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="drag",
                    className="dash-slider",
                ),
                style={"flex": "1"},
            ),
        ],
    )


# ── Layout ────────────────────────────────────────────────────────────────────

_init_storms = _storm_default()
_init_idx    = _best_idx(_INIT_F, np.array([1 / 3, 1 / 3, 1 / 3]))
_init_path   = build_path(_INIT_X[_init_idx].reshape(N_WP, 3))

app.layout = html.Div(
    style={
        "background": BG_DARK,
        "minHeight": "100vh",
        "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
        "color": LGRAY,
    },
    children=[
        # ── dcc.Store ────────────────────────────────────────────────────────
        dcc.Store(
            id="pareto-store",
            data={"F": _INIT_F.tolist(), "X": _INIT_X.tolist()},
        ),
        dcc.Store(
            id="storm-store",
            data=_init_storms,
        ),

        # ── Top bar ───────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": "#0d0d23",
                "borderBottom": "1px solid #2a2a4a",
                "padding": "10px 20px",
                "display": "flex",
                "alignItems": "center",
                "gap": "18px",
            },
            children=[
                html.H1(
                    "Multi-Objective Aircraft Path Planning",
                    style={"color": WHITE, "fontSize": "18px", "margin": "0", "fontWeight": "700"},
                ),
                html.Span(
                    "Weather Zone Avoidance · NSGA-II Pareto Optimisation",
                    style={"color": "#6688cc", "fontSize": "12px"},
                ),
                html.Div(style={"flex": "1"}),
                dcc.Loading(
                    id="loading-reopt",
                    type="circle",
                    color=CYAN,
                    children=[
                        html.Button(
                            "Re-optimize",
                            id="reoptimize-btn",
                            n_clicks=0,
                            style={
                                "background": "#1e3a1e",
                                "color": "#88ff88",
                                "border": "1px solid #44aa44",
                                "borderRadius": "6px",
                                "padding": "7px 16px",
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "fontSize": "12px",
                            },
                        ),
                        html.Div(id="reopt-status", style={"color": "#88ff88", "fontSize": "11px", "display": "inline"}),
                    ],
                ),
            ],
        ),

        # ── Main content ──────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "0", "padding": "10px 12px", "minHeight": "calc(100vh - 160px)"},
            children=[
                # Left column — 2D map (40%)
                html.Div(
                    style={"width": "40%", "display": "flex", "flexDirection": "column", "gap": "8px"},
                    children=[
                        dcc.Graph(
                            id="map-fig",
                            figure=build_map_fig(_INIT_F, _INIT_X, _init_idx, _init_storms, _INIT_PATHS),
                            style={"height": "430px"},
                            config={"displayModeBar": False},
                        ),
                        # Storm repositioning cards
                        html.Div(
                            style={
                                "background": BG_PANEL,
                                "border": "1px solid #2a2a4a",
                                "borderRadius": "8px",
                                "padding": "10px 12px",
                            },
                            children=[
                                html.Div(
                                    "Storm Positions",
                                    style={"color": "#aabbff", "fontWeight": "bold", "fontSize": "12px", "marginBottom": "8px"},
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                                    children=[_storm_card(i, s) for i, s in enumerate(_init_storms)],
                                ),
                                html.Button(
                                    "Reset Storms",
                                    id="reset-storms-btn",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "8px",
                                        "background": "#1a1a3a",
                                        "color": LGRAY,
                                        "border": "1px solid #445",
                                        "borderRadius": "5px",
                                        "padding": "5px 12px",
                                        "cursor": "pointer",
                                        "fontSize": "11px",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),

                # Right column — charts (60%)
                html.Div(
                    style={"width": "60%", "display": "flex", "flexDirection": "column", "gap": "8px", "paddingLeft": "10px"},
                    children=[
                        # Altitude profile
                        dcc.Graph(
                            id="alt-fig",
                            figure=build_alt_fig(_init_path),
                            style={"height": "220px"},
                            config={"displayModeBar": False},
                        ),
                        # Pairwise scatter
                        dcc.Graph(
                            id="scatter-fig",
                            figure=build_scatter_fig(_INIT_F, _init_idx),
                            style={"height": "260px"},
                            config={"displayModeBar": False},
                        ),
                        # Parallel coordinates
                        dcc.Graph(
                            id="pcp-fig",
                            figure=build_pcp_fig(_INIT_F, _init_idx),
                            style={"height": "220px"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),

        # ── Sliders ───────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": BG_PANEL,
                "borderTop": "1px solid #2a2a4a",
                "padding": "10px 30px",
            },
            children=[
                html.Div(
                    "Objective Weights",
                    style={"color": WHITE, "fontWeight": "bold", "fontSize": "12px", "marginBottom": "6px"},
                ),
                _slider_row("time-weight", "TIME",  "#ff6b6b"),
                _slider_row("risk-weight", "RISK",  "#6bffb8"),
                _slider_row("fuel-weight", "FUEL",  "#ffd93d"),
            ],
        ),

        # ── Status bar ────────────────────────────────────────────────────────
        html.Div(
            id="status-text",
            style={
                "background": "#0d0d1a",
                "borderTop": "1px solid #2a2a4a",
                "padding": "8px 20px",
                "fontFamily": "monospace",
                "fontSize": "11px",
                "color": WHITE,
                "minHeight": "30px",
            },
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("storm-store", "data"),
    [
        Input({"type": "storm-x", "index": dash.ALL}, "value"),
        Input({"type": "storm-y", "index": dash.ALL}, "value"),
        Input("reset-storms-btn", "n_clicks"),
    ],
    State("storm-store", "data"),
    prevent_initial_call=True,
)
def update_storm_store(xs, ys, reset_clicks, current_storms):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"]
    if "reset-storms-btn" in trigger_id:
        return _storm_default()

    storms = []
    for i, s in enumerate(current_storms):
        cx = xs[i] if xs[i] is not None else s["center"][0]
        cy = ys[i] if ys[i] is not None else s["center"][1]
        storms.append({
            "center": [float(cx), float(cy)],
            "radius": s["radius"],
            "name":   s["name"],
        })
    return storms


@app.callback(
    [
        Output("map-fig",    "figure"),
        Output("alt-fig",    "figure"),
        Output("scatter-fig","figure"),
        Output("pcp-fig",    "figure"),
        Output("status-text","children"),
    ],
    [
        Input("time-weight",  "value"),
        Input("risk-weight",  "value"),
        Input("fuel-weight",  "value"),
        Input("pareto-store", "data"),
        Input("storm-store",  "data"),
    ],
)
def update_all(tw, rw, fw, pareto_data, storms):
    # Parse stored Pareto front
    F = np.array(pareto_data["F"])
    X = np.array(pareto_data["X"])

    # Update WEATHER module-level state to reflect current storm positions
    for i, s in enumerate(storms):
        WEATHER[i]["center"] = np.array(s["center"])
        WEATHER[i]["radius"] = s["radius"]

    # Re-evaluate risk column with current storm positions
    # (only if storms changed — safe to always do since it's fast)
    paths = [build_path(X[j].reshape(N_WP, 3)) for j in range(len(X))]
    for j, p in enumerate(paths):
        _, new_risk, _ = evaluate_path(p)
        F[j, 1] = new_risk

    w = np.array([tw or 1/3, rw or 1/3, fw or 1/3])
    active_idx = _best_idx(F, w)
    active_path = paths[active_idx]

    # Metrics
    t_min   = F[active_idx, 0] / 60.0
    risk    = F[active_idx, 1]
    fuel    = F[active_idx, 2]
    plen    = _path_2d_length(active_path)
    direct  = float(np.linalg.norm(DEST[:2] - SOURCE[:2]))
    detour  = plen - direct
    agls    = _path_agls(active_path)
    wx_exp  = _weather_exposure(active_path)
    wn      = w / max(w.sum(), 1e-9)
    min_risk = F[:, 1].min()

    status = (
        f"Mission time: {t_min:.1f} min  |  "
        f"Survival risk: {risk:.1f}  (Pareto min={min_risk:.1f})  |  "
        f"Fuel burn: {fuel:.1f} kg  |  "
        f"Path: {plen:.0f} km  (+{detour:.0f} km detour)  |  "
        f"AGL range: {agls.min():.0f}–{agls.max():.0f} m  |  "
        f"Weather exposure: {wx_exp:.0f} s  "
        f"[T:{wn[0]:.2f}  R:{wn[1]:.2f}  F:{wn[2]:.2f}]"
    )

    map_fig     = build_map_fig(F, X, active_idx, storms, paths)
    alt_fig     = build_alt_fig(active_path)
    scatter_fig = build_scatter_fig(F, active_idx)
    pcp_fig     = build_pcp_fig(F, active_idx)

    return map_fig, alt_fig, scatter_fig, pcp_fig, status


@app.callback(
    [Output("pareto-store", "data"),
     Output("reopt-status", "children")],
    Input("reoptimize-btn", "n_clicks"),
    State("storm-store", "data"),
    prevent_initial_call=True,
)
def reoptimize(n_clicks, storms):
    if not n_clicks:
        raise PreventUpdate

    # Apply current storm positions before running optimiser
    for i, s in enumerate(storms):
        WEATHER[i]["center"] = np.array(s["center"])
        WEATHER[i]["radius"] = s["radius"]

    try:
        new_F, new_X = _run_nsga2(pop=120, ngen=100)
    except RuntimeError:
        raise PreventUpdate  # pymoo unavailable — keep current front
    return (
        {"F": new_F.tolist(), "X": new_X.tolist()},
        f" Done — {len(new_F)} Pareto solutions",
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
