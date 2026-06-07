# -*- coding: utf-8 -*-
"""
Failure Mode Classification + Dual-Layer SPC MLOps — real-time simulation dashboard.
Run:  streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

import pdm as P

# ----------------------------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance MLOps", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
#MainMenu{visibility:hidden}
footer{visibility:hidden}
/* keep the header/toolbar (which holds the sidebar expand arrow) usable */
[data-testid='stHeader']{background:transparent}
[data-testid='stToolbarActions']{visibility:hidden}
[data-testid='stSidebar']{visibility:visible !important}
[data-testid='stSidebarCollapsedControl']{visibility:visible !important; display:flex !important; left:0.5rem}
[data-testid='stSidebarCollapseButton']{visibility:visible !important; display:flex !important}
</style>""", unsafe_allow_html=True)

# Colors
NAVY="#1e293b"; BLUE="#2563eb"; AMBER="#d97706"; PURPLE="#7c3aed"
GREEN="#059669"; RED="#dc2626"; GRAY="#9ca3af"
BAND="#ecfdf5"   # in-control zone fill (faint green)
GRID="#eef2f7"; AXIS="#cbd5e1"

# Shared Plotly theme — one professional template applied to every chart so the
# whole dashboard reads as a single product (clean sans font, framed axes, soft
# grid, consistent hover). Individual figures only set what is chart-specific.
FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
pio.templates["pdm"] = go.layout.Template(layout=dict(
    font=dict(family=FONT_STACK, size=12, color=NAVY),
    plot_bgcolor="white", paper_bgcolor="white",
    colorway=[BLUE, PURPLE, AMBER, GREEN, RED, GRAY],
    margin=dict(t=46, b=28, l=14, r=14),
    title=dict(font=dict(size=13.5, color=NAVY), x=0.01, xanchor="left"),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS,
               ticks="outside", tickcolor=AXIS, ticklen=4, automargin=True),
    yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS,
               ticks="outside", tickcolor=AXIS, ticklen=4, automargin=True),
    legend=dict(font=dict(size=11.5), bgcolor="rgba(255,255,255,0)"),
    hoverlabel=dict(bgcolor="white", bordercolor=AXIS,
                    font=dict(family=FONT_STACK, size=12, color=NAVY)),
))
pio.templates.default = "pdm"

ROLL = 400  # rolling-metric window

# English display labels (config keeps Korean; the dashboard is English-only)
SENSOR_EN = {
    "Air temperature [K]": "Air Temp [K]",
    "Process temperature [K]": "Process Temp [K]",
    "Rotational speed [rpm]": "Rotational Speed [rpm]",
    "Torque [Nm]": "Torque [Nm]",
    "Tool wear [min]": "Tool Wear [min]",
}
MODE_EN = {
    "Normal": "Normal",
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}
# Model input feature -> display name (for feature-importance labels)
FEAT_EN = {
    "type_code": "Product Type (L/M/H)",
    "temp_diff": "Temp Diff (Process - Air)",
    "power": "Power (Torque x Speed)",
    **SENSOR_EN,
}

# Subtle styling — tone down metric cards, improve table readability
st.markdown("""
<style>
[data-testid="stMetricValue"]{font-size:1.5rem}
[data-testid="stMetricLabel"]{opacity:.75}
.block-container{padding-top:2.4rem}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models and preparing simulation...")
def get_sim():
    sim = P.run_simulation()
    y = sim["y_all"]
    # Rolling failure-recall / accuracy via the shared metrics helper (single source of truth)
    sim["rec_live"], sim["acc_live"] = P.rolling_recall_accuracy(y, sim["pred_live"], ROLL)
    sim["rec_nore"], sim["acc_nore"] = P.rolling_recall_accuracy(y, sim["pred_noretrain"], ROLL)
    return sim


sim = get_sim()
stream = sim["stream"]
N = len(stream)
limits = sim["limits"]
flags = sim["flags"]
psi = sim["psi_series"]
tstar = sim["retrain_t"]
promotion = sim.get("promotion")
promoted = bool(promotion and promotion["promoted"])


def phase_at(t):
    if t < P.N_BASELINE: return "Baseline", BLUE
    if t < P.N_STABLE_END: return "Stable", GREEN
    return "Drift", RED


# ----------------------------------------------------------------------------
# Sidebar — simulation controls
# ----------------------------------------------------------------------------
st.sidebar.title("Simulation Controls")
if "t" not in st.session_state:
    try:
        st.session_state.t = int(st.query_params.get("t", 3500))
    except Exception:
        st.session_state.t = 3500

st.sidebar.markdown("**Current time (stream index)**")
t = st.sidebar.slider("t", 500, N, st.session_state.t, step=50, label_visibility="collapsed")
st.session_state.t = t

colA, colB = st.sidebar.columns(2)
auto = colA.toggle("Auto-play")
speed = colB.select_slider("Speed", ["Slow", "Normal", "Fast"], value="Normal")
step = {"Slow": 100, "Normal": 250, "Fast": 500}[speed]

st.sidebar.divider()
st.sidebar.markdown(f"""
**Scenario phases**
- Baseline 0 - {P.N_BASELINE:,} : fit SPC limits + train RF v1
- Stable {P.N_BASELINE:,} - {P.N_STABLE_END:,}
- Drift {P.DRIFT_START:,} - {N:,} : inject +{P.TORQUE_SHIFT_MAX:.0f}Nm torque

**Retrain trigger**: t* = {tstar:,} (PSI >= {P.PSI_THRESHOLD})
""")
st.sidebar.caption("Drag the slider to move through time. Entering the drift phase activates SPC, PSI, and automatic retraining in sequence.")


# ----------------------------------------------------------------------------
# Top status bar
# ----------------------------------------------------------------------------
pname, pcol = phase_at(t)
live_v2 = bool(tstar and t >= tstar and promoted)   # v2 in production only after validation
model_ver = "v2 (retrained, validated)" if live_v2 else "v1 (initial)"
cur_psi = psi[t-1]
spc_on = bool(flags["any_ooc"].values[t-1])
# Alarm rate over the recent ROLL window
recent_alarm = flags["any_ooc"].values[max(0, t-ROLL):t].mean() * 100

st.title("Failure Mode Classification + Dual-Layer SPC MLOps")
st.caption("Built on the AI4I 2020 sensor stream. Real-time SPC monitoring, Random Forest failure diagnosis, and automatic retraining on drift detection.")

m = st.columns(5)
m[0].metric("Current time", f"{t:,} / {N:,}")
m[1].metric("Phase", pname)
m[2].metric("Production model", model_ver)
m[3].metric("SPC status", "Alarm" if spc_on else "Normal", f"Recent alarm rate {recent_alarm:.1f}%")
m[4].metric("Current PSI (Torque)", f"{cur_psi:.3f}",
            "Above threshold" if cur_psi >= P.PSI_THRESHOLD else "Stable",
            delta_color="inverse")

if tstar and t >= tstar:
    st.success(f"Retraining complete. At t={tstar:,}, PSI exceeded {P.PSI_THRESHOLD} (drift detected), so "
               f"RF v2 was trained on the accumulated data, validated, and deployed. Failure-detection "
               f"performance has recovered.")
elif t >= P.DRIFT_START:
    st.warning(f"Drift in progress. The torque distribution is shifting away from baseline. "
               f"Automatic retraining triggers once PSI reaches {P.PSI_THRESHOLD} (at t*={tstar:,}).")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Real-time Monitoring (SPC)", "Failure Diagnosis (Random Forest)",
     "Drift & Auto-Retraining", "Business Value"])


# ----------------------------------------------------------------------------
# Tab 1 — SPC control charts
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Layer 1 - SPC Control Charts (Shewhart 3 sigma) - real-time monitoring of 5 sensors")
    win = 500
    lo = max(0, t - win)
    xs = np.arange(lo, t)
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[SENSOR_EN[c] for c in P.SENSORS] + ["Any-sensor OOC rate"],
                        vertical_spacing=0.14, horizontal_spacing=0.07)
    pos = [(1,1),(1,2),(1,3),(2,1),(2,2)]
    for (c,(r,cc)) in zip(P.SENSORS, pos):
        vals = stream[c].values[lo:t]
        lim = limits[c]
        ooc = (vals > lim["ucl"]) | (vals < lim["lcl"])
        label = SENSOR_EN[c]
        # In-control zone (mean +/- 3 sigma) as a soft band — standard SPC look
        fig.add_hrect(y0=lim["lcl"], y1=lim["ucl"], fillcolor=BAND,
                      opacity=0.9, line_width=0, layer="below", row=r, col=cc)
        fig.add_trace(go.Scatter(x=xs, y=vals, mode="lines", name=label,
                      line=dict(color=BLUE, width=1.3), showlegend=False,
                      hovertemplate="t=%{x}<br>"+label+"=%{y:.1f}<extra></extra>"), r, cc)
        fig.add_trace(go.Scatter(x=xs[ooc], y=vals[ooc], mode="markers",
                      marker=dict(color=RED, size=5.5, line=dict(color="white", width=0.6)),
                      showlegend=False, hovertemplate="t=%{x}<br>OOC %{y:.1f}<extra></extra>"), r, cc)
        fig.add_hline(y=lim["center"], line=dict(color=GRAY, width=1), row=r, col=cc)
        for yv in (lim["ucl"], lim["lcl"]):
            fig.add_hline(y=yv, line=dict(color=RED, width=1, dash="dash"), row=r, col=cc)
    # 6th cell: OOC rate trend
    ooc_roll = pd.Series(flags["any_ooc"].values).rolling(ROLL).mean().values
    fig.add_trace(go.Scatter(x=np.arange(lo,t), y=ooc_roll[lo:t]*100, mode="lines",
                  line=dict(color=AMBER, width=2), fill="tozeroy",
                  fillcolor="rgba(217,119,6,0.10)", showlegend=False,
                  hovertemplate="t=%{x}<br>OOC rate %{y:.1f}%<extra></extra>"), 2, 3)
    fig.update_yaxes(title_text="%", row=2, col=3)
    fig.update_layout(height=560, margin=dict(t=44,b=24,l=12,r=12), font=dict(size=11))
    st.plotly_chart(fig, width='stretch')
    c1, c2 = st.columns([2,3])
    c1.info("Red points are out-of-control (OOC) values beyond the control limits (mean +/- 3 sigma). "
            "In the drift phase, the torque chart rises and OOC points increase.")
    c2.caption("SPC plays two roles. First, it detects anomalies in real time without labels. "
               "Second, it monitors model input drift through shifts in the sensor distribution. "
               "When the OOC rate (bottom right) rises, PSI soon reaches its threshold as well.")


# ----------------------------------------------------------------------------
# Tab 2 — Random Forest diagnosis
# ----------------------------------------------------------------------------
with tab2:
    active_ver = "v2" if live_v2 else "v1"
    active_model = sim["rf_v2"] if live_v2 else sim["rf_v1"]
    st.subheader(f"Layer 2 - Random Forest Multiclass - failure mode diagnosis  ·  Production model {active_ver}")
    pred = sim["pred_live"][:t]
    proba = sim["proba_live"][:t]
    true = sim["y_all"][:t]
    modes = [mm for mm in P.MODES if mm != "Normal"]

    c1, c2 = st.columns(2)
    with c1:
        # Predicted failure-mode distribution (excluding Normal)
        pcount = [int((pred == mm).sum()) for mm in modes]
        tcount = [int((true == mm).sum()) for mm in modes]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[MODE_EN[mm] for mm in modes], x=tcount, name="Actual",
                      orientation="h", marker_color=GRAY,
                      hovertemplate="%{y}<br>Actual %{x}<extra></extra>"))
        fig.add_trace(go.Bar(y=[MODE_EN[mm] for mm in modes], x=pcount, name="Predicted",
                      orientation="h", marker_color=PURPLE,
                      hovertemplate="%{y}<br>Predicted %{x}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.28, height=360,
                          title=dict(text="Cumulative count by failure mode (Actual vs Predicted)",
                                     y=0.97, yanchor="top"),
                          margin=dict(t=58,b=40,l=10,r=10),
                          legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
        st.plotly_chart(fig, width='stretch')
    with c2:
        # Explainability — feature importance (active production model)
        imp = active_model.feature_importances_
        names = sim["feature_names"]
        order = np.argsort(imp)
        labels = [FEAT_EN.get(names[i], names[i]) for i in order]
        bar_cols = [PURPLE if k >= len(order)-3 else "#c4b5fd" for k in range(len(order))]
        fig = go.Figure(go.Bar(x=imp[order], y=labels, orientation="h",
                               marker_color=bar_cols,
                               text=[f"{v*100:.0f}%" for v in imp[order]],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>importance %{x:.1%}<extra></extra>"))
        fig.update_layout(height=360, title=dict(
                              text=f"Feature Importance ({active_ver}) - basis for diagnosis",
                              y=0.97, yanchor="top"),
                          margin=dict(t=58,b=40,l=10,r=10))
        fig.update_xaxes(showgrid=True, gridcolor=GRID, tickformat=".0%", range=[0, max(imp)*1.18])
        st.plotly_chart(fig, width='stretch')

    # Metrics + recall by failure mode
    m2 = st.columns(4)
    m2[0].metric("Predicted failures (cumulative)", f"{int((pred != 'Normal').sum()):,}")
    m2[1].metric("Actual failures", f"{int((true != 'Normal').sum()):,}")
    m2[2].metric("Mean prediction confidence", f"{proba.mean()*100:.1f}%")
    overall_rec = P.failure_scores(true, pred)[1] * 100   # (acc, fail_recall, macro_f1)
    m2[3].metric("Failure recall", f"{overall_rec:.1f}%")
    st.caption("Input features: product type, 5 sensors, and derived variables (temp diff, power). "
               "Class imbalance is corrected with class_weight=balanced. "
               "The top-ranked features (purple) are the primary basis for maintenance decisions.")

    g1, g2 = st.columns([2,3])
    with g1:
        st.markdown("**Recall by failure mode**")
        pmr = P.per_mode_recall(true, pred, modes)
        rec_rows = [{"Failure Mode": MODE_EN[mm],
                     "Actual Count": pmr[mm]["support"],
                     "Recall": ("-" if pmr[mm]["support"] == 0
                                else f"{pmr[mm]['recall']*100:.0f}%")}
                    for mm in modes]
        st.dataframe(pd.DataFrame(rec_rows), width='stretch', hide_index=True)
    with g2:
        st.markdown("**Recent failure diagnoses**")
        idx = np.where(sim["pred_live"][:t] != "Normal")[0]
        rows = []
        for i in idx[-7:][::-1]:
            rows.append({
                "Time": int(i),
                "Predicted": MODE_EN[sim["pred_live"][i]],
                "Actual": MODE_EN[sim["y_all"][i]],
                "Confidence": f"{sim['proba_live'][i]*100:.0f}%",
                "SPC": "Alarm" if flags["any_ooc"].values[i] else "Normal",
                "Match": "Match" if sim["pred_live"][i] == sim["y_all"][i] else "Mismatch",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.info("No failures diagnosed yet (normal operation).")


# ----------------------------------------------------------------------------
# Tab 3 — drift & retraining
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Monitoring + Automation - drift detection and automatic retraining")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_vrect(x0=P.DRIFT_START, x1=N, fillcolor=RED, opacity=0.05, line_width=0,
                      annotation_text="drift", annotation_position="top right",
                      annotation_font_color=RED, annotation_font_size=11)
        fig.add_trace(go.Scatter(x=np.arange(t), y=psi[:t], mode="lines",
                      line=dict(color=AMBER, width=2), name="PSI (Torque)",
                      fill="tozeroy", fillcolor="rgba(217,119,6,0.10)",
                      hovertemplate="t=%{x}<br>PSI %{y:.3f}<extra></extra>"))
        fig.add_hline(y=P.PSI_THRESHOLD, line=dict(color=RED, dash="dash"),
                      annotation_text=f"Threshold {P.PSI_THRESHOLD}", annotation_position="top left")
        if tstar and t >= tstar:
            fig.add_vline(x=tstar, line=dict(color=PURPLE, width=2),
                          annotation_text=f"Retrain t*={tstar}", annotation_position="top right")
        fig.update_layout(height=330, title="Input drift indicator PSI (torque distribution shift)",
                          margin=dict(t=44,b=10,l=10,r=10), showlegend=False)
        fig.update_xaxes(range=[0, N])
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = go.Figure()
        xa = np.arange(t)
        fig.add_vrect(x0=P.DRIFT_START, x1=N, fillcolor=RED, opacity=0.05, line_width=0)
        fig.add_trace(go.Scatter(x=xa, y=sim["rec_nore"][:t]*100, mode="lines",
                      line=dict(color=RED, width=2, dash="dot"), name="No retraining (v1 fixed)",
                      hovertemplate="t=%{x}<br>recall %{y:.1f}%<extra>no retrain</extra>"))
        fig.add_trace(go.Scatter(x=xa, y=sim["rec_live"][:t]*100, mode="lines",
                      line=dict(color=GREEN, width=2.5), name="Production (retrain on drift)",
                      hovertemplate="t=%{x}<br>recall %{y:.1f}%<extra>production</extra>"))
        if tstar and t >= tstar:
            fig.add_vline(x=tstar, line=dict(color=PURPLE, width=2),
                          annotation_text=f"t*={tstar}", annotation_position="bottom right",
                          annotation_font_size=11)
        fig.update_layout(height=330, title="Failure recall (rolling) - retraining effect",
                          margin=dict(t=44,b=10,l=10,r=10), hovermode="x unified",
                          legend=dict(orientation="h", y=-0.18))
        fig.update_xaxes(range=[0, N])
        fig.update_yaxes(range=[0,100], title="recall %")
        st.plotly_chart(fig, width='stretch')

    if tstar and t >= tstar:
        st.success(f"Automatic retraining loop fired. PSI crossed {P.PSI_THRESHOLD} at t*={tstar:,}, triggering "
                   "training of RF v2 on the accumulated data, validation, and deployment. The production "
                   "model (green) recovers to a higher recall than the no-retrain baseline (red dotted).")
        # Validation gate — compare v2 vs v1 on the canary window, then promote/rollback (design doc ch.7, step 4)
        if promotion:
            v1r = promotion["v1"]["fail_recall"]*100
            v2r = promotion["v2"]["fail_recall"]*100
            lo, hi = promotion["window"]
            gc = st.columns([1,1,2])
            gc[0].metric("Validation v1 recall", f"{v1r:.1f}%")
            gc[1].metric("Validation v2 recall", f"{v2r:.1f}%", f"{v2r-v1r:+.1f}%p")
            verdict = "Promote (deploy)" if promoted else "Rollback (keep v1)"
            gc[2].markdown(
                f"**Validation gate result: {verdict}**  \n"
                f"The canary window `[{lo:,}, {hi:,}]` was not used to train either model. "
                f"On this window v2 outperformed v1, so it was deployed automatically. "
                f"Had it underperformed, the system would roll back to v1.")
    else:
        st.info("Move the slider into the drift phase (red) to watch PSI rise; retraining triggers when it reaches the threshold.")

    # SPC x RF cross-validation matrix (cumulative so far)
    st.markdown("**Dual-layer cross-validation - SPC x RF (cumulative)**")
    spc_flag = flags["any_ooc"].values[:t]
    rf_fail = (sim["pred_live"][:t] != "Normal")
    mtx = pd.DataFrame({
        "": ["SPC Normal", "SPC Alarm"],
        "RF Normal": [int((~spc_flag & ~rf_fail).sum()), int((spc_flag & ~rf_fail).sum())],
        "RF Failure": [int((~spc_flag & rf_fail).sum()), int((spc_flag & rf_fail).sum())],
    }).set_index("")
    st.table(mtx)
    st.caption("A growing 'SPC Alarm, RF Normal' cell means a new pattern the RF cannot keep up with (concept drift), "
               "which is the basis for triggering retraining.")

    # Model registry — per-version training history/metrics (Model layer, persisted under models/)
    with st.expander("Model Registry - version history (models/)"):
        reg_rows = [{
            "Version": "v1",
            "Train Range": f"0 - {P.N_BASELINE:,}",
            "Role": "Baseline initial model",
            "Drift recall": f"{promotion['v1']['fail_recall']*100:.1f}%" if promotion else "-",
            "Status": "Initial deploy",
        }]
        if sim["rf_v2"] is not None and promotion:
            reg_rows.append({
                "Version": "v2",
                "Train Range": f"0 - {tstar:,}",
                "Role": f"Retrained after drift (t*={tstar:,})",
                "Drift recall": f"{promotion['v2']['fail_recall']*100:.1f}%",
                "Status": "Validated, deployed" if promoted else "Rolled back",
            })
        st.dataframe(pd.DataFrame(reg_rows), width='stretch', hide_index=True)
        st.caption("Each version is saved under `models/` as a joblib model plus JSON metadata "
                   "(training timestamp, train range, hyperparameters, feature importances, metrics), "
                   "so any past model can be reproduced and audited.")


# ----------------------------------------------------------------------------
# Tab 4 — business value
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Business Value - before/after retraining comparison")
    dm = slice(P.DRIFT_START, t) if t > P.DRIFT_START else slice(P.DRIFT_START, P.DRIFT_START+1)
    yt = sim["y_all"][dm]
    if len(yt) > 10 and (yt != "Normal").sum() > 0:
        def stat(pred):
            acc, rec, _ = P.failure_scores(yt, pred[dm])   # shared metrics helper
            return rec, acc
        r_live, a_live = stat(sim["pred_live"])
        r_nore, a_nore = stat(sim["pred_noretrain"])
        c = st.columns(2)
        c[0].metric("Failure recall (drift phase)", f"{r_live*100:.1f}%",
                    f"{(r_live-r_nore)*100:+.1f}%p vs no-retrain")
        c[1].metric("Accuracy (drift phase)", f"{a_live*100:.1f}%",
                    f"{(a_live-a_nore)*100:+.1f}%p vs no-retrain")
        missed = int(((yt != "Normal") & (sim["pred_noretrain"][dm] == "Normal")).sum())
        caught = int(((yt != "Normal") & (sim["pred_live"][dm] != "Normal")).sum())
        miss_live = int(((yt != "Normal") & (sim["pred_live"][dm] == "Normal")).sum())
        st.markdown(f"""
        #### Summary
        - Without retraining, {missed} failures would have been missed in the drift phase (no-retrain v1 baseline).
        - With the dual-layer system and automatic retraining, missed failures in the same phase drop to {miss_live}.
        - SPC detects input drift immediately, and automatic retraining at the PSI threshold restores performance.
        """)
    else:
        st.info("Move the slider into the drift phase to see the before/after retraining comparison.")

    st.divider()
    st.markdown("""
    | Aspect | Standard ML (one-off PoC) | This system (Dual-layer MLOps) |
    |---|---|---|
    | Anomaly detection | Simple thresholds, many false alarms | SPC 3 sigma statistical detection, fewer false alarms |
    | Root-cause diagnosis | Manual teardown by an engineer | RF auto-classifies 5 failure modes with supporting evidence |
    | Drift response | Manual rework after a problem occurs | PSI auto-detection then automatic retraining and deployment |
    | Model lifespan | Performance decays over time | Sustained performance via automatic retraining |
    """)


# ----------------------------------------------------------------------------
# Auto-play
# ----------------------------------------------------------------------------
if auto and t < N:
    st.session_state.t = min(N, t + step)
    time.sleep(0.4)
    st.rerun()
