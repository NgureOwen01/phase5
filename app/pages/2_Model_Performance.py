import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Flexible import to support different working directories
try:
    from app.utils import load_metrics, load_models, load_css
except Exception:
    from utils import load_metrics, load_models, load_css  # type: ignore

st.set_page_config(page_title="Model Performance", page_icon=None, layout="wide")

# Inject global CSS
css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
# Blur-to-focus text animation (reuse from main)
blurtext_css = """
.blur-reveal { display: inline-block; }
.blur-reveal .w {
  display: inline-block;
  filter: blur(12px);
  opacity: 0;
  transform: translateY(0.4em);
  animation: br-reveal 0.9s cubic-bezier(0.2, 0.7, 0.2, 1) forwards;
  animation-delay: calc(var(--base, 0s) + (var(--i, 0) * var(--stagger, 0.06s)));
}
@keyframes br-reveal { to { filter: blur(0); opacity: 1; transform: translateY(0); } }
"""
st.markdown(f"<style>{blurtext_css}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1 style="margin-bottom:0.25rem;">
    <span class="blur-reveal" style="--stagger:.06s; --base:.33s;">
      <span class="w" style="--i:0">Model</span>
      <span class="w" style="--i:1">Performance</span>
      <span class="w" style="--i:2">Dashboard</span>
    </span>
  </h1>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-divider"><span class="label">Compare • Evaluate • Select</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="content-intro">
    <h4><span class="material-symbols-outlined">query_stats</span> Comprehensive Model Analysis</h4>
    <p>Compare machine learning models with detailed performance metrics, interactive visualizations, and comprehensive analytics. Identify the best performing models for your cropland classification tasks.</p>
    <p>💡 <strong>Metrics:</strong> Accuracy, F1-Score, and ROC-AUC provide different perspectives on model performance across various classification scenarios.</p>
</div>
""", unsafe_allow_html=True)

metrics = load_metrics()
if not metrics:
    metrics = {
        'Model': ['RandomForest', 'GradientBoosting', 'CatBoost', 'XGBoost', 'LightGBM'],
        'Accuracy': [0.735, 0.735, 0.715, 0.715, 0.705],
        'F1': [0.475, 0.418, 0.521, 0.486, 0.478],
        'ROC_AUC': [0.737, 0.697, 0.725, 0.703, 0.704]
    }

results_df = pd.DataFrame(metrics)
models_loaded = load_models()

# Top cards (compact)
st.markdown('<div class="glass-container">', unsafe_allow_html=True)
colA, colB, colC = st.columns(3)
for col, metric_name in zip([colA, colB, colC], ["Accuracy", "F1", "ROC_AUC"]):
    top_row = results_df.sort_values(metric_name, ascending=False).iloc[0]
    col.markdown(
        f"""
        <div class=\"glass-card\" style=\"padding:0.8rem; margin:0.5rem; min-height:110px; display:flex; flex-direction:column; justify-content:center;\"> 
            <h4 style=\"font-size:0.9rem; margin:0 0 0.25rem 0;\">Best {metric_name}</h4>
            <h2 style=\"font-size:1.05rem; margin:0 0 0.25rem 0;\">{top_row['Model']}</h2>
            <h3 style=\"font-size:1.2rem; margin:0;\">{top_row[metric_name]:.3f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# Tabs for charts
tabs = st.tabs(["Bars", "Radar", "3D", "Table"])

with tabs[0]:
    metric = st.selectbox("Select Metric", ["Accuracy", "F1", "ROC_AUC"], index=0)
    fig = px.bar(results_df, x='Model', y=metric, color=metric, color_continuous_scale='viridis',
                 title=f"Model {metric} Comparison")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        <div class="feature-guidance">
            <span class="material-symbols-outlined">query_stats</span>
            <strong>How to read this:</strong>
            <ul>
                <li><b>Accuracy</b>: Overall fraction of correct predictions. Simple but can be <i>misleading</i> on imbalanced data (e.g., many non‑cropland samples).</li>
                <li><b>F1‑Score</b>: Harmonic mean of precision and recall (class‑1). Prefer when classes are imbalanced or false negatives/positives are costly.</li>
                <li><b>ROC‑AUC</b>: Threshold‑independent ranking quality. Measures ability to rank cropland above non‑cropland; robust for imbalance.</li>
            </ul>
            <em>Tip:</em> For cropland detection (often imbalanced), prioritize F1 and ROC‑AUC; use Accuracy as a sanity check.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[1]:
    st.subheader("🧭 Radar Comparison")
    categories = ['Accuracy', 'F1', 'ROC_AUC']

    # Controls
    all_models = list(results_df['Model'].astype(str))
    selected_models = st.multiselect("Models to show", options=all_models, default=all_models)
    radar_opacity = st.slider("Fill opacity", 0.1, 0.8, 0.35, 0.05)
    show_mean = st.checkbox("Show mean shape", value=True)

    # Palette
    base_colors = ['#22c55e', '#06b6d4', '#a78bfa', '#f59e0b', '#ef4444', '#3b82f6', '#10b981', '#eab308']

    radar = go.Figure()

    # Add selected model polygons
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = str(row['Model'])
        if model_name not in selected_models:
            continue
        values = [float(row['Accuracy']), float(row['F1']), float(row['ROC_AUC'])]
        color = base_colors[i % len(base_colors)]
        hover_text = (
            f"<b>{model_name}</b><br>"
            f"Accuracy: {values[0]:.3f}<br>F1: {values[1]:.3f}<br>ROC-AUC: {values[2]:.3f}"
        )
        radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],  # close loop
                theta=categories + [categories[0]],
                fill='toself',
                name=model_name,
                line=dict(color=color, width=2),
                fillcolor=color,
                opacity=radar_opacity,
                text=[hover_text] * 4,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Add mean polygon across selected models
    if show_mean and selected_models:
        sub = results_df[results_df['Model'].astype(str).isin(selected_models)]
        means = [sub['Accuracy'].mean(), sub['F1'].mean(), sub['ROC_AUC'].mean()]
        mean_text = (
            f"<b>Mean</b><br>Accuracy: {means[0]:.3f}<br>F1: {means[1]:.3f}<br>ROC-AUC: {means[2]:.3f}"
        )
        radar.add_trace(
            go.Scatterpolar(
                r=means + [means[0]], theta=categories + [categories[0]],
                name='Mean',
                line=dict(color='white', width=3, dash='dash'),
                fill=None,
                text=[mean_text] * 4,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e5e7eb')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickfont=dict(color='#e5e7eb')),
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        margin=dict(t=20, l=20, r=20, b=40),
        hoverlabel=dict(bgcolor='rgba(17,24,39,0.95)', font_color='#e5e7eb', bordercolor='rgba(255,255,255,0.15)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(radar, use_container_width=True)

    st.markdown(
        """
        <div class="feature-guidance">
            <span class="icon">🧭</span>
            <strong>What this shows</strong>
            <ul>
                <li>Each <b>spoke</b> is a metric (Accuracy, F1, ROC‑AUC). Each <b>colored shape</b> is a model.</li>
                <li>Bigger shapes (closer to the outer ring) generally mean <b>better overall performance</b>.</li>
            </ul>
            <strong>How to read it</strong>
            <ul>
                <li><b>Balanced shape</b> (similar lengths on all spokes) = consistent model.</li>
                <li><b>Spiky shape</b> (one long, one short spoke) = trade‑off. E.g., great ROC‑AUC but weaker F1.</li>
                <li>For <b>imbalanced tasks</b> like cropland detection, pay extra attention to the <b>F1 spoke</b> and <b>ROC‑AUC</b>.</li>
            </ul>
            <strong>Quick pick guide</strong>
            <ul>
                <li>If you need an <b>all‑rounder</b>, choose the model with the <b>largest, most balanced</b> shape.</li>
                <li>If you care most about <b>catching cropland correctly</b>, prioritize a long <b>F1</b> spoke (and solid ROC‑AUC).</li>
                <li>If ROC‑AUC is high but F1 is low, the model ranks well but the threshold may be off — consider <b>threshold tuning</b> or <b>class weights</b>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[2]:
    st.subheader("🧩 3D Performance Space")

    color_by = st.selectbox("Color by", ["Accuracy", "F1", "ROC_AUC"], index=2)
    size_by = st.selectbox("Size by", ["Accuracy", "F1", "ROC_AUC"], index=1)
    size_min, size_max = st.slider("Marker size range", 6, 24, (10, 16))
    jitter = st.checkbox("Add slight jitter (spread overlapping points)", value=True)
    jitter_scale = st.slider("Jitter scale", 0.0, 0.05, 0.02, 0.005, disabled=not jitter)

    # Prepare coordinates and apply jitter if enabled
    Xv = results_df['Accuracy'].astype(float).values.copy()
    Yv = results_df['F1'].astype(float).values.copy()
    Zv = results_df['ROC_AUC'].astype(float).values.copy()
    if jitter and len(Xv) > 1:
        rng = np.random.default_rng(42)
        Xv = np.clip(Xv + rng.normal(0, jitter_scale, size=len(Xv)), 0, 1)
        Yv = np.clip(Yv + rng.normal(0, jitter_scale, size=len(Yv)), 0, 1)
        Zv = np.clip(Zv + rng.normal(0, jitter_scale, size=len(Zv)), 0, 1)

    # Normalize size_by to [size_min, size_max]
    vals = results_df[size_by].astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-9:
        sizes = np.full(len(vals), (size_min + size_max) / 2.0)
    else:
        sizes = size_min + (vals - vmin) * (size_max - size_min) / (vmax - vmin)

    hovertemplate = (
        "<b>%{text}</b><br>"
        + "Accuracy: %{x:.3f}<br>F1: %{y:.3f}<br>ROC-AUC: %{z:.3f}<extra></extra>"
    )

    fig3d = go.Figure(data=go.Scatter3d(
        x=Xv, y=Yv, z=Zv,
        mode='markers+text', text=results_df['Model'], textposition='top center',
        marker=dict(size=sizes, color=results_df[color_by], colorscale='Viridis', showscale=True,
                    colorbar=dict(title=color_by), line=dict(width=1, color='white')),
        hovertemplate=hovertemplate,
    ))

    # Auto-zoom to data bounds with a small padding
    pad = 0.05
    xrange = [max(0.0, float(Xv.min()) - pad), min(1.0, float(Xv.max()) + pad)]
    yrange = [max(0.0, float(Yv.min()) - pad), min(1.0, float(Yv.max()) + pad)]
    zrange = [max(0.0, float(Zv.min()) - pad), min(1.0, float(Zv.max()) + pad)]

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(title='Accuracy', range=xrange, gridcolor='rgba(255,255,255,0.08)', zeroline=False),
            yaxis=dict(title='F1', range=yrange, gridcolor='rgba(255,255,255,0.08)', zeroline=False),
            zaxis=dict(title='ROC AUC', range=zrange, gridcolor='rgba(255,255,255,0.08)', zeroline=False),
            aspectmode='cube',
        ),
        margin=dict(t=10, l=10, r=10, b=10),
        hoverlabel=dict(bgcolor='rgba(17,24,39,0.95)', font_color='#e5e7eb', bordercolor='rgba(255,255,255,0.15)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown(
        """
        <div class="feature-guidance">
            <span class="icon">🧩</span>
            <strong>How to read this:</strong>
            <ul>
                <li>The <b>ideal point</b> is near (1.0, 1.0, 1.0). Models <b>closer</b> to that corner are better overall.</li>
                <li><b>Marker color</b> shows the selected metric; <b>marker size</b> can highlight your priority metric.</li>
                <li>Use color/size together to quickly spot strong candidates and potential trade‑offs.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.subheader("📋 Results Table")
    st.markdown(
        """
        <div class="feature-guidance">
            <span class="icon">📋</span>
            <strong>How to use this:</strong>
            <ul>
                <li>Each row is a model with <b>Accuracy</b>, <b>F1</b>, and <b>ROC‑AUC</b> scores. Higher is better for all.</li>
                <li><b>Sort</b> by the column that matters most for your task (e.g., F1 for imbalanced cropland detection).</li>
                <li>Use the <b>Download</b> button to export metrics for reports or further analysis.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(results_df, use_container_width=True)
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download metrics (CSV)", data=csv, file_name="model_metrics.csv", mime="text/csv")
