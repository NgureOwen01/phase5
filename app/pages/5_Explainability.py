import streamlit as st
from pathlib import Path
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Flexible import to tolerate different run contexts
try:
    from app.utils import (
        load_models,
        get_feature_schema,
        load_css,
        inject_css_block,
        align_to_schema,
        load_feature_defaults,
        load_parquet_if_exists,
        get_model_feature_schema,
        unwrap_model,
    )
except Exception:
    from utils import (  # type: ignore
        load_models,
        get_feature_schema,
        load_css,
        inject_css_block,
        align_to_schema,
        load_feature_defaults,
        load_parquet_if_exists,
        get_model_feature_schema,
        unwrap_model,
    )

st.set_page_config(page_title="Explainability", page_icon=None, layout="wide")

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
      <span class="w" style="--i:1">Explainability</span>
    </span>
  </h1>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-divider"><span class="label">Explain • Compare • Act</span></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="content-intro">
        <h4><span class=\"material-symbols-outlined\">psychology</span> Interpret and trust your models</h4>
        <p>Explore feature contributions with SHAP values, visualize feature effects via PDP/ICE, and inspect top drivers of predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Lottie animations removed for a leaner app footprint

models = load_models()
model_names = list(models.keys())
if not model_names:
    st.warning("No models found in `models/`. Save *.pkl or *.joblib models to enable explainability.")
    st.stop()

with st.sidebar:
    st.subheader("Settings")
    model_name = st.selectbox("Model", options=model_names, index=0)
    sample_size = st.slider("Sample size for explanations", 100, 5000, 800, 50)
    top_k = st.slider("Top features (importance)", 5, 30, 12)
    st.markdown("---")
    uploaded = st.file_uploader("Optional: Upload data CSV for explainability", type=["csv"]) 

mdl = models[model_name]
# Use model-specific schema when available
model_schema = get_model_feature_schema(mdl) or get_feature_schema(models)

# Load background data X
X_source: pd.DataFrame
source_label = ""
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if model_schema:
            defaults = load_feature_defaults(model_schema)
            X_source = align_to_schema(df_in, model_schema, defaults)
        else:
            X_source = df_in.select_dtypes(include=[np.number]).fillna(0.0)
        source_label = "(uploaded)"
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        X_source = pd.DataFrame()
else:
    # Fallback: try to assemble from any numeric columns in available data files
    # Prefer light-weight synthetic sample if nothing available
    X_source = pd.DataFrame()

# If no data yet, synthesize a small dataframe based on known features
if X_source.empty:
    if model_schema:
        # Try artifacts background first
        raw = load_parquet_if_exists(Path.cwd() / "artifacts" / "merged_data.parquet")
        if not raw.empty:
            defaults = load_feature_defaults(model_schema)
            X_source = align_to_schema(raw, model_schema, defaults)
            source_label = "(artifacts/merged_data.parquet)"
        else:
            rng = np.random.default_rng(42)
            X_source = pd.DataFrame(rng.normal(size=(max(sample_size, 500), len(model_schema))), columns=model_schema)
            source_label = "(synthetic)"
    else:
        st.info("No feature schema detected. Upload a CSV with your model features to enable explainability.")
        st.stop()

# Downsample for performance
if len(X_source) > sample_size:
    X = X_source.sample(sample_size, random_state=42)
else:
    X = X_source.copy()

# Inject a lightweight, CSS-only loader we can show while computing
loader_css = """
.loader-bars { display:flex; gap:8px; align-items:flex-end; justify-content:center; height:42px; margin: 0.25rem 0 0.5rem; }
.loader-bars .bar { width:10px; height:16px; background: var(--primary-color); border-radius: 6px; animation: loader-wave 1s ease-in-out infinite; box-shadow: 0 2px 8px rgba(16,185,129,0.25); }
.loader-bars .bar:nth-child(2) { animation-delay: 0.08s; }
.loader-bars .bar:nth-child(3) { animation-delay: 0.16s; }
.loader-bars .bar:nth-child(4) { animation-delay: 0.24s; }
.loader-bars .bar:nth-child(5) { animation-delay: 0.32s; }
@keyframes loader-wave { 0%,100% { transform: scaleY(0.5); opacity: 0.6; } 50% { transform: scaleY(1.45); opacity: 1; } }
.loader-caption { color: var(--text-secondary); font-size: 0.95rem; opacity: 0.9; }
"""
inject_css_block(loader_css)

# Helper: get model output function for probability of class 1 when available
def predict_function(model, data: pd.DataFrame) -> np.ndarray:
    core = unwrap_model(model)
    # CatBoost: use Pool and ensure column order
    try:
        import catboost  # type: ignore
        is_cat = core.__class__.__module__.startswith("catboost")
    except Exception:
        catboost = None  # type: ignore
        is_cat = False
    try:
        if is_cat:
            pool = catboost.Pool(data, feature_names=list(data.columns))  # type: ignore
            if hasattr(core, "predict_proba"):
                proba = core.predict_proba(pool)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]
                return proba.ravel()
            preds = core.predict(pool)
            return np.array(preds, dtype=float).ravel()
        # scikit-learn / xgb / lgbm path
        if hasattr(core, "predict_proba"):
            proba = core.predict_proba(data)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.ravel()
        if hasattr(core, "decision_function"):
            scores = core.decision_function(data)
            return np.array(scores, dtype=float).ravel()
        preds = core.predict(data)
        return np.array(preds, dtype=float).ravel()
    except Exception:
        return np.zeros(len(data), dtype=float)

# Helper: robust mean(|SHAP|) per feature across different SHAP shapes
def mean_abs_shap_per_feature(shap_values_obj) -> np.ndarray:
    vals = getattr(shap_values_obj, "values", shap_values_obj)
    try:
        arr = np.array(vals)
    except Exception:
        # e.g., list of per-class arrays -> stack on last axis
        arr = np.stack(vals, axis=-1)
    # (n_samples, n_features) or (n_samples, n_features, n_outputs) or variants
    if arr.ndim == 1:
        return np.abs(arr)
    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)
    # average across all axes except feature axis -> choose axis with size n_features
    n_features = X.shape[1]
    feature_axes = [i for i, s in enumerate(arr.shape) if s == n_features]
    feat_axis = feature_axes[0] if feature_axes else 1
    axes_to_mean = tuple(i for i in range(arr.ndim) if i != feat_axis)
    return np.abs(arr).mean(axis=axes_to_mean)

# Helper: extract a sample-length SHAP vector for a selected feature
def sample_shap_for_feature(shap_values_obj, feature_idx: int, n_samples: int, n_features: int) -> np.ndarray:
    vals = getattr(shap_values_obj, "values", shap_values_obj)
    try:
        arr = np.array(vals)
    except Exception:
        arr = np.stack(vals, axis=0)
    # Identify axes
    sample_axes = [i for i, s in enumerate(arr.shape) if s == n_samples]
    feature_axes = [i for i, s in enumerate(arr.shape) if s == n_features]
    sample_axis = sample_axes[0] if sample_axes else 0
    feature_axis = feature_axes[0] if feature_axes else (1 if arr.ndim > 1 else 0)
    # Select the requested feature along its axis
    arr_sel = np.take(arr, indices=feature_idx, axis=feature_axis)
    # Now reduce any remaining non-sample axes by mean
    if arr_sel.ndim == 1:
        # ideally already length n_samples
        if arr_sel.shape[0] != n_samples:
            return np.resize(arr_sel, n_samples)
        return arr_sel
    # Find the sample axis in arr_sel
    sample_axes_sel = [i for i, s in enumerate(arr_sel.shape) if s == n_samples]
    if not sample_axes_sel:
        # fallback: try last axis as samples
        sample_axes_sel = [arr_sel.ndim - 1]
    s_ax = sample_axes_sel[0]
    axes_to_mean = tuple(i for i in range(arr_sel.ndim) if i != s_ax)
    y = arr_sel
    if axes_to_mean:
        y = y.mean(axis=axes_to_mean)
    # Ensure 1D length n_samples
    y = np.asarray(y).ravel()
    if y.shape[0] != n_samples:
        y = np.resize(y, n_samples)
    return y

# Compute SHAP values (gracefully degrade) with fun loader
shap_values = None
explainer_name = ""
loading_placeholder = st.empty()
funny_lines = [
    "Converting photons into insights… please hold the tractor!",
    "Feeding satellite pixels to hungry SHAP gremlins…",
    "Teaching the model to spot crops faster than a scarecrow…",
    "Asking the AI to explain itself. It’s writing an essay…",
    "Squeezing secrets out of spectral bands… like orange juice…",
]
caption = random.choice(funny_lines)
loading_placeholder.markdown(
    f"""
    <div class=\"glass-container\" style=\"text-align:center;\">\n      <div class=\"loader-bars\">\n        <div class=\"bar\"></div><div class=\"bar\"></div><div class=\"bar\"></div><div class=\"bar\"></div><div class=\"bar\"></div>\n      </div>\n      <div class=\"loader-caption\">{caption}</div>\n    </div>
    """,
    unsafe_allow_html=True,
)
try:
    with st.spinner("Summoning SHAP sprites… brewing explanations"):
        import shap
        core_model = unwrap_model(mdl)
        # Prefer TreeExplainer for tree-based models like CatBoost
        is_catboost = False
        try:
            is_catboost = core_model.__class__.__module__.startswith("catboost")
        except Exception:
            is_catboost = False
        if is_catboost:
            explainer = shap.TreeExplainer(core_model)
            shap_values = explainer.shap_values(X)
            explainer_name = "TreeExplainer(CatBoost)"
        else:
            explainer = shap.Explainer(core_model, X, feature_names=list(X.columns))
            shap_values = explainer(X)
            explainer_name = explainer.__class__.__name__
finally:
    loading_placeholder.empty()
if shap_values is None:
    st.warning("SHAP explainability unavailable. Try a smaller sample or upload a compact CSV.")

# Tabs for visualizations
tabs = st.tabs(["Overview", "Feature Importance", "Dependence", "PDP / ICE"]) 

with tabs[0]:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Samples used", f"{len(X):,}")
    with c2:
        st.metric("Features", f"{X.shape[1]:,}")
    with c3:
        st.metric("Data source", source_label or "(unknown)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="feature-guidance">
            <span class="material-symbols-outlined">info</span>
            <strong>What you are seeing</strong>
            <ul>
                <li><b>SHAP</b> explains a prediction by assigning each feature a contribution (positive pushes prediction up; negative pushes it down).</li>
                <li><b>Mean |SHAP|</b> aggregates absolute contributions across samples to rank features by overall impact.</li>
                <li>Values are relative to the model's <b>baseline</b> (expected prediction over the sample).</li>
            </ul>
            <em>Tip:</em> Use a representative sample size (sidebar) for stable insights.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if shap_values is not None:
        try:
            mean_abs = mean_abs_shap_per_feature(shap_values)
            # Align lengths defensively
            n = min(len(X.columns), len(mean_abs))
            imp_df = pd.DataFrame({"feature": list(X.columns)[:n], "importance": np.asarray(mean_abs).ravel()[:n]})
            imp_df = imp_df.sort_values("importance", ascending=False)
            fig = px.bar(imp_df.head(top_k), x="importance", y="feature", orientation='h', title="Mean |SHAP| by feature")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
                <div class="feature-guidance">
                    <span class="material-symbols-outlined">query_stats</span>
                    <strong>How to read this</strong>
                    <ul>
                        <li>Bars show average absolute impact on the prediction: longer bar ⇒ stronger influence.</li>
                        <li>Highly <b>correlated features</b> may share credit; consider grouping or domain checks.</li>
                        <li>Use <b>Top features</b> slider (sidebar) to adjust how many you review.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.info(f"Unable to compute SHAP importances: {e}")
    else:
        st.info("SHAP values not available. Try uploading a small CSV (200–1000 rows) matching your model features.")

with tabs[1]:
    st.subheader("Top Feature Importances (Mean |SHAP|)")
    if shap_values is not None:
        try:
            mean_abs = mean_abs_shap_per_feature(shap_values)
            n = min(len(X.columns), len(mean_abs))
            imp_df = pd.DataFrame({"feature": list(X.columns)[:n], "importance": np.asarray(mean_abs).ravel()[:n]})
            imp_df = imp_df.sort_values("importance", ascending=False)
            fig = px.bar(imp_df.head(top_k), x="importance", y="feature", orientation='h', color="importance", color_continuous_scale='viridis')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            csv_imp = imp_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download full importances (CSV)", data=csv_imp, file_name="shap_importances.csv", mime="text/csv")
        except Exception as e:
            st.info(f"Unable to compute SHAP importances: {e}")
    else:
        st.info("SHAP not available. Ensure `shap` is installed and provide a tabular dataset.")

with tabs[2]:
    st.subheader("SHAP Dependence")
    if shap_values is not None:
        feat = st.selectbox("Feature", options=list(X.columns), index=0, key="dep_feat")
        color_feat = st.selectbox("Color by", options=[None] + list(X.columns), index=0, key="dep_color")
        try:
            idx = list(X.columns).index(feat)
            y_vals = sample_shap_for_feature(shap_values, idx, n_samples=len(X), n_features=X.shape[1])
            x_vals = X.iloc[:, idx].values
            color_vals = X[color_feat] if color_feat else None
            fig = px.scatter(
                x=x_vals, y=y_vals, color=color_vals,
                labels={"x": feat, "y": "SHAP value"}, opacity=0.7, color_continuous_scale='viridis',
                title=f"Dependence: {feat} vs SHAP value"
            )
            fig.add_hline(y=0, line_dash="dash", opacity=0.5)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
                <div class="feature-guidance">
                    <span class="material-symbols-outlined">explore</span>
                    <strong>Interpretation</strong>
                    <ul>
                        <li><b>X‑axis</b>: feature value; <b>Y‑axis</b>: contribution to prediction (↑ increases, ↓ decreases).</li>
                        <li><b>Color</b> a second feature to reveal potential <b>interactions</b> (e.g., effect depends on another variable).</li>
                        <li>Patterns show <b>non‑linear</b> effects; a flat cloud ⇒ weak influence in this region.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Could not render dependence plot: {e}")
    else:
        st.info("Provide a dataset to compute SHAP dependence plots.")

with tabs[3]:
    st.subheader("Partial Dependence and ICE")
    feat = st.selectbox("Feature for PDP/ICE", options=list(X.columns), index=0, key="pdp_feat")
    grid_points = st.slider("Grid points", 10, 80, 30)
    ice_curves = st.slider("ICE curves (samples)", 5, 100, 30)

    # Build grid over the selected feature
    try:
        x_series = X[feat]
        vmin, vmax = np.nanpercentile(x_series, [1, 99])
        grid = np.linspace(vmin, vmax, grid_points)

        # PDP: hold others at median
        X_ref = X.median(numeric_only=True)
        X_pdp = pd.DataFrame(np.tile(X_ref.values, (grid_points, 1)), columns=X.columns)
        X_pdp[feat] = grid
        y_pdp = predict_function(mdl, X_pdp)

        fig_pdp = go.Figure()
        fig_pdp.add_trace(go.Scatter(x=grid, y=y_pdp, mode='lines', name='PDP', line=dict(width=3, color="#60A5FA")))

        # ICE: sample a few rows and vary feature
        ice_idx = X.sample(min(ice_curves, len(X)), random_state=42)
        for i, (_, row) in enumerate(ice_idx.iterrows()):
            X_ice = pd.DataFrame(np.tile(row.values, (grid_points, 1)), columns=X.columns)
            X_ice[feat] = grid
            y_ice = predict_function(mdl, X_ice)
            fig_pdp.add_trace(go.Scatter(x=grid, y=y_ice, mode='lines', line=dict(width=1.2, color='rgba(255,255,255,0.35)'), showlegend=False))

        fig_pdp.update_layout(title=f"PDP/ICE for {feat}", xaxis_title=feat, yaxis_title="Model output",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pdp, use_container_width=True)
        st.markdown(
            """
            <div class="feature-guidance">
                <span class="material-symbols-outlined">splitscreen</span>
                <strong>PDP vs ICE</strong>
                <ul>
                    <li><b>PDP</b> (thick line): average effect of changing the feature while keeping others fixed at median.</li>
                    <li><b>ICE</b> (thin lines): effect per individual sample; spread indicates <b>heterogeneous</b> behavior or interactions.</li>
                    <li>Y‑axis is the model output (e.g., probability for class 1); look for thresholds and saturations.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not compute PDP/ICE: {e}")
