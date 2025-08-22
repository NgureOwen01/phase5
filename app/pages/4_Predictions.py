import streamlit as st
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px

# Flexible import to support different working directories
try:
    from app.utils import (
        load_models,
        get_feature_schema,
        format_number,
        load_css,
        load_feature_defaults,
        align_to_schema,
        get_model_feature_schema,
        unwrap_model,
        normalize_input_columns,
        augment_with_engineered_features,
        enrich_with_satellite_features,
    )
except Exception:
    from utils import (  # type: ignore
        load_models,
        get_feature_schema,
        format_number,
        load_css,
        load_feature_defaults,
        align_to_schema,
        get_model_feature_schema,
        unwrap_model,
        normalize_input_columns,
        augment_with_engineered_features,
        enrich_with_satellite_features,
    )

st.set_page_config(page_title="Predictions", page_icon=None, layout="wide")

# Inject global CSS
css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
# Blur-to-focus text animation (same as main)
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
      <span class="w" style="--i:0">Make</span>
      <span class="w" style="--i:1">Predictions</span>
    </span>
  </h1>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-divider"><span class="label">Inputs • Confidence • Actions</span></div>', unsafe_allow_html=True)

models = load_models()
if not models:
    st.warning("No saved models found in 'models/' or 'artifacts/'. Place *.pkl or *.joblib files there.")

feature_names = get_feature_schema(models)

with st.sidebar:
    st.subheader("Settings")
    model_name = st.selectbox("Model", options=list(models.keys()) or ["-"], index=0)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    simplified = st.checkbox("Simplified mode (edit Top‑K features)", value=True, help="Edit only the most important features; others are auto‑filled with defaults.")
    top_k = st.slider("Top‑K features", 3, 20, 8, disabled=not simplified)

# Tabs for single vs batch
tabs = st.tabs(["Single Input", "Batch (CSV)"])

def infer_top_features(model, schema: list, k: int = 8) -> list:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(getattr(model, "feature_importances_")).ravel()
        idx = np.argsort(imp)[::-1]
        order = [schema[i] for i in idx if i < len(schema)]
        return order[:k]
    # Fallback: first K features in schema
    return schema[:k]

with tabs[0]:
    st.markdown("""
    <div class="content-intro">
        <h4><span class=\"material-symbols-outlined\">target</span> Single Prediction</h4>
        <p>Enter the feature values below to receive a prediction from the selected machine learning model. The model will output a classification (0 or 1) and a confidence score based on your inputs.</p>
        <p><strong>Tip:</strong> Adjust the 'Confidence Threshold' in the sidebar to fine-tune the classification sensitivity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="modern-form">', unsafe_allow_html=True)
    if not feature_names:
        st.markdown("""
        <div class="feature-guidance">
            <span class="material-symbols-outlined">warning</span>
            <strong>Demo Mode:</strong> Feature schema not detected. Using a simplified 4-feature input form for demonstration purposes.
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.number_input("Feature 1", value=0.0, help="Enter a numeric value for the first feature")
            f2 = st.number_input("Feature 2", value=0.0, help="Enter a numeric value for the second feature")
        with col2:
            f3 = st.number_input("Feature 3", value=0.0, help="Enter a numeric value for the third feature")
            f4 = st.number_input("Feature 4", value=0.0, help="Enter a numeric value for the fourth feature")
        input_array = np.array([[f1, f2, f3, f4]], dtype=float)
        shown_features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
    else:
        mdl = models.get(model_name)
        active_schema = get_model_feature_schema(mdl) or feature_names
        defaults = load_feature_defaults(active_schema)
        input_mode = st.radio("Input source", ["Manual", "From CSV row"], horizontal=True)

        prefill_row = None
        if input_mode == "From CSV row":
            csv_file = st.file_uploader("Upload CSV with your model's feature columns", type=["csv"], key="single_csv")
            if csv_file is not None:
                try:
                    df_in = pd.read_csv(csv_file)
                    X_aligned = align_to_schema(df_in, active_schema, defaults)
                    st.write("Input preview:")
                    st.dataframe(X_aligned.head(8), use_container_width=True)
                    row_idx = st.number_input("Row index to use", min_value=0, max_value=max(0, len(X_aligned)-1), value=0, step=1)
                    if len(X_aligned) > 0:
                        prefill_row = X_aligned.iloc[int(row_idx)]
                except Exception as e:
                    st.error(f"Failed to read/align CSV: {e}")

        if simplified and mdl is not None:
            top_feats = infer_top_features(mdl, active_schema, k=top_k)
            st.markdown(f"""
            <div class="feature-guidance">
                <span class="material-symbols-outlined">fact_check</span>
                <strong>Schema Detected:</strong> Editing Top‑{len(top_feats)} features out of {len(feature_names)}. Others use defaults.
            </div>
            """, unsafe_allow_html=True)

            base_vals = defaults.copy()
            if prefill_row is not None:
                for f in active_schema:
                    v = float(prefill_row.get(f, base_vals.get(f, 0.0)))
                    base_vals[f] = v

            cols = st.columns(3)
            edited = {}
            for i, feat in enumerate(top_feats):
                with cols[i % 3]:
                    edited[feat] = st.number_input(feat, value=float(base_vals.get(feat, 0.0)))

            row_dict = base_vals.copy()
            row_dict.update(edited)
            X_one = pd.DataFrame([row_dict], columns=active_schema)
            input_array = X_one.values
            shown_features = top_feats
        else:
            st.markdown(f"""
            <div class="feature-guidance">
                <span class="material-symbols-outlined">fact_check</span>
                <strong>Schema Detected:</strong> Found {len(active_schema)} features from the selected model. Please input values for each feature below.
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(3)
            values = []
            for i, feat in enumerate(active_schema):
                default_val = float(defaults.get(feat, 0.0))
                if prefill_row is not None:
                    default_val = float(prefill_row.get(feat, default_val))
                with cols[i % 3]:
                    values.append(st.number_input(feat, value=default_val))
            X_one = pd.DataFrame([values], columns=active_schema, dtype=float)
            input_array = X_one.values
            shown_features = active_schema

    if st.button("🚀 Predict", type="primary"):
        if not models:
            st.error("No model available for prediction.")
        else:
            mdl = models.get(model_name)
            if mdl is None:
                st.error("Selected model not found.")
            else:
                try:
                    core_model = unwrap_model(mdl)
                    if hasattr(core_model, "predict_proba"):
                        proba = core_model.predict_proba(input_array)[0]
                        p1 = float(proba[1] if len(proba) > 1 else proba[0])
                    elif hasattr(core_model, "decision_function"):
                        score = float(core_model.decision_function(input_array)[0])
                        r = pd.Series([score]).rank().values
                        p1 = float((r - r.min()) / (r.max() - r.min() + 1e-9))
                    else:
                        pred = int(core_model.predict(input_array)[0])
                        p1 = float(pred)
                    pred_label = int(p1 >= confidence_threshold)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    p1, pred_label = 0.0, 0

                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"Prediction: {pred_label}")
                    st.metric("Confidence (P1)", f"{p1:.2%}")
                with c2:
                    fig = px.bar(
                        x=['Class 0', 'Class 1'], 
                        y=[1-p1, p1], 
                        title="Prediction Probabilities",
                        color=[1-p1, p1],
                        color_continuous_scale=['#ef4444', '#10b981']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown("""
    <div class="content-intro">
        <h4><span class=\"material-symbols-outlined\">dataset</span> Batch Predictions</h4>
        <p>Upload a CSV file containing multiple rows of feature data to generate predictions in bulk. The CSV should have columns that match the features used to train the selected model.</p>
        <p><strong>Tip:</strong> Missing columns will be automatically filled with zeros. The system will process up to 10,000 rows efficiently.</p>
    </div>
    """, unsafe_allow_html=True)
    
    csv_file = st.file_uploader("Upload CSV", type=['csv'])
    if csv_file is not None:
        try:
            df_in = pd.read_csv(csv_file)
            st.write("Input preview:")
            st.dataframe(df_in.head(10), use_container_width=True)

            mdl = models.get(model_name)
            if mdl is None:
                st.error("Selected model not found.")
            else:
                # Resolve schema for the selected model specifically (ordered)
                model_schema = get_model_feature_schema(mdl) or feature_names
                if model_schema:
                    defaults = load_feature_defaults(model_schema)
                    # 1) Normalize column names from raw samples
                    df_norm = normalize_input_columns(df_in)
                    # Ensure unique columns after normalization
                    if df_norm.columns.duplicated().any():
                        df_norm = df_norm.loc[:, ~df_norm.columns.duplicated()]
                    # 2) Enrich with Sentinel aggregates using translated_lat/lon
                    df_enriched = enrich_with_satellite_features(df_norm)
                    # 3) Opportunistically compute missing engineered cols the model expects
                    df_aug = augment_with_engineered_features(df_enriched, model_schema)
                    if df_aug.columns.duplicated().any():
                        df_aug = df_aug.loc[:, ~df_aug.columns.duplicated()]
                    # 4) Align strictly to ordered schema
                    X = align_to_schema(df_aug, model_schema, defaults)
                    missing = [c for c in model_schema if c not in df_in.columns]
                    if missing:
                        st.warning(f"Missing columns filled with defaults: {missing[:10]}{'...' if len(missing)>10 else ''}")
                else:
                    X = df_in.select_dtypes(include=[np.number]).fillna(0.0)

                # Predict probabilities -> labels (CatBoost requires strict name/order; use Pool)
                core_model = unwrap_model(mdl)
                try:
                    import catboost  # type: ignore
                    is_catboost = core_model.__class__.__module__.startswith("catboost")
                except Exception:
                    catboost = None  # type: ignore
                    is_catboost = False

                if is_catboost:
                    Xc = X.copy()
                    # Ensure exact column order
                    Xc = Xc[model_schema]
                    pool = catboost.Pool(Xc, feature_names=list(Xc.columns))  # type: ignore
                    if hasattr(core_model, "predict_proba"):
                        probas = core_model.predict_proba(pool)
                        p1 = probas[:, 1] if probas.ndim == 2 and probas.shape[1] > 1 else probas.ravel()
                    else:
                        preds = core_model.predict(pool)
                        p1 = preds.astype(float)
                elif hasattr(core_model, "predict_proba"):
                    probas = core_model.predict_proba(X)
                    p1 = probas[:, 1] if probas.ndim == 2 and probas.shape[1] > 1 else probas.ravel()
                elif hasattr(core_model, "decision_function"):
                    scores = core_model.decision_function(X)
                    r = pd.Series(scores).rank().values
                    p1 = (r - r.min()) / (r.max() - r.min() + 1e-9)
                else:
                    preds = core_model.predict(X)
                    p1 = preds.astype(float)
                labels = (p1 >= confidence_threshold).astype(int)

                out = df_in.copy()
                out['prediction'] = labels
                out['confidence'] = p1

                st.success(f"Predicted {len(out)} rows")
                st.dataframe(out.head(100), use_container_width=True)

                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
