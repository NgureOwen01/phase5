import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Flexible import to support different working directories
try:
    from app.utils import load_models, get_feature_schema, format_number, load_css
except Exception:
    from utils import load_models, get_feature_schema, format_number, load_css  # type: ignore

st.set_page_config(page_title="🔮 Predictions", page_icon="🔮", layout="wide")

# Inject global CSS
css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.title("🔮 Make Predictions")
st.markdown('<div class="section-divider"><span class="label">Inputs • Confidence • Actions</span></div>', unsafe_allow_html=True)

models = load_models()
if not models:
    st.warning("No saved models found in 'models/'. Place *.pkl or *.joblib files there.")

feature_names = get_feature_schema(models)

with st.sidebar:
    st.subheader("Settings")
    model_name = st.selectbox("Model", options=list(models.keys()) or ["-"], index=0)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Tabs for single vs batch
tabs = st.tabs(["Single Input", "Batch (CSV)"])

with tabs[0]:
    st.markdown("""
    <div class="content-intro">
        <h4>🎯 Single Prediction</h4>
        <p>Enter the feature values below to receive a prediction from the selected machine learning model. The model will output a classification (0 or 1) and a confidence score based on your inputs.</p>
        <p>💡 <strong>Tip:</strong> Adjust the 'Confidence Threshold' in the sidebar to fine-tune the classification sensitivity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="modern-form">', unsafe_allow_html=True)
    if not feature_names:
        st.markdown("""
        <div class="feature-guidance">
            <span class="icon">⚠️</span>
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
        st.markdown(f"""
        <div class="feature-guidance">
            <span class="icon">✅</span>
            <strong>Schema Detected:</strong> Found {len(feature_names)} features from the trained model. Please input values for each feature below.
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(3)
        values = []
        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                values.append(st.number_input(feat, value=0.0, help=f"Enter value for {feat}"))
        input_array = np.array([values], dtype=float)
        shown_features = feature_names

    if st.button("🚀 Predict", type="primary"):
        if not models:
            st.error("No model available for prediction.")
        else:
            mdl = models.get(model_name)
            if mdl is None:
                st.error("Selected model not found.")
            else:
                try:
                    if hasattr(mdl, "predict_proba"):
                        proba = mdl.predict_proba(input_array)[0]
                        p1 = float(proba[1] if len(proba) > 1 else proba[0])
                    elif hasattr(mdl, "decision_function"):
                        score = float(mdl.decision_function(input_array)[0])
                        r = pd.Series([score]).rank().values
                        p1 = float((r - r.min()) / (r.max() - r.min() + 1e-9))
                    else:
                        pred = int(mdl.predict(input_array)[0])
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
        <h4>📊 Batch Predictions</h4>
        <p>Upload a CSV file containing multiple rows of feature data to generate predictions in bulk. The CSV should have columns that match the features used to train the selected model.</p>
        <p>💡 <strong>Tip:</strong> Missing columns will be automatically filled with zeros. The system will process up to 10,000 rows efficiently.</p>
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
                # Align columns if schema is available
                if feature_names:
                    missing = [c for c in feature_names if c not in df_in.columns]
                    if missing:
                        st.warning(f"Missing columns filled with 0: {missing[:10]}{'...' if len(missing)>10 else ''}")
                    X = df_in.reindex(columns=feature_names).fillna(0.0)
                else:
                    X = df_in.select_dtypes(include=[np.number]).fillna(0.0)

                # Predict probabilities -> labels
                if hasattr(mdl, "predict_proba"):
                    probas = mdl.predict_proba(X)
                    p1 = probas[:, 1] if probas.ndim == 2 and probas.shape[1] > 1 else probas.ravel()
                elif hasattr(mdl, "decision_function"):
                    scores = mdl.decision_function(X)
                    r = pd.Series(scores).rank().values
                    p1 = (r - r.min()) / (r.max() - r.min() + 1e-9)
                else:
                    preds = mdl.predict(X)
                    p1 = preds.astype(float)
                labels = (p1 >= confidence_threshold).astype(int)

                out = df_in.copy()
                out['prediction'] = labels
                out['confidence'] = p1

                st.success(f"Predicted {len(out)} rows")
                st.dataframe(out.head(100), use_container_width=True)

                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
