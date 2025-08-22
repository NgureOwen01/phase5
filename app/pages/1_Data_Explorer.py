import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Flexible import to support different working directories
try:
    from app.utils import (
        load_training_samples,
        load_csv_sample,
        load_css,
        normalize_input_columns,
        aggregate_s1,
        aggregate_s2,
        enrich_with_satellite_features,
        augment_with_engineered_features,
        get_feature_schema,
        load_feature_defaults,
        align_to_schema,
    )
except Exception:
    from utils import (  # type: ignore
        load_training_samples,
        load_csv_sample,
        load_css,
        normalize_input_columns,
        aggregate_s1,
        aggregate_s2,
        enrich_with_satellite_features,
        augment_with_engineered_features,
        get_feature_schema,
        load_feature_defaults,
        align_to_schema,
    )

st.set_page_config(page_title="Data Explorer", page_icon=None, layout="wide")

# Inject global CSS
css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
# Blur-to-focus text animation (same as main header)
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
      <span class="w" style="--i:0">Data</span>
      <span class="w" style="--i:1">Explorer</span>
    </span>
  </h1>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-divider"><span class="label">Ingest • Explore • Export</span></div>', unsafe_allow_html=True)

# Uploader and preprocessing pipeline
with st.expander("Upload and preprocess your data (CSV + optional shapefile + Sentinel‑1/2)", expanded=False):
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        user_csv = st.file_uploader("Upload points CSV (require translated_lat, translated_lon)", type=["csv"], key="ux_points")
        s1_csv = st.file_uploader("Optional: Sentinel‑1 CSV", type=["csv"], key="ux_s1")
        s2_csv = st.file_uploader("Optional: Sentinel‑2 CSV", type=["csv"], key="ux_s2")
    with up_col2:
        shp_main = st.file_uploader("Optional: Shapefile (.shp)", type=["shp"], key="ux_shp")
        shp_dbf = st.file_uploader("Optional: Shapefile DBF (.dbf)", type=["dbf"], key="ux_dbf")
        shp_shx = st.file_uploader("Optional: Shapefile SHX (.shx)", type=["shx"], key="ux_shx")
        shp_prj = st.file_uploader("Optional: Shapefile PRJ (.prj)", type=["prj"], key="ux_prj")

    run_pre = st.button("Run preprocessing", type="primary")
    pre_out = st.empty()
    if run_pre:
        try:
            # 1) Base points
            if user_csv is None:
                st.error("Please upload a points CSV with translated_lat/translated_lon.")
                st.stop()
            pts = pd.read_csv(user_csv)
            pts = normalize_input_columns(pts)
            if not {'translated_lat','translated_lon'}.issubset(pts.columns):
                st.error("CSV must contain translated_lat and translated_lon columns after normalization.")
                st.stop()

            # 2) Optional shapefile enrichment (nearest join for attributes)
            if shp_main is not None:
                try:
                    import tempfile
                    import geopandas as gpd
                    with tempfile.TemporaryDirectory() as td:
                        td = Path(td)
                        (td / "upload.shp").write_bytes(shp_main.read())
                        if shp_dbf: (td / "upload.dbf").write_bytes(shp_dbf.read())
                        if shp_shx: (td / "upload.shx").write_bytes(shp_shx.read())
                        if shp_prj: (td / "upload.prj").write_bytes(shp_prj.read())
                        gdf = gpd.read_file(td / "upload.shp")
                        if not gdf.empty and 'geometry' in gdf.columns:
                            gdf = gdf.to_crs(epsg=4326)
                            geo_pts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts['translated_lon'], pts['translated_lat']), crs='EPSG:4326')
                            joined = gpd.sjoin_nearest(geo_pts, gdf, how='left')
                            pts = pd.DataFrame(joined.drop(columns=['geometry','index_right'], errors='ignore'))
                except Exception:
                    pass

            # 3) Build S1/S2 aggregates from uploads if provided
            s1_agg = pd.DataFrame(); s2_agg = pd.DataFrame()
            if s1_csv is not None:
                try:
                    s1_raw = pd.read_csv(s1_csv)
                    s1_agg = aggregate_s1(s1_raw)
                except Exception:
                    pass
            if s2_csv is not None:
                try:
                    s2_raw = pd.read_csv(s2_csv)
                    s2_agg = aggregate_s2(s2_raw)
                except Exception:
                    pass

            # 4) Enrich with satellite features
            enriched = enrich_with_satellite_features(pts, s1_agg=s1_agg if not s1_agg.empty else None,
                                                          s2_agg=s2_agg if not s2_agg.empty else None)

            # 5) Engineer features and align to schema if available
            feature_schema = get_feature_schema({})
            engineered = augment_with_engineered_features(enriched, feature_schema)
            if feature_schema:
                defaults = load_feature_defaults(feature_schema)
                X = align_to_schema(engineered, feature_schema, defaults)
            else:
                X = engineered.select_dtypes(include=[np.number]).copy()
                X = X.loc[:, ~X.columns.duplicated()]

            pre_out.success(f"Preprocessing complete: {len(X)} rows, {X.shape[1]} features")
            st.dataframe(X.head(200), use_container_width=True)
            csv_x = X.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download preprocessed features (CSV)", data=csv_x, file_name="preprocessed_features.csv", mime="text/csv")
        except Exception as e:
            pre_out.error(f"Preprocessing failed: {e}")

st.markdown("""
<div class="content-intro">
    <h4><span class="material-symbols-outlined">search</span> Explore Your Agricultural Data</h4>
    <p>Dive deep into your cropland mapping datasets with interactive visualizations and statistical insights. Use the sidebar controls to filter and customize your analysis.</p>
    <p>💡 <strong>Features:</strong> Sample data for performance, visualize relationships, and export filtered datasets for further analysis.</p>
</div>
""", unsafe_allow_html=True)

train_df, _ = load_training_samples()

# Sidebar controls
with st.sidebar:
    st.subheader("Filters")
    sample_size = st.slider("Sample Size", 500, 50_000, 5_000, step=500)
    add_train = st.checkbox("Include training features", value=not train_df.empty)
    add_s1 = st.checkbox("Include Sentinel-1 sample", value=True)
    add_s2 = st.checkbox("Include Sentinel-2 sample", value=True)
    st.markdown("---")
    target_filter = None
    region_filter = None
    if add_train and not train_df.empty:
        if 'Cropland' in train_df.columns:
            target_filter = st.selectbox("Filter by Cropland", options=["All", 0, 1], index=0)
        if 'region' in train_df.columns:
            regions = ["All"] + sorted(list(map(str, train_df['region'].dropna().unique())))
            region_filter = st.selectbox("Filter by Region", options=regions, index=0)

# Prepare datasets with prefixes to avoid duplicate columns
frames = []
if add_train and not train_df.empty:
    df_t = train_df.copy()
    if target_filter in [0, 1]:
        df_t = df_t[df_t.get('Cropland') == target_filter]
    if region_filter and region_filter != "All" and 'region' in df_t.columns:
        df_t = df_t[df_t['region'].astype(str) == str(region_filter)]
    frames.append(df_t.add_prefix("train_"))

if add_s1:
    s1 = load_csv_sample("Sentinel1.csv", nrows=100_000)
    if not s1.empty:
        frames.append(s1.add_prefix("s1_"))

if add_s2:
    s2 = load_csv_sample("Sentinel2.csv", nrows=100_000)
    if not s2.empty:
        frames.append(s2.add_prefix("s2_"))

frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
if frames:
    data = pd.concat([f.reset_index(drop=True) for f in frames], axis=1)
else:
    data = pd.DataFrame()

# Keep only numeric columns for plots
num_data = data.select_dtypes(include=[np.number]).copy()
if num_data.empty:
    st.markdown("""
    <div class="feature-guidance">
        <span class="material-symbols-outlined">insights</span>
        <strong>No Data Available:</strong> No numeric columns found in the selected datasets. Please check your data sources or adjust the filters in the sidebar.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Downsample
if len(num_data) > sample_size:
    sampled = num_data.sample(n=sample_size, random_state=42)
else:
    sampled = num_data

# Ensure unique columns (belt and suspenders)
sampled = sampled.loc[:, ~sampled.columns.duplicated()]

# Helper: curated feature descriptions
S2_BAND_NAMES = {
    'B1': 'Aerosol (Coastal)', 'B2': 'Blue', 'B3': 'Green', 'B4': 'Red',
    'B5': 'Red Edge 1', 'B6': 'Red Edge 2', 'B7': 'Red Edge 3', 'B8': 'NIR',
    'B8A': 'NIR Narrow', 'B9': 'Water Vapor', 'B11': 'SWIR 1', 'B12': 'SWIR 2'
}

def describe_feature(col: str) -> str:
    lower = col.lower()
    # Sentinel-1
    if lower.startswith('s1_vh') or lower.endswith('_vh'):
        return 'Sentinel-1 SAR backscatter VH polarization (mean-aggregated). Units typically in dB.'
    if lower.startswith('s1_vv') or lower.endswith('_vv'):
        return 'Sentinel-1 SAR backscatter VV polarization (mean-aggregated). Units typically in dB.'
    # Sentinel-2 bands
    if lower.startswith('s2_b'):
        band = col.split('_')[-1].upper()
        label = S2_BAND_NAMES.get(band, band)
        return f'Sentinel-2 {label} reflectance (mean-aggregated across observations).'
    if lower.endswith('cloud_pct'):
        return 'Sentinel-2 Cloud percentage over the observation area (0–100).'
    if 'solar_azimuth' in lower:
        return 'Sentinel-2 Solar azimuth angle in degrees.'
    if 'solar_zenith' in lower:
        return 'Sentinel-2 Solar zenith angle in degrees.'
    # Training
    if lower.startswith('train_longitude'):
        return 'Training sample geographic longitude in degrees.'
    if lower.startswith('train_latitude'):
        return 'Training sample geographic latitude in degrees.'
    if lower.startswith('train_cropland'):
        return 'Training label: Cropland (1) vs Non-cropland (0).'
    # Fallback
    return 'No curated description available. See statistical summary below.'


tabs = st.tabs(["Overview", "Plots", "Correlation"]) 

with tabs[0]:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Rows", f"{len(sampled):,}")
    with colB:
        st.metric("Numeric features", f"{sampled.shape[1]:,}")
    with colC:
        st.metric("Missing (any)", f"{int(sampled.isnull().any(axis=1).sum()):,}")
    with colD:
        st.metric("Complete rows", f"{int(sampled.dropna().shape[0]):,}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Data Preview")
    st.dataframe(sampled.head(200), use_container_width=True, height=360)

    st.markdown("""
    <div class="feature-guidance">
        <span class="material-symbols-outlined">download</span>
        <strong>Export Data:</strong> Download the current filtered and sampled dataset for external analysis or model training.
    </div>
    """, unsafe_allow_html=True)
    csv = sampled.to_csv(index=False).encode('utf-8')
    st.download_button("Download sampled data (CSV)", data=csv, file_name="data_sample.csv", mime="text/csv")

with tabs[1]:
    st.subheader("Visualizations")

    # Feature description dropdown (themed) and stats panel
    cols = list(sampled.columns)
    desc_feat = st.selectbox("Feature description", options=cols, index=0, key="desc_feature")

    with st.container():
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        left, right = st.columns([2, 3])
        with left:
            st.markdown(f"### {desc_feat}")
            st.markdown(describe_feature(desc_feat))
        with right:
            series = sampled[desc_feat].dropna()
            if not series.empty:
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("count", f"{len(series):,}")
                miss_pct = 100 * (1 - len(series) / len(sampled))
                c2.metric("missing %", f"{miss_pct:.1f}%")
                c3.metric("mean", f"{series.mean():.4f}")
                c4.metric("std", f"{series.std():.4f}")
                c5.metric("min", f"{series.min():.4f}")
                c6.metric("max", f"{series.max():.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Plot controls
    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox("X Axis", options=cols, index=0, key="dx")
    with c2:
        y_col = st.selectbox("Y Axis", options=cols, index=min(1, len(cols)-1), key="dy")
    with c3:
        color_col = st.selectbox("Color (optional)", options=[None] + cols, index=0, key="dc")
    opacity = st.slider("Point Opacity", 0.1, 1.0, 0.6, 0.05)
    kind = st.radio("Chart", ["Scatter", "Histogram", "Box", "Density"], horizontal=True)

    if kind == "Scatter":
        fig = px.scatter(sampled, x=x_col, y=y_col, color=color_col, opacity=opacity)
    elif kind == "Histogram":
        bins = st.slider("Bins", 10, 120, 40, 5)
        fig = px.histogram(sampled, x=x_col, color=color_col, nbins=bins, opacity=opacity)
    elif kind == "Box":
        fig = px.box(sampled, x=color_col, y=x_col) if color_col else px.box(sampled, y=x_col)
    else:
        fig = px.density_heatmap(sampled, x=x_col, y=y_col, nbinsx=40, nbinsy=40, color_continuous_scale='Viridis')

    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Correlation Heatmap")
    top_n = st.slider("Top features by variance", 5, min(50, sampled.shape[1]), min(20, sampled.shape[1]))
    variances = sampled.var(numeric_only=True).sort_values(ascending=False)
    top_cols = list(variances.head(top_n).index)
    corr = sampled[top_cols].corr()
    fig = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)
