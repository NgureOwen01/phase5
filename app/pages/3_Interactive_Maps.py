import streamlit as st
import os
import folium
from streamlit_folium import st_folium
import geopandas as gpd

# Flexible import to support different working directories
try:
    from app.utils import load_training_samples, load_css
except Exception:
    from utils import load_training_samples, load_css  # type: ignore

st.set_page_config(page_title="Interactive Maps", page_icon=None, layout="wide")

# Inject global CSS
css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
# Blur-to-focus text animation (reuse)
blurtext_css = """
.blur-reveal { display: inline-block; }
.blur-reveal .w {
  display: inline-block;
  filter: blur(12px);
  opacity: 0;
  transform: translateY(0.4em);
  animation: br-reveal 0.9s cubic-bezier(0.2,  0.7, 0.2, 1) forwards;
  animation-delay: calc(var(--base, 0s) + (var(--i, 0) * var(--stagger, 0.06s)));
}
@keyframes br-reveal { to { filter: blur(0); opacity: 1; transform: translateY(0); } }
"""
st.markdown(f"<style>{blurtext_css}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1 style="margin-bottom:0.25rem;">
    <span class="blur-reveal" style="--stagger:.06s; --base:.33s;">
      <span class="w" style="--i:0">Interactive</span>
      <span class="w" style="--i:1">Cropland</span>
      <span class="w" style="--i:2">Maps</span>
    </span>
  </h1>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-divider"><span class="label">Layers • Context • Insight</span></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="content-intro">
        <h4><span class=\"material-symbols-outlined\">map</span> Geospatial Data Visualization</h4>
        <p>Explore cropland training data on interactive maps with clustering and heatmaps. Click the map to get exact coordinates.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<h3 class="sticky-label"><span class="material-symbols-outlined">tune</span> Map Controls</h3>', unsafe_allow_html=True)
    zoom = st.slider("Zoom", 2, 12, 5)
    basemap_label = st.selectbox(
        "Basemap",
        [
            "OpenStreetMap",
            "Stamen Terrain",
            "CartoDB Positron",
            "CartoDB Dark Matter",
            "Esri World Imagery",
        ],
        index=0,
    )
    show_points = st.checkbox("Show training points", True)
    use_cluster = st.checkbox("Cluster points", True)
    show_heatmap = st.checkbox("Show heatmap", False)

train_df, train_gdf = load_training_samples()

# Center map from data if available
center = [41.3, 69.3]
if train_gdf is not None and not train_gdf.empty:
    try:
        center = [float(train_gdf.geometry.y.mean()), float(train_gdf.geometry.x.mean())]
    except Exception:
        pass

# Explicit tile URL templates and attributions
TILE_SOURCES = {
    "OpenStreetMap": (
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors",
    ),
    # Stamen Terrain via Stadia Maps mirror (public tiles, proper attribution)
    "Stamen Terrain": (
        "https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png",
        "Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors — Tiles courtesy of Stadia Maps",
    ),
    "CartoDB Positron": (
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "© OpenStreetMap contributors © CARTO",
    ),
    "CartoDB Dark Matter": (
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "© OpenStreetMap contributors © CARTO",
    ),
    "Esri World Imagery": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Tiles © Esri — Source: Esri, Earthstar Geographics, and the GIS User Community",
    ),
}

# Track settings to avoid unnecessary recompute
settings_tuple = (basemap_label, show_points, use_cluster, show_heatmap)
if st.session_state.get("_map_prev_settings") != settings_tuple:
    st.session_state["_map_prev_settings"] = settings_tuple
    st.session_state.pop("_map_sample" , None)
    st.session_state.pop("_map_bounds_fitted" , None)

# Build map in left column, info on right
map_col, info_col = st.columns([3, 2])

with map_col:
    # Start map with no base tiles, then add the selected provider explicitly
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)

    # Add selected base layer using explicit URL and attribution
    # Handle Stamen Terrain (Stadia Maps) API key requirement in production
    if basemap_label == "Stamen Terrain":
        api_key = os.getenv("STADIA_API_KEY") or os.getenv("STADIA_MAPS_API_KEY")
        if api_key:
            tiles_url = f"https://tiles.stadiamaps.com/tiles/stamen_terrain/{{z}}/{{x}}/{{y}}{{r}}.png?api_key={api_key}"
            tiles_attr = (
                "Tiles courtesy of Stadia Maps — Map tiles by Stamen Design, CC BY 3.0 — "
                "Map data © OpenStreetMap contributors"
            )
        else:
            st.warning(
                "Stamen Terrain requires a Stadia Maps API key (set STADIA_API_KEY). "
                "Falling back to CartoDB Positron."
            )
            tiles_url, tiles_attr = TILE_SOURCES.get("CartoDB Positron", TILE_SOURCES["OpenStreetMap"]) 
    else:
        tiles_url, tiles_attr = TILE_SOURCES.get(basemap_label, TILE_SOURCES["OpenStreetMap"]) 
    folium.TileLayer(tiles=tiles_url, name=basemap_label, attr=tiles_attr).add_to(m)

    # Fit to data bounds once per settings change
    if train_gdf is not None and not train_gdf.empty and not st.session_state.get("_map_bounds_fitted"):
        try:
            minx, miny, maxx, maxy = train_gdf.total_bounds
            m.fit_bounds([[miny, minx], [maxy, maxx]])
            st.session_state["_map_bounds_fitted"] = True
        except Exception:
            pass

    # Cached sample of points for stable rendering
    displayed_points = 0
    if show_points and train_gdf is not None and not train_gdf.empty:
        try:
            if "_map_prev_settings" not in st.session_state or st.session_state.get("_map_prev_settings") != (basemap_label, show_points, use_cluster, show_heatmap):
                st.session_state["_map_prev_settings"] = (basemap_label, show_points, use_cluster, show_heatmap)
                st.session_state.pop("_map_sample", None)
            if "_map_sample" not in st.session_state:
                sample_n = min(3000 if use_cluster else 1500, len(train_gdf))
                st.session_state["_map_sample"] = train_gdf.sample(sample_n, random_state=42)
            gdf_sample = st.session_state["_map_sample"]
            displayed_points = len(gdf_sample)

            if use_cluster:
                from folium.plugins import MarkerCluster
                cluster = MarkerCluster().add_to(m)
                for _, row in gdf_sample.iterrows():
                    coords = row.geometry
                    if coords is not None:
                        folium.CircleMarker(
                            location=[coords.y, coords.x], radius=2,
                            color="#2a5298" if int(row.get("Cropland", 0)) == 1 else "#999999",
                            fill=True, fill_opacity=0.7
                        ).add_to(cluster)
            else:
                for _, row in gdf_sample.iterrows():
                    coords = row.geometry
                    if coords is not None:
                        folium.CircleMarker(
                            location=[coords.y, coords.x], radius=2,
                            color="#2a5298" if int(row.get("Cropland", 0)) == 1 else "#999999",
                            fill=True, fill_opacity=0.7
                        ).add_to(m)
        except Exception:
            pass

    # Optional heatmap
    if show_heatmap and train_gdf is not None and not train_gdf.empty:
        try:
            from folium.plugins import HeatMap
            coords = [[g.y, g.x] for g in train_gdf.geometry.sample(min(5000, len(train_gdf)), random_state=42).values]
            HeatMap(coords, radius=10, blur=15, min_opacity=0.2).add_to(m)
        except Exception:
            pass

    folium.LayerControl(collapsed=True).add_to(m)

    # Render responsively with a stable component key to minimize re-mounting
    map_data = st_folium(m, height=620, width=None, use_container_width=True, key="main_map")

with info_col:
    st.subheader("📍 Location Info")
    st.markdown(
        """
        <div class="feature-guidance"><span class="icon">🎯</span>
        Click the map to view coordinates and download them as CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if map_data and map_data.get('last_clicked'):
        lat = map_data['last_clicked']['lat']
        lng = map_data['last_clicked']['lng']
        st.write(f"Lat: {lat:.5f}, Lng: {lng:.5f}")
        st.download_button(
            label="⬇️ Download clicked point (CSV)",
            data=f"lat,lng\n{lat},{lng}\n",
            file_name="clicked_point.csv",
            mime="text/csv"
        )
    else:
        st.write("Click anywhere on the map to view coordinates.")

    st.markdown("---")
    st.subheader("📊 Dataset Summary")
    if train_gdf is not None and not train_gdf.empty:
        total = len(train_gdf)
        cropland = int((train_gdf.get('Cropland', 0) == 1).sum()) if 'Cropland' in train_gdf.columns else None
        st.metric("Points displayed", f"{displayed_points:,}")
        st.metric("Total training points", f"{total:,}")
        if cropland is not None:
            non_cropland = total - cropland
            st.metric("Cropland points", f"{cropland:,}")
            st.metric("Non-cropland points", f"{non_cropland:,}")
    else:
        st.markdown(
            """
            <div class="feature-guidance"><span class="icon">📍</span>
            No geographic training samples found.
            </div>
            """,
            unsafe_allow_html=True,
        )
