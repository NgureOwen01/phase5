import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

try:
    import streamlit as st
except Exception:  # Allow non-streamlit contexts
    class _Dummy:
        def cache_data(self, func=None, **kwargs):
            def wrapper(f):
                return f
            return wrapper
        def cache_resource(self, func=None, **kwargs):
            def wrapper(f):
                return f
            return wrapper
        def session_state(self):
            return {}
        def markdown(self, *args, **kwargs):
            pass
    st = _Dummy()  # type: ignore


ROOT_DIR = Path.cwd()
TRAIN_DIR = ROOT_DIR / "Train"
CACHE_DIR = ROOT_DIR / "cache"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ASSETS_DIR = ROOT_DIR / "app" / "assets"

# Remote dataset configuration (Hugging Face Hub)
# You can override via environment variables if needed
USE_HF_DATA = os.getenv("USE_HF_DATA", "1") != "0"
HF_DATA_REPO_ID = os.getenv("HF_DATA_REPO", "cnyagaka/satellite-data")
HF_DATA_SUBDIR = os.getenv("HF_DATA_SUBDIR", "data")  # files live under this prefix in the repo

CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True, parents=True)


def _try_hf_download(relative_path: str) -> Optional[Path]:
    """Download a file from the configured HF dataset repo and return the cached local path.

    relative_path is relative to the dataset's data subdirectory (e.g., 'Sentinel1.csv' or 'Train/xxx.shp').
    """
    if not USE_HF_DATA:
        return None
    try:
        # Import locally to avoid hard dependency if unused
        from huggingface_hub import hf_hub_download  # type: ignore

        path_in_repo = f"{HF_DATA_SUBDIR}/{relative_path}".replace("\\", "/")
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO_ID,
            repo_type="dataset",
            filename=path_in_repo,
        )
        return Path(downloaded)
    except Exception:
        return None


def _find_file(filename: str) -> Optional[Path]:
    """Resolve a file by checking local paths first, then Hugging Face dataset if enabled.

    The filename can include subdirectories (e.g., 'Test/Test.csv').
    """
    # Local candidates
    candidates = [
        ROOT_DIR / filename,
        ROOT_DIR / "data" / filename,  # optional local 'data' dir if present
        TRAIN_DIR / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    # Remote (HF Hub) fallback
    return _try_hf_download(filename)


@st.cache_data(show_spinner=False)
def load_training_samples() -> Tuple[pd.DataFrame, Optional[gpd.GeoDataFrame]]:
    """Load and combine Fergana and Orenburg training samples. Returns (df, gdf).

    If local files are not present, fetch shapefile components from the configured
    Hugging Face dataset repo under 'data/Train/'.
    """
    def resolve_shapefile(base_name: str) -> Optional[Path]:
        # 1) Local path
        local_shp = TRAIN_DIR / f"{base_name}.shp"
        if local_shp.exists():
            return local_shp
        # 2) Remote via HF Hub
        shp_path = None
        if USE_HF_DATA:
            # Ensure supporting files are cached too
            for ext in [".shp", ".dbf", ".shx", ".prj"]:
                _ = _try_hf_download(f"Train/{base_name}{ext}")
                if ext == ".shp" and _ is not None:
                    shp_path = _
        return shp_path

    gdfs: List[gpd.GeoDataFrame] = []
    for base in ["Fergana_training_samples", "Orenburg_training_samples"]:
        shp = resolve_shapefile(base)
        if shp is not None and shp.exists():
            try:
                gdfs.append(gpd.read_file(shp))
            except Exception:
                continue
    if not gdfs:
        return pd.DataFrame(), None

    # Label regions if available
    for gdf, region in zip(gdfs, ["Fergana", "Orenburg"]):
        if "region" not in gdf.columns:
            gdf["region"] = region

    gdf_all = pd.concat(gdfs, ignore_index=True)
    df = pd.DataFrame(gdf_all.drop(columns=["geometry"]))

    # Ensure lon/lat
    if "translated_lon" not in df.columns and hasattr(gdf_all, "geometry"):
        df["translated_lon"] = gdf_all.geometry.x
    if "translated_lat" not in df.columns and hasattr(gdf_all, "geometry"):
        df["translated_lat"] = gdf_all.geometry.y

    # Keep essential cols if present
    for col in ["longitude", "latitude"]:
        if col not in df.columns:
            if col == "longitude" and "translated_lon" in df.columns:
                df[col] = df["translated_lon"]
            if col == "latitude" and "translated_lat" in df.columns:
                df[col] = df["translated_lat"]

    return df, gdf_all


@st.cache_data(show_spinner=False)
def load_csv_sample(name: str, usecols: Optional[List[str]] = None, nrows: Optional[int] = 100_000) -> pd.DataFrame:
    """Load a CSV by name from local paths or the HF dataset. Optionally sample first nrows for performance."""
    path = _find_file(name)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, nrows=nrows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_parquet_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def list_model_files() -> List[Path]:
    patterns = ["*.pkl", "*.joblib"]
    out: List[Path] = []
    for pat in patterns:
        out.extend(list(MODELS_DIR.glob(pat)))
        out.extend(list(ARTIFACTS_DIR.glob(pat)))
    return sorted(out)


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, object]:
    import joblib
    models: Dict[str, object] = {}
    for p in list_model_files():
        try:
            models[p.stem] = joblib.load(p)
        except Exception:
            continue
    return models


@st.cache_data(show_spinner=False)
def load_metrics() -> Dict:
    metrics_path = CACHE_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            return pd.read_json(metrics_path).to_dict(orient="list")  # type: ignore
        except Exception:
            try:
                import json
                return json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def load_css() -> str:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        try:
            return css_path.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def inject_css_block(css: str) -> None:
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_feature_schema(models: Dict[str, object]) -> List[str]:
    """Attempt to infer feature names from a saved schema or from a model."""
    schema_json = CACHE_DIR / "final_features.json"
    if schema_json.exists():
        try:
            import json
            return json.loads(schema_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Try scikit-learn feature_names_in_
    for mdl in models.values():
        feats = getattr(mdl, "feature_names_in_", None)
        if feats is not None:
            return list(map(str, feats))
    return []


def get_model_feature_schema(model: object) -> List[str]:
    """Return an ordered feature schema for a specific model if available.

    Resolution order:
    1) If model is a dict with {'features': [...]} (e.g., champion bundle), use that
    2) scikit-learn: feature_names_in_
    3) CatBoost: feature_names_ or get_feature_names()
    4) Fallback: cache/final_features.json if present
    """
    # Unwrap possible champion bundle
    underlying = model.get('model') if isinstance(model, dict) and 'model' in model else model

    # 1) scikit-learn style feature_names_in_
    feats = getattr(underlying, 'feature_names_in_', None)
    if feats is not None:
        try:
            return list(map(str, feats))
        except Exception:
            pass

    # 2) LightGBM feature names
    try:
        lgb_feats = getattr(underlying, 'feature_name_', None)
        if not lgb_feats and hasattr(underlying, 'booster_'):
            booster = getattr(underlying, 'booster_', None)
            if booster is not None and hasattr(booster, 'feature_name'):
                lgb_feats = booster.feature_name()
        if lgb_feats:
            return list(map(str, lgb_feats))
    except Exception:
        pass

    # 3) XGBoost feature names
    try:
        xgb_feats = getattr(underlying, 'feature_names_in_', None)
        if not xgb_feats and hasattr(underlying, 'get_booster'):
            booster = underlying.get_booster()
            xgb_feats = getattr(booster, 'feature_names', None)
        if xgb_feats:
            return list(map(str, xgb_feats))
    except Exception:
        pass

    # 4) CatBoost native
    try:
        cb_feats = getattr(underlying, 'feature_names_', None)
        if cb_feats is None and hasattr(underlying, 'get_feature_names'):
            cb_feats = underlying.get_feature_names()  # type: ignore
        if cb_feats:
            return list(map(str, cb_feats))
    except Exception:
        pass

    # 5) Champion-style bundle explicit list (last resort)
    if isinstance(model, dict):
        feats = model.get('features')
        if isinstance(feats, list) and feats:
            return list(map(str, feats))

    # 6) Fallback to global schema file
    try:
        import json
        schema_json = CACHE_DIR / 'final_features.json'
        if schema_json.exists():
            return json.loads(schema_json.read_text(encoding='utf-8'))
    except Exception:
        pass
    return []


def unwrap_model(model: object) -> object:
    """Return the underlying estimator when a champion bundle dict is provided."""
    if isinstance(model, dict) and 'model' in model:
        return model['model']
    return model


def load_feature_defaults(feature_names: List[str]) -> Dict[str, float]:
    """Return per-feature default values (medians where possible, else 0.0)."""
    if not feature_names:
        return {}
    # Optional precomputed defaults
    defaults_path = CACHE_DIR / "feature_defaults.json"
    if defaults_path.exists():
        try:
            import json
            d = json.loads(defaults_path.read_text(encoding="utf-8"))
            return {k: float(d.get(k, 0.0)) for k in feature_names}
        except Exception:
            pass
    # Try compute medians from artifacts/merged_data.parquet
    df = load_parquet_if_exists(ARTIFACTS_DIR / "merged_data.parquet")
    if not df.empty:
        try:
            med = (
                df.reindex(columns=feature_names)
                  .select_dtypes(include=[np.number])
                  .median(numeric_only=True)
            )
            med = med.reindex(index=feature_names).fillna(0.0)
            return {k: float(med.get(k, 0.0)) for k in feature_names}
        except Exception:
            pass
    # Fallback zeros
    return {k: 0.0 for k in feature_names}


def align_to_schema(df: pd.DataFrame, feature_names: List[str], defaults: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Reindex to schema, coerce numeric, fill missing with provided defaults or 0.0."""
    if df is None or df.empty or not feature_names:
        return pd.DataFrame(columns=feature_names)
    out = df.reindex(columns=feature_names)
    out = out.apply(pd.to_numeric, errors="coerce")
    if defaults:
        for c in feature_names:
            fill_val = defaults.get(c, 0.0)
            out[c] = out[c].fillna(fill_val)
    else:
        out = out.fillna(0.0)
    return out


def format_number(x: float, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


# Lottie helpers removed


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common source column name patterns to match model expectations.

    - Convert s2_* → S2_* and s1_* → S1_* (case-insensitive)
    - Map raw band names like B2 → S2_B2 if S2_B2 missing
    - Standardize translated_lat/lon
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    # 0) Consolidate coordinates before renaming to avoid duplicate labels later
    def _pick_preferred(cols: List[str]) -> Optional[str]:
        if not cols:
            return None
        # Prefer s2_ → S2_ sources, then any
        for pref in [lambda c: c.lower().startswith('s2_'), lambda c: True]:
            for c in cols:
                if pref(c):
                    return c
        return cols[0]

    lat_candidates = [c for c in out.columns if c.lower().endswith('translated_lat')]
    lon_candidates = [c for c in out.columns if c.lower().endswith('translated_lon')]
    # Create unified translated_lat
    if 'translated_lat' not in out.columns:
        src = _pick_preferred(lat_candidates)
        if src is not None:
            out['translated_lat'] = out[src]
    # Create unified translated_lon
    if 'translated_lon' not in out.columns:
        src = _pick_preferred(lon_candidates)
        if src is not None:
            out['translated_lon'] = out[src]
    # Drop other translated_* sources to prevent duplicates after renaming
    drop_cols = [c for c in lat_candidates + lon_candidates if c not in {'translated_lat','translated_lon'}]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors='ignore')
    # Case-normalize
    new_cols: Dict[str, str] = {}
    for c in out.columns:
        c_new = c
        if c.lower().startswith('s2_'):
            c_new = 'S2_' + c[3:]
        if c.lower().startswith('s1_'):
            c_new = 'S1_' + c[3:]
        if c.lower() in {'translated_lat', 's2_translated_lat', 'train_translated_lat'}:
            c_new = 'translated_lat'
        if c.lower() in {'translated_lon', 's2_translated_lon', 'train_translated_lon'}:
            c_new = 'translated_lon'
        new_cols[c] = c_new
    out = out.rename(columns=new_cols)

    # Map plain band names B2→S2_B2, etc., if not already present
    band_names = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
    for b in band_names:
        if b in out.columns and f'S2_{b}' not in out.columns:
            out[f'S2_{b}'] = out[b]
    # SAR short names
    if 'VH' in out.columns and 'S1_VH' not in out.columns:
        out['S1_VH'] = out['VH']
    if 'VV' in out.columns and 'S1_VV' not in out.columns:
        out['S1_VV'] = out['VV']
    # Drop any duplicate columns created by renaming (keep first occurrence)
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    return out


def augment_with_engineered_features(df: pd.DataFrame, target_features: List[str]) -> pd.DataFrame:
    """Compute a subset of engineered features required by the model if base bands exist.

    Only computes features that appear in target_features to avoid unnecessary work.
    """
    if df is None or df.empty or not target_features:
        return df
    out = df.copy()
    to_num = lambda s: pd.to_numeric(s, errors='coerce')

    cols = set(out.columns)
    have = lambda *names: all(n in cols for n in names)

    # Base bands
    if have('S2_B8','S2_B4'):
        b8 = to_num(out['S2_B8']); b4 = to_num(out['S2_B4'])
        if 'NDVI_from_S2' in target_features and 'NDVI_from_S2' not in cols:
            out['NDVI_from_S2'] = (b8 - b4) / (b8 + b4 + 1e-10)
        if 'EVI2' in target_features and 'EVI2' not in cols:
            out['EVI2'] = 2.5 * (b8 - b4) / (b8 + 2.4 * b4 + 1)
        # Generic NDVI (+ transforms)
        if 'NDVI' in target_features and 'NDVI' not in cols:
            out['NDVI'] = (b8 - b4) / (b8 + b4 + 1e-10)
        # SAVI (L=0.5) variants
        savi05 = (1 + 0.5) * (b8 - b4) / (b8 + b4 + 0.5)
        if 'SAVI_from_S2' in target_features and 'SAVI_from_S2' not in cols:
            out['SAVI_from_S2'] = savi05
        if 'SAVI_L50' in target_features and 'SAVI_L50' not in cols:
            out['SAVI_L50'] = savi05
    if have('S2_B3','S2_B8'):
        b3 = to_num(out['S2_B3']); b8 = to_num(out['S2_B8'])
        if 'NDWI_from_S2' in target_features and 'NDWI_from_S2' not in cols:
            out['NDWI_from_S2'] = (b3 - b8) / (b3 + b8 + 1e-10)
        if 'GCI_from_S2' in target_features and 'GCI_from_S2' not in cols:
            out['GCI_from_S2'] = (b8 / (b3 + 1e-10)) - 1
        if 'ratio_B8_B3' in target_features and 'ratio_B8_B3' not in cols:
            out['ratio_B8_B3'] = b8 / (b3 + 1e-10)
        # Generic NDWI (+ transforms)
        if 'NDWI' in target_features and 'NDWI' not in cols:
            out['NDWI'] = (b3 - b8) / (b3 + b8 + 1e-10)
    if have('S2_B8','S2_B5'):
        b8 = to_num(out['S2_B8']); b5 = to_num(out['S2_B5'])
        if 'RECI_from_S2' in target_features and 'RECI_from_S2' not in cols:
            out['RECI_from_S2'] = (b8 / (b5 + 1e-10)) - 1
        if 'ratio_B5_B4' in target_features and 'ratio_B5_B4' not in cols and 'S2_B4' in out.columns:
            b4 = to_num(out['S2_B4']); out['ratio_B5_B4'] = b5 / (b4 + 1e-10)
    if have('S2_B5','S2_B4'):
        b5 = to_num(out['S2_B5']); b4 = to_num(out['S2_B4'])
        if 'NDRE' in target_features and 'NDRE' not in cols:
            out['NDRE'] = (b5 - b4) / (b5 + b4 + 1e-10)
    if have('S2_B12','S2_B8'):
        b12 = to_num(out['S2_B12']); b8 = to_num(out['S2_B8'])
        if 'NBR' in target_features and 'NBR' not in cols:
            out['NBR'] = (b8 - b12) / (b8 + b12 + 1e-10)
        if 'ratio_B12_B8' in target_features and 'ratio_B12_B8' not in cols:
            out['ratio_B12_B8'] = b12 / (b8 + 1e-10)
    if have('S2_B8','S2_B2'):
        b8 = to_num(out['S2_B8']); b2 = to_num(out['S2_B2'])
        if 'ratio_B8_B2' in target_features and 'ratio_B8_B2' not in cols:
            out['ratio_B8_B2'] = b8 / (b2 + 1e-10)

    # SAVI / MSAVI and transforms (require B8/B4)
    if have('S2_B8','S2_B4'):
        b8 = to_num(out['S2_B8']); b4 = to_num(out['S2_B4'])
        for L, key in [(0.25,'SAVI_L25'), (0.5,'SAVI_L50'), (0.75,'SAVI_L75')]:
            val = (1 + L) * (b8 - b4) / (b8 + b4 + L)
            if key in target_features and key not in out.columns:
                out[key] = val
            if f'{key}_log' in target_features and f'{key}_log' not in out.columns:
                out[f'{key}_log'] = np.log1p(np.maximum(val, 0))
            if f'{key}_squared' in target_features and f'{key}_squared' not in out.columns:
                out[f'{key}_squared'] = val ** 2
            if f'{key}_cubed' in target_features and f'{key}_cubed' not in out.columns:
                out[f'{key}_cubed'] = val ** 3
            if f'{key}_normalized' in target_features and f'{key}_normalized' not in out.columns:
                std = val.std()
                out[f'{key}_normalized'] = (val - val.mean()) / std if std and std > 0 else 0.0
        # MSAVI / MSAVI2
        msavi = (2 * b8 + 1 - np.sqrt((2 * b8 + 1)**2 - 8 * (b8 - b4))) / 2
        msavi2 = 0.5 * (2 * b8 + 1 - np.sqrt((2 * b8 + 1)**2 - 8 * (b8 - b4)))
        if 'MSAVI_from_S2' in target_features and 'MSAVI_from_S2' not in out.columns:
            out['MSAVI_from_S2'] = msavi
        if 'MSAVI2' in target_features and 'MSAVI2' not in out.columns:
            out['MSAVI2'] = msavi2
        if 'MSAVI_from_S2_cubed' in target_features and 'MSAVI_from_S2_cubed' not in out.columns:
            out['MSAVI_from_S2_cubed'] = msavi ** 3
        if 'MSAVI_from_S2_log' in target_features and 'MSAVI_from_S2_log' not in out.columns and (msavi > 0).any():
            out['MSAVI_from_S2_log'] = np.log1p(np.maximum(msavi, 0))
        if 'MSAVI_from_S2_normalized' in target_features and 'MSAVI_from_S2_normalized' not in out.columns:
            std = msavi.std()
            out['MSAVI_from_S2_normalized'] = (msavi - msavi.mean()) / std if std and std > 0 else 0.0
        if 'MSAVI2_squared' in target_features and 'MSAVI2_squared' not in out.columns:
            out['MSAVI2_squared'] = msavi2 ** 2

    # Transforms for NDVI/NDWI/EVI2/SAVI_from_S2 (_from_S2 and generic)
    for base_name in ['NDVI_from_S2','NDVI','NDWI_from_S2','NDWI','EVI2','SAVI_from_S2']:
        if base_name in out.columns:
            base = to_num(out[base_name])
            if f'{base_name}_log' in target_features and f'{base_name}_log' not in out.columns and (base > 0).any():
                out[f'{base_name}_log'] = np.log1p(np.maximum(base, 0))
            if f'{base_name}_sqrt' in target_features and f'{base_name}_sqrt' not in out.columns and (base > 0).any():
                out[f'{base_name}_sqrt'] = np.sqrt(np.maximum(base, 0))
            if f'{base_name}_squared' in target_features and f'{base_name}_squared' not in out.columns:
                out[f'{base_name}_squared'] = base ** 2
            if f'{base_name}_cubed' in target_features and f'{base_name}_cubed' not in out.columns:
                out[f'{base_name}_cubed'] = base ** 3
            if f'{base_name}_normalized' in target_features and f'{base_name}_normalized' not in out.columns:
                std = base.std()
                out[f'{base_name}_normalized'] = (base - base.mean()) / std if std and std > 0 else 0.0

    # Interaction terms between NDVI and NDWI variants
    ndvi_fs = out.get('NDVI_from_S2'); ndwi_fs = out.get('NDWI_from_S2')
    ndvi = out.get('NDVI'); ndwi = out.get('NDWI')
    def _num(s):
        return to_num(s) if isinstance(s, pd.Series) else None
    ndvi_fs = _num(ndvi_fs); ndwi_fs = _num(ndwi_fs); ndvi = _num(ndvi); ndwi = _num(ndwi)
    if ndvi_fs is not None and ndwi is not None:
        if 'NDVI_from_S2_NDWI_ratio' in target_features and 'NDVI_from_S2_NDWI_ratio' not in out.columns:
            out['NDVI_from_S2_NDWI_ratio'] = ndvi_fs / (np.abs(ndwi) + 1e-10)
        if 'NDVI_from_S2_NDWI_diff' in target_features and 'NDVI_from_S2_NDWI_diff' not in out.columns:
            out['NDVI_from_S2_NDWI_diff'] = ndvi_fs - ndwi
        if 'NDVI_from_S2_NDWI_from_S2_diff' in target_features and 'NDVI_from_S2_NDWI_from_S2_diff' not in out.columns and ndwi_fs is not None:
            out['NDVI_from_S2_NDWI_from_S2_diff'] = ndvi_fs - ndwi_fs
    if ndvi is not None and ndwi_fs is not None:
        if 'NDVI_NDWI_from_S2_ratio' in target_features and 'NDVI_NDWI_from_S2_ratio' not in out.columns:
            out['NDVI_NDWI_from_S2_ratio'] = ndvi / (np.abs(ndwi_fs) + 1e-10)
        if 'NDVI_NDWI_from_S2_diff' in target_features and 'NDVI_NDWI_from_S2_diff' not in out.columns:
            out['NDVI_NDWI_from_S2_diff'] = ndvi - ndwi_fs
    if ndvi is not None and ndwi is not None:
        if 'NDVI_NDWI_diff' in target_features and 'NDVI_NDWI_diff' not in out.columns:
            out['NDVI_NDWI_diff'] = ndvi - ndwi

    return out


# Satellite enrichment for uploaded points (approximate join)
@st.cache_data(show_spinner=False)
def _load_aggregated_s1() -> pd.DataFrame:
    """Load pre-aggregated Sentinel-1 if cached, else aggregate from Sentinel1.csv.

    Returns columns: translated_lat, translated_lon, S1_VH, S1_VV
    """
    try:
        cached = sorted(CACHE_DIR.glob('s1_agg_*.parquet'))
        if cached:
            return pd.read_parquet(cached[-1])
    except Exception:
        pass

    path = _find_file('Sentinel1.csv')
    if path is None:
        return pd.DataFrame()
    try:
        header = list(pd.read_csv(path, nrows=0).columns)
        usecols = [c for c in ['translated_lat','translated_lon','VH','VV'] if c in header]
        s1 = pd.read_csv(path, usecols=usecols)
        if s1.empty:
            return pd.DataFrame()
        grp = s1.groupby(['translated_lat','translated_lon'])[[c for c in ['VH','VV'] if c in s1.columns]].mean().reset_index()
        grp = grp.rename(columns={'VH':'S1_VH','VV':'S1_VV'})
        try:
            outp = CACHE_DIR / f"s1_agg_{abs(hash(tuple(grp.columns))):x}.parquet"
            grp.to_parquet(outp, index=False)
        except Exception:
            pass
        return grp
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_aggregated_s2() -> pd.DataFrame:
    """Load pre-aggregated Sentinel-2 if cached, else aggregate from Sentinel2.csv.

    Returns columns: translated_lat, translated_lon, S2_B*, S2_cloud_pct, S2_solar_azimuth, S2_solar_zenith
    """
    try:
        cached = sorted(CACHE_DIR.glob('s2_agg_*.parquet'))
        if cached:
            return pd.read_parquet(cached[-1])
    except Exception:
        pass

    path = _find_file('Sentinel2.csv')
    if path is None:
        return pd.DataFrame()
    try:
        header = list(pd.read_csv(path, nrows=0).columns)
        needed = ['translated_lat','translated_lon','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','cloud_pct','solar_azimuth','solar_zenith']
        usecols = [c for c in needed if c in header]
        s2 = pd.read_csv(path, usecols=usecols)
        if s2.empty:
            return pd.DataFrame()
        num_cols = [c for c in s2.columns if c not in ['translated_lat','translated_lon']]
        grp = s2.groupby(['translated_lat','translated_lon'])[num_cols].mean().reset_index()
        rename_map: Dict[str, str] = {}
        for c in num_cols:
            if c.startswith('B'):
                rename_map[c] = f'S2_{c}'
            elif c in {'cloud_pct','solar_azimuth','solar_zenith'}:
                rename_map[c] = f'S2_{c}'
        grp = grp.rename(columns=rename_map)
        try:
            outp = CACHE_DIR / f"s2_agg_{abs(hash(tuple(grp.columns))):x}.parquet"
            grp.to_parquet(outp, index=False)
        except Exception:
            pass
        return grp
    except Exception:
        return pd.DataFrame()


def _approximate_join(points: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Approximate-join agg features to points by rounding lat/lon at multiple precisions.

    Fills missing values only; preserves existing columns.
    """
    if points is None or points.empty or agg is None or agg.empty:
        return points
    base = points.copy()
    for prec in (4, 3, 2):
        left = base.copy()
        right = agg.copy()
        left['_lat_r'] = left['translated_lat'].round(prec)
        left['_lon_r'] = left['translated_lon'].round(prec)
        right['_lat_r'] = right['translated_lat'].round(prec)
        right['_lon_r'] = right['translated_lon'].round(prec)
        cols_add = [c for c in right.columns if c not in ['translated_lat','translated_lon','_lat_r','_lon_r']]
        merged = left.merge(right[['_lat_r','_lon_r'] + cols_add], on=['_lat_r','_lon_r'], how='left')
        for c in cols_add:
            if c in merged.columns:
                if c in base.columns:
                    base[c] = base[c].fillna(merged[c])
                else:
                    base[c] = merged[c]
        base = base.drop(columns=['_lat_r','_lon_r'], errors='ignore')
        # If most rows have at least one of the added columns populated, we can stop
        if cols_add:
            filled = merged[cols_add].notna().any(axis=1).mean()
            if filled > 0.9:
                break
    # Ensure unique columns
    if base.columns.duplicated().any():
        base = base.loc[:, ~base.columns.duplicated()]
    return base


def aggregate_s1(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Sentinel-1 dataframe by translated_lat/lon to S1_VH/S1_VV means."""
    if df is None or df.empty:
        return pd.DataFrame()
    header = list(df.columns)
    need = [c for c in ['translated_lat','translated_lon'] if c in header]
    if len(need) < 2:
        return pd.DataFrame()
    cols = [c for c in ['VH','VV'] if c in header]
    if not cols:
        return pd.DataFrame()
    grp = df.groupby(['translated_lat','translated_lon'])[cols].mean().reset_index()
    rename = {c: f'S1_{c}' for c in cols}
    return grp.rename(columns=rename)


def aggregate_s2(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Sentinel-2 dataframe by translated_lat/lon to S2_* means."""
    if df is None or df.empty:
        return pd.DataFrame()
    header = list(df.columns)
    need = [c for c in ['translated_lat','translated_lon'] if c in header]
    if len(need) < 2:
        return pd.DataFrame()
    num_cols = [
        c for c in header
        if c not in ['translated_lat','translated_lon']
    ]
    if not num_cols:
        return pd.DataFrame()
    grp = df.groupby(['translated_lat','translated_lon'])[num_cols].mean().reset_index()
    rename_map: Dict[str, str] = {}
    for c in num_cols:
        if c.startswith('B'):
            rename_map[c] = f'S2_{c}'
        elif c in {'cloud_pct','solar_azimuth','solar_zenith'}:
            rename_map[c] = f'S2_{c}'
    return grp.rename(columns=rename_map)


def enrich_with_satellite_features(points: pd.DataFrame, s1_agg: Optional[pd.DataFrame] = None, s2_agg: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Enrich uploaded points (with translated_lat/lon) using aggregated S1/S2 features.

    This mirrors the notebook's preprocessing where lat/lon points are matched to
    Sentinel-1/2 aggregates before feature engineering.
    """
    if points is None or points.empty:
        return points
    if not {'translated_lat','translated_lon'}.issubset(points.columns):
        return points
    s1 = s1_agg if s1_agg is not None else _load_aggregated_s1()
    s2 = s2_agg if s2_agg is not None else _load_aggregated_s2()
    enriched = points.copy()
    # Prefer nearest-neighbor join (BallTree haversine) to match notebook behavior
    def _nn_join(enriched_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
        try:
            from sklearn.neighbors import BallTree  # type: ignore
            if agg_df.empty:
                return enriched_df
            # Build BallTree on radians
            agg_rad = np.radians(agg_df[['translated_lat','translated_lon']].to_numpy(dtype=float))
            tree = BallTree(agg_rad, metric='haversine')
            pts_rad = np.radians(enriched_df[['translated_lat','translated_lon']].to_numpy(dtype=float))
            # Query nearest neighbor
            dist_rad, idx = tree.query(pts_rad, k=1)
            dist_deg = np.degrees(dist_rad.ravel())
            # Progressive thresholds like notebook (degrees)
            thresholds = [0.005, 0.01, 0.02, 0.05, 0.1]
            cols_add = [c for c in agg_df.columns if c not in ['translated_lat','translated_lon']]
            out_df = enriched_df.copy()
            matched_any = np.zeros(len(out_df), dtype=bool)
            for thr in thresholds:
                mask = (dist_deg <= thr) & (~matched_any)
                if not mask.any():
                    continue
                src = agg_df.iloc[idx.ravel()[mask]][cols_add].reset_index(drop=True)
                # Assign by position where mask true
                for c in cols_add:
                    if c not in out_df.columns:
                        out_df[c] = np.nan
                    out_df.loc[np.where(mask)[0], c] = src[c].values
                matched_any |= mask
                # Break early if high coverage
                if matched_any.mean() > 0.9:
                    break
            return out_df
        except Exception:
            # Fallback to approximate rounding join
            return _approximate_join(enriched_df, agg_df)

    if not s1.empty:
        enriched = _nn_join(enriched, s1)
    if not s2.empty:
        enriched = _nn_join(enriched, s2)
    if enriched.columns.duplicated().any():
        enriched = enriched.loc[:, ~enriched.columns.duplicated()]
    return enriched
