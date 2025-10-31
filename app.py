"""
============================================================================
INTERACTIVE SPATIAL DASHBOARD FOR INFECTIOUS DISEASES IN INDONESIA
STREAMLIT VERSION - OPTIMIZED
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error
import warnings
from datetime import datetime
import base64
from io import BytesIO
import json

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Indonesia Disease Risk Explorer",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .risk-card {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_disease_data():
    """Load and preprocess disease data"""
    try:
        # Path relatif ke folder data
        data = pd.read_csv("DATA SPASIAL EPIDEM - VARIABEL DEPENDEN (1).csv")
        # st.write('pertama banget')
        data.drop(columns='Rabies', inplace=True)
        
        # Rename columns
        data.columns = [
            "province", "malaria", "dbd", "filariasis",
            "sanitation", "pop_density", "hospitals", "poor_pct", "population"
        ]
        # st.write('ini bisa')
        # Convert to numeric
        numeric_cols = ["malaria", "dbd", "filariasis", "sanitation", 
                       "pop_density", "hospitals", "poor_pct", "population"]
        
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '.'), errors='coerce')
            data[col].fillna(data[col].median(), inplace=True)
        
        # Calculate prevalence rates
        data['population_actual'] = data['population'] * 1000
        
        for disease in ['malaria', 'dbd', 'filariasis']:
            data[f'{disease}_prev'] = (data[disease] / data['population_actual']) * 100000
            data[f'{disease}_prev'].replace([np.inf, -np.inf], 0, inplace=True)
            data[f'{disease}_prev'].fillna(0, inplace=True)
            
            batas25 = data[f'{disease}_prev'].quantile(0.25)
            batas50 = data[f'{disease}_prev'].quantile(0.50)
            batas75 = data[f'{disease}_prev'].quantile(0.75)

            # Pre-calculate risk levels
            data[f'{disease}_risk'] = data[f'{disease}_prev'].apply(
                lambda x: get_risk_level(x, disease, batas25, batas50, batas75)
            )
        return data
        # st.write(data)
    
    except Exception as e:
        st.error(f"Error loading disease data: {e}")
        return None


@st.cache_data
def load_spatial_data():
    # """Load spatial boundary data"""
    # try:
        # Path relatif ke folder maps
        # gdf = gpd.read_file('maps/Batas_Provinsi_Indonesia.geojson')
        gdf = gpd.read_file('gadm41_IDN_1.json')
        # gdf = gpd.read_file('gadm41_IDN_1_DENGAN_PAPUA.json')
        # Validate and fix geometries
        # gdf['geometry'] = gdf['geometry'].make_valid()
        
        #     if isinstance(geom, Polygon):
        #         return MultiPolygon([geom])
        #     return geom
        
        # gdf['geometry'] = gdf['geometry'].apply(to_multipoly
        # Cast to MULTIPOLYGON
        # from shapely.geometry import MultiPolygon, Polygon
        
        # def to_multipolygon(geom):gon)
        
        return gdf
    
    # except Exception as e:
    #     st.error(f"Error loading spatial data: {e}")
    #     return None

@st.cache_data
def merge_data(disease_data, _spatial_data): # <- Tambahkan underscore di depan 'spatial_data'
    """Merge disease and spatial data"""
    try:
        # PENTING: Di dalam fungsi, Anda harus menggunakan nama argumen yang baru
        merged = _spatial_data.merge( 
            disease_data, 
            left_on='NAME_1', 
            right_on='province', 
            how='left'
        )
        merged = merged.dropna(subset=['dbd'])
        return merged
    
    except Exception as e:
        st.error(f"Error merging data: {e}")
        return None

# ============================================================================
# SPATIAL ANALYSIS FUNCTIONS
# ============================================================================

@st.cache_data
def create_weights(_gdf, k=3):
    """Create spatial weights matrix"""
    try:
        w = KNN.from_dataframe(_gdf, k=k)
        w.transform = 'r'
        return w
    except Exception as e:
        st.error(f"Error creating weights: {e}")
        return None

def calculate_morans_i(gdf, disease_var, w):
    """Calculate Global Moran's I"""
    try:
        prevalence_var = f"{disease_var}_prev"
        y = gdf[prevalence_var].values
        
        # Remove NaN values
        valid_idx = ~np.isnan(y)
        gdf_valid = gdf.loc[valid_idx].reset_index(drop=True) # baru
        y = y[valid_idx]

        # Rebuild weights for the filtered gdf
        from libpysal.weights import Queen
        w_valid = Queen.from_dataframe(gdf_valid)
        w_valid.transform = 'r'
        
        if len(y) < 3:
            return None
        
        moran = Moran(y, w_valid)
        
        return {
            'Statistic': moran.I,
            'P_value': moran.p_sim,
            'Expected': moran.EI,
            'Z_score': moran.z_sim
        }
    
    except Exception as e:
        st.error(f"Error in Moran's I: {e}")
        return None

def calculate_lisa(gdf, disease_var, w):
    """Calculate Local Moran's I (LISA)"""
    try:
        prevalence_var = f"{disease_var}_prev"
        y = gdf[prevalence_var].values
        
        # Remove NaN
        valid_idx = ~np.isnan(y)
        y = y[valid_idx]
        gdf_valid = gdf[valid_idx].copy()

        # Rebuild weights for the filtered gdf
        from libpysal.weights import Queen
        w_valid = Queen.from_dataframe(gdf_valid)
        w_valid.transform = 'r'
        
        if len(y) < 3:
            return None
        
        lisa = Moran_Local(y, w_valid)
        
        gdf_valid['Ii'] = lisa.Is
        gdf_valid['p_value'] = lisa.p_sim
        
        # Standardize
        y_std = (y - y.mean()) / y.std()
        from libpysal.weights import lag_spatial
        spatial_lag = lag_spatial(w_valid, y_std)
        
        # Classify clusters
        gdf_valid['Cluster'] = 'Not Significant'
        sig_mask = gdf_valid['p_value'] < 0.05
        
        gdf_valid.loc[sig_mask & (y_std > 0) & (spatial_lag > 0), 'Cluster'] = 'HH (Hotspot)'
        gdf_valid.loc[sig_mask & (y_std < 0) & (spatial_lag < 0), 'Cluster'] = 'LL (Coldspot)'
        gdf_valid.loc[sig_mask & (y_std > 0) & (spatial_lag < 0), 'Cluster'] = 'HL (Outlier)'
        gdf_valid.loc[sig_mask & (y_std < 0) & (spatial_lag > 0), 'Cluster'] = 'LH (Outlier)'
        
        return gdf_valid
    
    except Exception as e:
        st.error(f"Error in LISA: {e}")
        return None

def fit_spatial_models(gdf, disease_var, x_vars, w):
    """Fit spatial econometric models"""
    try:
        prevalence_var = f"{disease_var}_prev"
        
        # Prepare data
        data = gdf[[prevalence_var] + x_vars].dropna()
        y = data[prevalence_var].values.reshape(-1, 1)
        X = data[x_vars].values
        
        if len(y) < 5:
            return None
        
        # Rebuild spatial weights for valid subset
        from libpysal.weights import Queen
        gdf_valid = gdf.loc[data.index]
        w_valid = Queen.from_dataframe(gdf_valid)
        w_valid.transform = 'r'
        
        models = {}
        
        # OLS
        try:
            models['ols'] = OLS(y, X, name_x=x_vars, name_y=prevalence_var)
        except Exception as e:
            st.warning(f"OLS failed: {e}")
            models['ols'] = None
        
        # Spatial Lag (SAR)
        try:
            models['lag'] = ML_Lag(y, X, w_valid, name_x=x_vars, name_y=prevalence_var)
        except Exception as e:
            st.warning(f"SAR failed: {e}")
            models['lag'] = None
        
        # Spatial Error (SEM)
        try:
            models['error'] = ML_Error(y, X, w_valid, name_x=x_vars, name_y=prevalence_var)
        except Exception as e:
            st.warning(f"SEM failed: {e}")
            models['error'] = None
        
        return models
    
    except Exception as e:
        st.error(f"Error fitting models: {e}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_level(value, disease_var, q1, q2, q3):
    """Determine risk level based on prevalence"""
    thresholds = {
        'dbd': {'low': q1, 'medium': q2, 'high': q3},
        'malaria': {'low': q1, 'medium': q2, 'high': q3},
        'filariasis': {'low': q1, 'medium': q2, 'high': q3}
    }
    thresh = thresholds.get(disease_var, thresholds[disease_var])
    
    if pd.isna(value):
        return "No Data"
    elif value < thresh['low']:
        return "Low Risk"
    elif value < thresh['medium']:
        return "Medium Risk"
    elif value < thresh['high']:
        return "High Risk"
    else:
        return "Very High Risk"

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        "Low Risk": "#FFEB3B",
        "Medium Risk": "#FF9800",
        "High Risk": "#F0592B",
        "Very High Risk": "#020000",
        "No Data": "#9E9E9E"
    }
    return colors.get(risk_level, "#0A0000")

def create_download_link(fig, filename):
    """Create download link for matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download Map</a>'

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🦟 Indonesia Disease Risk Explorer")
    
    # Sidebar
    st.sidebar.header("⚙ Settings")
    
    # Disease selection
    disease_names = {
        'dbd': '🦟 Dengue Fever (DBD)',
        'malaria': '🦟 Malaria',
        'filariasis': '🦟 Filariasis (Elephantiasis)'
    }
    
    selected_disease = st.sidebar.selectbox(
        "Choose Disease:",
        options=list(disease_names.keys()),
        format_func=lambda x: disease_names[x]
    )
    
    # Risk factors selection
    risk_factors = {
        'sanitation': 'Clean Water Access',
        'pop_density': 'Population Density',
        'hospitals': 'Hospital Availability',
        'poor_pct': 'Poverty Rate'
    }
    
    selected_factors = st.sidebar.multiselect(
        "Risk Factors:",
        options=list(risk_factors.keys()),
        default=list(risk_factors.keys()),
        format_func=lambda x: risk_factors[x]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        disease_data = load_disease_data()  
        spatial_data = load_spatial_data()
        
        if disease_data is None or spatial_data is None:
            st.error("Failed to load data. Please check file paths.")
            return
        
        spatial_data.iloc[6, 3] = 'DKIJakarta'
        spatial_data.iloc[33, 3] = 'DaerahIstimewaYogyakarta'
        spatial_data.iloc[2, 3] = 'KepulauanBangkaBelitung'
        spatial_data.iloc[34:38, 3] = spatial_data.iloc[34:38, 11]
        spatial_data['NAME_1'] = spatial_data['NAME_1'].astype(str).str.replace(' ', '', regex=False)
        disease_data['province'] = disease_data['province'].astype(str).str.replace(' ', '', regex=False)

        merged_data = merge_data(disease_data, spatial_data)
        merged_data.reset_index(drop=True, inplace=True)
        
        if merged_data is None:
            st.error("Failed to merge data.")
            return
        
        weights = create_weights(merged_data)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺 Interactive Map",
        "🔍 Province Explorer",
        "📊 Risk Patterns",
        "📈 Statistical Model",
        "📋 Data Summary",
        "ℹ How to Use"
    ])
    
    # TAB 1: INTERACTIVE MAP
    with tab1:
        # ===== METRICS =====
        col1, col2, col3, col4 = st.columns(4)
        total_cases = merged_data[selected_disease].sum()
        risk_var = f"{selected_disease}_risk"
        
        with col1:
            st.metric("Total Cases", f"{int(total_cases):,}")
        
        with col2:
            high_risk = (merged_data[risk_var].isin(['High Risk', 'Very High Risk'])).sum()
            st.metric("High Risk Provinces", high_risk)
        
        with col3:
            medium_risk = (merged_data[risk_var] == 'Medium Risk').sum()
            st.metric("Medium Risk Provinces", medium_risk)
        
        with col4:
            low_risk = (merged_data[risk_var] == 'Low Risk').sum()
            st.metric("Low Risk Provinces", low_risk)
        
        # ===== MAP SECTION =====
        if selected_disease == 'filariasis':
            st.subheader("Disease Prevalence Map")
        else:
            st.subheader('Disease Incidence Map')
        
        prevalence_var = f"{selected_disease}_prev"
        prevalence_risk = f"{selected_disease}_risk"
        
        # ⚠ Convert GeoDataFrame to geojson dict
        merged_data = merged_data.set_index("NAME_1")  # pakai nama provinsi sebagai key
        # geojson_data = merged_data._geo_interface_
        geojson_data = json.loads(merged_data.to_json())

        with st.container(border=True):
            # ===== Choropleth =====
            fig = px.choropleth(
                merged_data,
                geojson=geojson_data,
                locations=merged_data.index,
                color=prevalence_risk,
                hover_name=merged_data.index,
                hover_data={
                    selected_disease: ':,.0f',
                    prevalence_var: ':.2f',
                    risk_var: True,
                    # merged_data.index: False
                },
                # color_continuous_scale=['#FFC107', '#FF9800', '#FF5722', "#170302"],
                color_discrete_map={
                    "Low Risk":  "#F6D745",
                    "Medium Risk": "#E55C30",
                    "High Risk": "#84206B",
                    "Very High Risk": "#140B34"
                },
                # labels={prevalence_var: 'Prevalence (per 100k)'}
            )
            # fig.update_geos(center={"lat": -2.5, "lon": 118})
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(height=700, margin={"r":0,"t":0,"l":0,"b":0})
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Tip: Interactive map with zoom/pan. Hover to see details. Yellow = Low risk, Orange-Red = High risk.")
    
    # TAB 2: PROVINCE EXPLORER
    with tab2:
        selected_province = st.selectbox(
            "Choose Province:",
            options=sorted(merged_data.index.unique())
        )
        
        province_data = merged_data[merged_data.index == selected_province].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Province Risk Profile")
            
            risk_level = province_data[risk_var]
            risk_color = get_risk_color(risk_level)
            
            st.markdown(f"""
            <div style="background-color: {risk_color}; padding: 20px; border-radius: 10px; color: white;">
                <h2>{selected_province}</h2>
                <h3>{risk_level}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.write(f"*Total Cases:* {int(province_data[selected_disease]):,}")
            st.write(f"*Prevalence:* {province_data[prevalence_var]:.2f} per 100,000")
            st.write(f"*Population:* {int(province_data['population']):,} thousand")
            st.write("")
            st.write(f"*Clean Water Access:* {province_data['sanitation']:.1f}%")
            st.write(f"*Population Density:* {province_data['pop_density']:.0f} people/km²")
            st.write(f"*Hospitals:* {int(province_data['hospitals'])}")
            st.write(f"*Poverty Rate:* {province_data['poor_pct']:.1f}%")
        
        with col2:
            st.subheader("Location Map")

            # Ambil tetangga dari provinsi yang dipilih
            selected_index = merged_data.index.get_loc(selected_province)
            tetangga_idx = weights.neighbors.get(selected_index, [])
            tetangga_names = [merged_data.index[n] for n in tetangga_idx]
            
            try:
                merged_data['map_category'] = 'Other Provinces'
                merged_data.loc[merged_data.index.isin(tetangga_names), 'map_category'] = 'Neighbor'
                merged_data.loc[[selected_province], 'map_category'] = 'Selected Province'

                is_neighbor = merged_data['map_category'] == 'Neighbor'
                is_very = merged_data[risk_var] == 'Very High Risk'
                is_high = merged_data[risk_var] == 'High Risk'
                is_medium = merged_data[risk_var] == 'Medium Risk'
                is_low = merged_data[risk_var] == 'Low Risk'

                merged_data.loc[is_neighbor & is_very, 'map_category'] = 'Neighbor - Very High Risk'
                merged_data.loc[is_neighbor & is_high, 'map_category'] = 'Neighbor - High Risk'
                merged_data.loc[is_neighbor & is_medium, 'map_category'] = 'Neighbor - Medium Risk'
                merged_data.loc[is_neighbor & is_low, 'map_category'] = 'Neighbor - Low Risk'
                
                color_map = {
                    'Other Provinces': 'lightgray',
                    'Selected Province': '#2196F3',
                    
                    'Neighbor - Very High Risk': '#140B34',
                    'Neighbor - High Risk': "#84206B",
                    'Neighbor - Medium Risk': "#E55C30",
                    'Neighbor - Low Risk': "#F6D746"
                }
                title_text = f"Location: {selected_province} and Neighbors"

            except Exception:
                merged_data['map_category'] = 'Other Provinces'
                merged_data.loc[[selected_province], 'map_category'] = 'Selected Province'
                
                color_map = {
                    'Other Provinces': 'lightgray',
                    'Selected Province': '#2196F3'
                }
                title_text = f"Location: {selected_province}"

            fig = px.choropleth(
                merged_data,
                geojson=geojson_data,
                locations=merged_data.index,
                color='map_category',
                color_discrete_map=color_map,
                hover_name=merged_data.index,
                hover_data={'map_category': False, 'Risk': merged_data[risk_var]}
            )

            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(
                showlegend=False,
                margin={"r":0,"t":0,"l":0,"b":0}
            )

            st.plotly_chart(fig, use_container_width=True)

            # ================================================================
    
    # TAB 3: RISK PATTERNS
    with tab3:
        st.subheader("Understanding Disease Prevalence Patterns")
        
        st.write("""
        This analysis uses prevalence rates (per 100,000 population) to check if provinces 
        with high disease prevalence tend to be near other provinces with high prevalence (clustering).
        """)
        
        if weights is not None:
            morans = calculate_morans_i(merged_data, selected_disease, weights)
            
            if morans:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Clustering Score", f"{morans['Statistic']:.4f}")
                
                with col2:
                    is_significant = "YES - Pattern is real" if morans['P_value'] < 0.05 else "NO - Could be random"
                    st.metric("Statistical Significance", is_significant)
                
                # Interpretation
                if morans['P_value'] < 0.05:
                    if morans['Statistic'] > 0:
                        st.error("""
                        🔥 *Disease Clustering Detected!*
                        
                        Provinces with high disease prevalence are located close to other provinces with high prevalence.
                        
                        *Action needed:* Focus on hotspot clusters and implement border control measures.
                        """)
                    else:
                        st.info("📍 *Scattered Pattern Detected* - High and low prevalence provinces are randomly distributed.")
                else:
                    st.success("✅ *No Clear Pattern* - Disease distribution appears random.")
                
                # LISA Map
                st.subheader("Prevalence Pattern Map (LISA Analysis)")
                
                lisa_result = calculate_lisa(merged_data, selected_disease, weights)
                
                if lisa_result is not None:
                    cluster_colors = {
                        'HH (Hotspot)': '#d73027',
                        'LL (Coldspot)': '#4575b4',
                        'HL (Outlier)': '#fee090',
                        'LH (Outlier)': '#91bfdb',
                        'Not Significant': '#f7f7f7'
                    }
                    with st.container(border=True):
                        fig, ax = plt.subplots(figsize=(14, 10))
                        
                        for cluster, color in cluster_colors.items():
                            cluster_data = lisa_result[lisa_result['Cluster'] == cluster]
                            if len(cluster_data) > 0:
                                cluster_data.plot(ax=ax, color=color, edgecolor='black', 
                                                linewidth=0.5, label=cluster)
                        
                        ax.set_title(f"Disease Prevalence Pattern - {selected_disease.upper()}", 
                                fontsize=16, fontweight='bold')
                        ax.axis('off')
                        ax.legend(loc='lower right')
                        
                        st.pyplot(fig)
                    
                    st.write("*What do the colors mean?*")
                    st.write("- *Red (Hotspot - HH):* High prevalence area surrounded by other high prevalence areas")
                    st.write("- *Blue (Coldspot - LL):* Low prevalence area surrounded by other low prevalence areas")
                    st.write("- *Yellow (Outlier - HL):* High prevalence area surrounded by low prevalence areas")
                    st.write("- *Light Blue (Outlier - LH):* Low prevalence area surrounded by high prevalence areas")
                    st.write("- *Gray (Not Significant):* No clear pattern detected")
    
    # TAB 4: STATISTICAL MODEL
    with tab4:
        st.subheader("What Factors Affect Disease Prevalence?")
        
        st.write("""
        This analysis compares different statistical models to understand how factors influence disease prevalence.
        
        - *OLS:* Standard regression
        - *Spatial Lag (SAR):* Considers neighboring provinces' disease prevalence
        - *Spatial Error (SEM):* Accounts for spatial patterns in errors
        """)
        
        if weights is not None and len(selected_factors) > 0:
            with st.spinner("Fitting spatial models..."):
                models = fit_spatial_models(merged_data, selected_disease, selected_factors, weights)
            
            if models:
                # Model comparison
                st.subheader("📊 Model Comparison")
                
                comparison_data = []
                
                for model_name, model in models.items():
                    if model is not None:
                        y = model.y
                        y_pred = model.predy
                        r2_manual = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
                        comparison_data.append({
                            'Model': model_name.upper(),
                            'AIC': model.aic if hasattr(model, 'aic') else np.nan,
                            'Log-Likelihood': model.logll if hasattr(model, 'logll') else np.nan,
                            'R²': model.r2 if hasattr(model, 'r2') else r2_manual
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Highlight best model (lowest AIC)
                    best_idx = comparison_df['AIC'].idxmin()
                    comparison_df['Status'] = ''
                    comparison_df.loc[best_idx, 'Status'] = '⭐ BEST'
                    
                    st.dataframe(comparison_df.style.highlight_max(subset=['R²'], color='lightgreen')
                                .highlight_min(subset=['AIC'], color='lightgreen'))
                    
                    st.success(f"🏆 *Recommended Model:* {comparison_df.loc[best_idx, 'Model']}")
                    
                    # Detailed coefficients
                    st.subheader("Factor Impact Analysis")
                    
                    selected_model_type = st.selectbox(
                        "Choose Model to View:",
                        options=list(models.keys()),
                        format_func=lambda x: x.upper()
                    )
                    
                    model = models[selected_model_type]
                    if model is not None:
                        try:
                            coef_df = pd.DataFrame({
                                'Factor': selected_factors,
                                'Coefficient': model.betas[1:-1].flatten(),
                                'Std Error': np.sqrt(np.diag(model.vm))[1:-1],
                            })
                            coef_df['T-value'] = coef_df['Coefficient'] / coef_df['Std Error']
                            coef_df['Significant'] = coef_df['T-value'].abs() > 1.96
                            st.dataframe(
                                coef_df.style.apply(
                                    lambda x: [
                                        'background-color: lightgreen' if x['Significant'] else ''
                                        ] * len(x), 
                                        axis=1
                                )
                            )
                        except:
                            coef_df = pd.DataFrame({
                                'Factor': selected_factors,
                                'Coefficient': model.betas[1:].flatten(),
                                'Std Error': np.sqrt(np.diag(model.vm))[1:],
                            })
                            coef_df['T-value'] = coef_df['Coefficient'] / coef_df['Std Error']
                            coef_df['Significant'] = coef_df['T-value'].abs() > 1.96
                            st.dataframe(
                                coef_df.style.apply(
                                    lambda x: [
                                        'background-color: lightgreen' if x['Significant'] else ''
                                        ] * len(x), 
                                        axis=1
                                )
                            )
  
                        
                        # Model quality
                        col1, col2, col3 = st.columns(3)
                        with col1: # Indentasi 1 (8 spasi)
                            try:
                                st.metric("R²", f"{model.r2:.4f}")
                            except:
                                y = model.y
                                y_pred = model.predy
                                r2_manual = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
                                # print("Manual R²:", r2_manual)
                                st.metric("R²", f"{r2_manual:.4f}")

                        with col2: # Indentasi 1 (8 spasi)
                            st.metric("AIC", f"{model.aic:.2f}")
                        with col3: # Indentasi 1 (8 spasi)
                            try:
                                quality = "Excellent" if model.r2 > 0.7 else \
                                    "Good" if model.r2 > 0.5 else \
                                    "Moderate" if model.r2 > 0.3 else "Poor"
                            except:
                                quality = "Excellent" if r2_manual > 0.7 else \
                                    "Good" if r2_manual > 0.5 else \
                                    "Moderate" if r2_manual > 0.3 else "Poor"
                            st.metric("Model Quality", quality)
            # app.py, baris 636
            if weights is not None and len(selected_factors) > 0: # <-- Baris 636
                with st.spinner("Fitting spatial models..."): # <-- INDENTASI DIMULAI DI SINI
                    models = fit_spatial_models(merged_data, selected_disease, selected_factors, weights)
                    if models:
                        # st.subheader("📊 Model Comparison")
                        comparison_data = []
                    else:
                        pass
            else: # <-- Baris 'else' ini HARUS SEJAJAR dengan 'if' di baris 636
                st.warning("Please select at least one risk factor to fit models.")
    
    # TAB 5: DATA SUMMARY
    with tab5:
        st.subheader("Province Statistics")
        
        # Summary statistics
        summary_vars = ['population', selected_disease, prevalence_var] + selected_factors
        summary_stats = merged_data[summary_vars].describe().T
        
        st.dataframe(summary_stats)
        
        # Correlation matrix
        st.subheader("Relationship Between Factors")
        
        corr_vars = [prevalence_var] + selected_factors
        corr_matrix = merged_data[corr_vars].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        with st.container(border=True):
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
        
        # Full dataset
        st.subheader("Full Dataset")
        
        display_cols = ['population', selected_disease, prevalence_var, risk_var] + selected_factors
        st.dataframe(merged_data[display_cols])
    
    # TAB 6: HOW TO USE
    with tab6:
        st.subheader("📖 Quick Start Guide")
        
        st.write("""
        ### 1. Interactive Map Tab
        - View disease risk levels across all Indonesian provinces
        - Yellow colors indicate low risk, orange to red indicate increasing risk levels
        - Hover over provinces to see detailed information
        
        ### 2. Province Explorer Tab
        - Select a province from the dropdown to see detailed information
        - View neighboring provinces colored by their risk levels
        - Compare prevalence with neighboring provinces
        
        ### 3. Risk Patterns Tab
        - Discover if disease cases are clustering in certain regions
        - Red areas show hotspots, blue areas show coldspots
        
        ### 4. Statistical Model Tab
        - Compare different spatial econometric models
        - View which factors significantly affect disease prevalence
        
        ### 5. Data Summary Tab
        - View comprehensive statistics for all provinces
        - Explore relationships between different factors
        
        ---
        
        ### 🎨 About Color Schemes
        
        The dashboard uses intuitive color coding:
        
        - *Interactive Map:* Yellow (low risk) → Orange → Red (high risk)
        - *Province Explorer:* Neighbors colored by risk level, selected province in contrasting color
        - *Risk Patterns:* Red (hotspots), Blue (coldspots), Yellow/Light Blue (outliers)
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if _name_ == "_main_":
    main()