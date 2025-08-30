import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import st_folium

# Initialize session state (must be at the top)
if "ranked_locations" not in st.session_state:
    st.session_state["ranked_locations"] = None
if "map" not in st.session_state:
    st.session_state["map"] = None

# Debug: Confirm initialization
st.write("Session state initialized:", "ranked_locations" in st.session_state, "map" in st.session_state)

# App title and description
st.title("OPPLA \n The app for your optimal places!")
st.markdown("""
Select options to find optimal locations for opening a new business in your city.
Customize the business type, city, and target segment to tune the analysis.
""")

# Sidebar for user inputs
st.sidebar.header("Analysis Parameters")

# Business Type
business_types = ["Any", "cafe", "restaurant", "shop", "atm", "bar", "casino", "cinema", "college", "fast food", "gambling", "music venue", "nightclub", "pub", "theatre", "university"]
business_type = st.sidebar.selectbox("Business Type", business_types, help="Filter POIs by business type")

# City (Madrid for MVP, extensible for others)
cities = ["Madrid"]
city = st.sidebar.selectbox("City", cities, help="Select the city for analysis")

# Segment: Price and Audience
price_segment = st.sidebar.radio("Price Segment", ["Low-Cost", "Luxury"], help="Affects real estate scoring")
audience_segment = st.sidebar.radio("Target Audience", ["Anyone", "Young", "Aged"], help="Affects demographic scoring")

# Optional weight sliders
st.sidebar.subheader("Adjust Criteria Weights")
poi_weight = st.sidebar.slider("POI Density Weight", 0.0, 1.0, 0.3)
demo_weight = st.sidebar.slider("Demographics Weight", 0.0, 1.0, 0.3)
metro_weight = st.sidebar.slider("Metro Accessibility Weight", 0.0, 1.0, 0.2)
price_weight = st.sidebar.slider("Real Estate Affordability Weight", 0.0, 1.0, 0.2)

# Normalize weights to sum to 1
total_weight = poi_weight + demo_weight + metro_weight + price_weight
if total_weight > 0:
    poi_weight /= total_weight
    demo_weight /= total_weight
    metro_weight /= total_weight
    price_weight /= total_weight

# Load GeoJSON files (cached)
@st.cache_data
def load_data():
    pois = gpd.read_file('pois.geojson').to_crs(epsg=4326)
    metro = gpd.read_file('metro_stations.geojson').to_crs(epsg=4326)
    demographics = gpd.read_file('demographics.geojson').to_crs(epsg=4326)
    real_estate = gpd.read_file('real_estate.geojson').to_crs(epsg=4326)
    return pois, metro, demographics, real_estate

pois, metro, demographics, real_estate = load_data()

# City-specific bounding box (Madrid for MVP)
city_bounds = {
    "Madrid": {"min_lon": -3.8, "max_lon": -3.5, "min_lat": 40.3, "max_lat": 40.5}
}

# Function to perform MCA (cached)
@st.cache_data
def run_mca(_pois, _metro, _demographics, _real_estate, business_type, price_segment, audience_segment, weights, _city):
    # Grid setup
    min_lon, max_lon = city_bounds[_city]["min_lon"], city_bounds[_city]["max_lon"]
    min_lat, max_lat = city_bounds[_city]["min_lat"], city_bounds[_city]["max_lat"]
    grid_size = 0.005  # Approx 500m

    grid_cells = []
    grid_centroids = []
    for lon in np.arange(min_lon, max_lon, grid_size):
        for lat in np.arange(min_lat, max_lat, grid_size):
            cell = box(lon, lat, lon + grid_size, lat + grid_size)
            centroid = Point(lon + grid_size / 2, lat + grid_size / 2)
            grid_cells.append(cell)
            grid_centroids.append({'geometry': centroid, 'lon': centroid.x, 'lat': centroid.y})

    grid_gdf = gpd.GeoDataFrame(grid_centroids, geometry='geometry', crs='EPSG:4326')
    grid_cells_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs='EPSG:4326')

    # Filter POIs by business type
    if business_type != "Any":
        poi_categories = {
            "cafe": ["cafe", "coffee"],
            "restaurant": ["restaurant", "food"],
            "shop": ["shop", "retail"],
            "atm": ["atm"],
            "bar": ["bar"],
            "casino": ["casino"],
            "cinema": ["cinema"],
            "college": ["college"],
            "fast food": ["fast_food"],
            "gambling": ["gambling"],
            "music venue": ["music_venue"],
            "nightclub": ["nightclub"],
            "pub": ["pub"],
            "theatre": ["theatre"],
            "university": ["university"]
        }
        # Use the business type directly if no additional categories are specified
        categories = poi_categories[business_type]
        if not categories:  # If the category list is empty, use the business type itself
            categories = [business_type]
        if 'type' in _pois.columns:
            _pois = _pois[_pois['type'].isin(categories)]
        else:
            st.warning(f"Column 'type' not found in POIs dataset. Using all POIs.")

    # Score each criterion
    scores = []
    for idx, centroid in grid_gdf.iterrows():
        cell = grid_cells_gdf.iloc[idx].geometry
        centroid_point = centroid.geometry

        # POI Density
        poi_count = len(_pois[_pois.geometry.within(cell)])

        # Demographics: Adjust based on audience segment
        demo_match = _demographics[_demographics.geometry.contains(centroid_point)]
        if not demo_match.empty:
            if audience_segment == "Young":
                demo_score = demo_match.get('young_population', demo_match['young_pop_dens']).iloc[0]
            elif audience_segment == "Aged":
                demo_score = demo_match.get('aged_population', demo_match['old_pop_dens']).iloc[0]
            else:
                demo_score = demo_match['total_pop_density'].iloc[0]
        else:
            demo_score = _demographics['total_pop_density'].mean()

        # Metro Accessibility
        min_distance = _metro.geometry.distance(centroid_point).min()
        metro_score = 1 / (1 + min_distance)

        # Real Estate: Adjust based on price segment
        price_match = _real_estate[_real_estate.geometry.contains(centroid_point)]
        price = price_match['sqm_rental'].iloc[0] if not price_match.empty else _real_estate['sqm_rental'].mean()
        price_score = 1 / price if price_segment == "Low-Cost" else price  # Luxury prefers higher prices

        scores.append({
            'lon': centroid.lon,
            'lat': centroid.lat,
            'poi_score': poi_count,
            'demo_score': demo_score,
            'metro_score': metro_score,
            'price_score': price_score
        })

    # Normalize scores
    scores_df = pd.DataFrame(scores)
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores_df[['poi_score', 'demo_score', 'metro_score', 'price_score']])
    scores_df[['poi_score', 'demo_score', 'metro_score', 'price_score']] = normalized_scores

    # Weighted sum
    scores_df['final_score'] = (
        weights['poi'] * scores_df['poi_score'] +
        weights['demo'] * scores_df['demo_score'] +
        weights['metro'] * scores_df['metro_score'] +
        weights['price'] * scores_df['price_score']
    )

    return scores_df.sort_values(by='final_score', ascending=False)

# Clear results button
if st.button("Clear Results"):
    st.session_state["ranked_locations"] = None
    st.session_state["map"] = None
    st.rerun()

# Run analysis
if st.button("Run Analysis"):
    weights = {'poi': poi_weight, 'demo': demo_weight, 'metro': metro_weight, 'price': price_weight}
    # Run MCA and store results
    with st.spinner("Running analysis..."):
        st.session_state["ranked_locations"] = run_mca(
            pois, metro, demographics, real_estate, business_type, price_segment, audience_segment, weights, city
        )

    # Create and store Folium map
    m = folium.Map(location=(40.40, -3.65), zoom_start=11, tiles="CartoDB positron")
    for _, row in st.session_state["ranked_locations"].iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=4,
            color='green' if row['final_score'] > st.session_state["ranked_locations"]['final_score'].quantile(0.9) else 'none',
            fill=True,
            fill_opacity=0.6,
            popup=f"Score: {row['final_score']:.2f}"
        ).add_to(m)
    st.session_state["map"] = m

# Display results if available
if st.session_state.get("ranked_locations") is not None:
    st.subheader("Top Locations")
    st.dataframe(st.session_state["ranked_locations"][['lon', 'lat', 'final_score']].head(10))

    # Display map
    if st.session_state.get("map") is not None:
        st_folium(st.session_state["map"], width=700, height=500, key="folium_map")

    # Download results
    st.download_button(
        label="Download Results",
        data=st.session_state["ranked_locations"].to_csv(index=False),
        file_name="app_ranked_locations.csv",
        mime="text/csv"
    )