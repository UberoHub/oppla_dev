import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import st_folium
import requests
import time
from functools import lru_cache
import json

# Initialize session state (must be at the top)
if "ranked_locations" not in st.session_state:
    st.session_state["ranked_locations"] = None
if "map" not in st.session_state:
    st.session_state["map"] = None
if "foursquare_calls" not in st.session_state:
    st.session_state["foursquare_calls"] = 0
if "foursquare_cache" not in st.session_state:
    st.session_state["foursquare_cache"] = {}

# Foursquare API Configuration
FOURSQUARE_API_KEY = st.secrets.get("FOURSQUARE_API_KEY", "YOUR_API_KEY_HERE")  # Add to secrets.toml
DAILY_API_LIMIT = 10000
MAX_CALLS_PER_ANALYSIS = 500  # Conservative limit per analysis run

# Foursquare category mapping
FOURSQUARE_CATEGORIES = {
    "cafe": "13032",      # Coffee Shop
    "restaurant": "13065", # Restaurant
    "shop": "17069",      # Shop & Service
    "bar": "13003",       # Bar
    "casino": "10032",    # Casino
    "cinema": "10017",    # Movie Theater
    "college": "12013",   # College & Education
    "fast food": "13145", # Fast Food Restaurant
    "nightclub": "10032", # Nightclub
    "pub": "13003",       # Bar/Pub
    "theatre": "10017",   # Arts & Entertainment
    "university": "12013", # College & Education
    "atm": "12020",       # Bank/ATM
    "music venue": "10017" # Arts & Entertainment
}

# Debug: Confirm initialization
st.write("Session state initialized:", "ranked_locations" in st.session_state, "map" in st.session_state)

# App title and description
st.title("OPPLA \n The app for your optimal places!")
st.markdown("""
Select options to find optimal locations for opening a new business in your city.
Customize the business type, city, and target segment to tune the analysis.
**✨ Now enhanced with Foursquare Places API for real-time business intelligence!**
""")

# Sidebar for user inputs
st.sidebar.header("Analysis Parameters")

# API Status
if FOURSQUARE_API_KEY and FOURSQUARE_API_KEY != "YOUR_API_KEY_HERE":
    st.sidebar.success("🔗 Foursquare API Connected")
    st.sidebar.info(f"API calls used today: {st.session_state['foursquare_calls']}")
else:
    st.sidebar.error("⚠️ Foursquare API Key needed")
    st.sidebar.info("Add FOURSQUARE_API_KEY to secrets.toml")

# Business Type
business_types = ["Any", "cafe", "restaurant", "shop", "atm", "bar", "casino", "cinema", "college", "fast food", "gambling", "music venue", "nightclub", "pub", "theatre", "university"]
business_type = st.sidebar.selectbox("Business Type", business_types, help="Filter POIs by business type")

# City (Madrid for MVP, extensible for others)
cities = ["Madrid"]
city = st.sidebar.selectbox("City", cities, help="Select the city for analysis")

# Segment: Price and Audience
price_segment = st.sidebar.radio("Price Segment", ["Low-Cost", "Luxury"], help="Affects real estate scoring")
audience_segment = st.sidebar.radio("Target Audience", ["Anyone", "Young", "Aged"], help="Affects demographic scoring")

# Enhanced options with Foursquare
st.sidebar.subheader("🔥 Foursquare Enhancement")
use_foursquare = st.sidebar.checkbox("Enable Foursquare Analysis", value=True, 
                                   help="Use real-time business data from Foursquare")
foursquare_radius = st.sidebar.slider("Search Radius (meters)", 100, 1000, 300, 
                                    help="Radius for Foursquare venue search")

# Optional weight sliders
st.sidebar.subheader("Adjust Criteria Weights")
poi_weight = st.sidebar.slider("POI Density Weight", 0.0, 1.0, 0.25)
demo_weight = st.sidebar.slider("Demographics Weight", 0.0, 1.0, 0.25)
metro_weight = st.sidebar.slider("Metro Accessibility Weight", 0.0, 1.0, 0.2)
price_weight = st.sidebar.slider("Real Estate Affordability Weight", 0.0, 1.0, 0.15)
foursquare_weight = st.sidebar.slider("Foursquare Intelligence Weight", 0.0, 1.0, 0.15)

# Normalize weights to sum to 1
total_weight = poi_weight + demo_weight + metro_weight + price_weight + foursquare_weight
if total_weight > 0:
    poi_weight /= total_weight
    demo_weight /= total_weight
    metro_weight /= total_weight
    price_weight /= total_weight
    foursquare_weight /= total_weight

# Foursquare API functions
@lru_cache(maxsize=1000)
def get_foursquare_venues_cached(lat_lng_str, radius, categories, limit):
    """Cached Foursquare API call to avoid duplicates"""
    lat, lng = map(float, lat_lng_str.split(','))
    
    if st.session_state["foursquare_calls"] >= DAILY_API_LIMIT:
        return {"results": []}
    
    url = "https://api.foursquare.com/v3/places/search"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {FOURSQUARE_API_KEY}"
    }
    
    params = {
        "ll": f"{lat},{lng}",
        "radius": radius,
        "limit": min(limit, 50),
        "fields": "name,categories,location,stats,popularity,price,rating"
    }
    
    if categories:
        params["categories"] = categories
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        st.session_state["foursquare_calls"] += 1
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"results": []}
    except:
        return {"results": []}

def calculate_foursquare_score(lat, lng, business_type, radius=300):
    """Calculate comprehensive Foursquare-based score"""
    
    # Use cache key
    cache_key = f"{lat:.4f},{lng:.4f},{business_type},{radius}"
    if cache_key in st.session_state["foursquare_cache"]:
        return st.session_state["foursquare_cache"][cache_key]
    
    # Get competitor category
    competitor_category = FOURSQUARE_CATEGORIES.get(business_type)
    
    # 1. Get all venues in area
    all_venues = get_foursquare_venues_cached(
        f"{lat},{lng}", radius, None, 50
    )
    
    # 2. Get competitor venues
    competitor_venues = get_foursquare_venues_cached(
        f"{lat},{lng}", radius//2, competitor_category, 20
    ) if competitor_category else {"results": []}
    
    venues = all_venues.get("results", [])
    competitors = competitor_venues.get("results", [])
    
    # Calculate metrics
    total_venues = len(venues)
    competitor_count = len(competitors)
    
    # Popularity score (average of all venues)
    popularity_scores = [v.get("popularity", 0) for v in venues if "popularity" in v]
    avg_popularity = np.mean(popularity_scores) if popularity_scores else 0
    
    # Price level analysis
    price_levels = [v.get("price", 2) for v in venues if "price" in v]
    avg_price = np.mean(price_levels) if price_levels else 2
    
    # Rating analysis
    ratings = [v.get("rating", 0) for v in venues if "rating" in v and v["rating"] > 0]
    avg_rating = np.mean(ratings) if ratings else 0
    
    # Category diversity (complementary businesses)
    categories = set()
    for venue in venues:
        for cat in venue.get("categories", []):
            categories.add(cat.get("name", ""))
    category_diversity = len(categories)
    
    # Combined score calculation
    # Higher venue density and diversity is good
    # Lower competitor density is good
    # Higher popularity and ratings are good
    venue_density_score = min(total_venues / 20, 1.0)  # Normalize to max 20 venues
    competitor_penalty = max(0, 1 - competitor_count / 10)  # Penalize high competition
    popularity_score = min(avg_popularity / 100, 1.0)  # Normalize popularity
    quality_score = min(avg_rating / 10, 1.0)  # Normalize rating
    diversity_score = min(category_diversity / 15, 1.0)  # Normalize diversity
    
    final_score = (
        venue_density_score * 0.3 +
        competitor_penalty * 0.2 +
        popularity_score * 0.2 +
        quality_score * 0.15 +
        diversity_score * 0.15
    )
    
    # Cache result
    result = {
        'score': final_score,
        'total_venues': total_venues,
        'competitors': competitor_count,
        'avg_popularity': avg_popularity,
        'avg_rating': avg_rating,
        'category_diversity': category_diversity
    }
    
    st.session_state["foursquare_cache"][cache_key] = result
    return result

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

# Function to perform Enhanced MCA (cached)
@st.cache_data
def run_enhanced_mca(_pois, _metro, _demographics, _real_estate, business_type, price_segment, 
                    audience_segment, weights, _city, use_foursquare, foursquare_radius):
    # Grid setup
    min_lon, max_lon = city_bounds[_city]["min_lon"], city_bounds[_city]["max_lon"]
    min_lat, max_lat = city_bounds[_city]["min_lat"], city_bounds[_city]["max_lat"]
    grid_size = 0.01  # Larger grid to reduce API calls (approx 1km)

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

    # Limit grid size for API calls
    if len(grid_gdf) > MAX_CALLS_PER_ANALYSIS and use_foursquare:
        # Sample grid points to stay within API limits
        sample_size = min(MAX_CALLS_PER_ANALYSIS, len(grid_gdf))
        sample_indices = np.random.choice(len(grid_gdf), sample_size, replace=False)
        grid_gdf = grid_gdf.iloc[sample_indices].reset_index(drop=True)
        grid_cells_gdf = grid_cells_gdf.iloc[sample_indices].reset_index(drop=True)

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
        categories = poi_categories.get(business_type, [business_type])
        if 'type' in _pois.columns:
            _pois = _pois[_pois['type'].isin(categories)]
        else:
            st.warning(f"Column 'type' not found in POIs dataset. Using all POIs.")

    # Score each criterion
    scores = []
    progress_bar = st.progress(0)
    
    for idx, centroid in grid_gdf.iterrows():
        progress_bar.progress((idx + 1) / len(grid_gdf))
        
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
        price_score = 1 / price if price_segment == "Low-Cost" else price

        # Foursquare Score
        foursquare_data = {'score': 0, 'total_venues': 0, 'competitors': 0, 
                          'avg_popularity': 0, 'avg_rating': 0, 'category_diversity': 0}
        if use_foursquare and FOURSQUARE_API_KEY != "YOUR_API_KEY_HERE":
            foursquare_data = calculate_foursquare_score(
                centroid.lat, centroid.lon, business_type, foursquare_radius
            )
            # Small delay to respect rate limits
            time.sleep(0.1)

        scores.append({
            'lon': centroid.lon,
            'lat': centroid.lat,
            'poi_score': poi_count,
            'demo_score': demo_score,
            'metro_score': metro_score,
            'price_score': price_score,
            'foursquare_score': foursquare_data['score'],
            'fs_venues': foursquare_data['total_venues'],
            'fs_competitors': foursquare_data['competitors'],
            'fs_popularity': foursquare_data['avg_popularity'],
            'fs_rating': foursquare_data['avg_rating'],
            'fs_diversity': foursquare_data['category_diversity']
        })

    progress_bar.progress(1.0)
    progress_bar.empty()

    # Normalize scores
    scores_df = pd.DataFrame(scores)
    scaler = MinMaxScaler()
    score_columns = ['poi_score', 'demo_score', 'metro_score', 'price_score', 'foursquare_score']
    normalized_scores = scaler.fit_transform(scores_df[score_columns])
    scores_df[score_columns] = normalized_scores

    # Weighted sum
    scores_df['final_score'] = (
        weights['poi'] * scores_df['poi_score'] +
        weights['demo'] * scores_df['demo_score'] +
        weights['metro'] * scores_df['metro_score'] +
        weights['price'] * scores_df['price_score'] +
        weights['foursquare'] * scores_df['foursquare_score']
    )

    return scores_df.sort_values(by='final_score', ascending=False)

# Clear results button
if st.button("Clear Results"):
    st.session_state["ranked_locations"] = None
    st.session_state["map"] = None
    st.rerun()

# Run analysis
if st.button("🚀 Run Enhanced Analysis"):
    if use_foursquare and (not FOURSQUARE_API_KEY or FOURSQUARE_API_KEY == "YOUR_API_KEY_HERE"):
        st.error("Please add your Foursquare API key to secrets.toml to use Foursquare features")
    else:
        weights = {
            'poi': poi_weight, 
            'demo': demo_weight, 
            'metro': metro_weight, 
            'price': price_weight,
            'foursquare': foursquare_weight
        }
        
        # Run Enhanced MCA and store results
        with st.spinner("Running enhanced analysis with Foursquare data..."):
            st.session_state["ranked_locations"] = run_enhanced_mca(
                pois, metro, demographics, real_estate, business_type, 
                price_segment, audience_segment, weights, city, 
                use_foursquare, foursquare_radius
            )

        # Create enhanced Folium map
        m = folium.Map(location=(40.40, -3.65), zoom_start=11, tiles="CartoDB positron")
        
        top_locations = st.session_state["ranked_locations"].head(20)  # Show top 20
        
        for _, row in top_locations.iterrows():
            # Color coding based on score percentile
            if row['final_score'] > st.session_state["ranked_locations"]['final_score'].quantile(0.95):
                color = 'darkgreen'
                radius = 8
            elif row['final_score'] > st.session_state["ranked_locations"]['final_score'].quantile(0.80):
                color = 'green'
                radius = 6
            else:
                color = 'lightgreen'
                radius = 4
                
            # Enhanced popup with Foursquare data
            popup_text = f"""
            <b>Overall Score: {row['final_score']:.3f}</b><br>
            📍 Location: {row['lat']:.4f}, {row['lon']:.4f}<br>
            🏢 Nearby Venues: {row['fs_venues']}<br>
            🏪 Competitors: {row['fs_competitors']}<br>
            ⭐ Avg Rating: {row['fs_rating']:.1f}<br>
            📊 Popularity: {row['fs_popularity']:.1f}<br>
            🎯 Diversity: {row['fs_diversity']} categories
            """
            
            folium.CircleMarker(
                location=(row['lat'], row['lon']),
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
            
        st.session_state["map"] = m

# Display results if available
if st.session_state.get("ranked_locations") is not None:
    st.subheader("🎯 Top Optimal Locations")
    
    # Enhanced results display
    top_results = st.session_state["ranked_locations"].head(10)
    
    # Create display columns
    display_columns = ['lon', 'lat', 'final_score', 'fs_venues', 'fs_competitors', 
                      'fs_popularity', 'fs_rating', 'fs_diversity']
    column_names = ['Longitude', 'Latitude', 'Final Score', 'Nearby Venues', 
                   'Competitors', 'Popularity', 'Avg Rating', 'Category Diversity']
    
    display_df = top_results[display_columns].copy()
    display_df.columns = column_names
    display_df = display_df.round(3)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Key insights
    if use_foursquare:
        st.subheader("📊 Foursquare Insights")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_venues = top_results['fs_venues'].mean()
            st.metric("Avg Nearby Venues", f"{avg_venues:.1f}")
        
        with col2:
            avg_competitors = top_results['fs_competitors'].mean()
            st.metric("Avg Competitors", f"{avg_competitors:.1f}")
            
        with col3:
            avg_rating = top_results['fs_rating'].mean()
            st.metric("Area Avg Rating", f"{avg_rating:.1f}")
            
        with col4:
            total_api_calls = st.session_state["foursquare_calls"]
            st.metric("API Calls Used", f"{total_api_calls}")

    # Display map
    if st.session_state.get("map") is not None:
        st.subheader("📍 Location Map")
        st_folium(st.session_state["map"], width=700, height=500, key="folium_map")

    # Enhanced download with Foursquare data
    st.download_button(
        label="📥 Download Enhanced Results",
        data=st.session_state["ranked_locations"].to_csv(index=False),
        file_name=f"oppla_enhanced_results_{business_type}_{city}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**OPPLA** - Powered by spatial analysis and Foursquare Places API 🚀")