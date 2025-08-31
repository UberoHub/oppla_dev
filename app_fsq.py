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

# Initialize session state
if "ranked_locations" not in st.session_state:
    st.session_state["ranked_locations"] = None
if "map" not in st.session_state:
    st.session_state["map"] = None
if "foursquare_calls" not in st.session_state:
    st.session_state["foursquare_calls"] = 0
if "foursquare_cache" not in st.session_state:
    st.session_state["foursquare_cache"] = {}

# Foursquare API Configuration
FOURSQUARE_API_KEY = st.secrets.get("FOURSQUARE_API_KEY", "YOUR_API_KEY_HERE")
DAILY_API_LIMIT = 1000
MAX_CALLS_PER_ANALYSIS = 100

# Foursquare Dining & Drinking Categories
DINING_DRINKING_CATEGORIES = [
    "Dining and Drinking > Bagel Shop",
    "Dining and Drinking > Bakery",
    "Dining and Drinking > Bar",
    "Dining and Drinking > Breakfast Spot",
    "Dining and Drinking > Brewery",
    "Dining and Drinking > Cafe, Coffee, and Tea House",
    "Dining and Drinking > Cafeteria",
    "Dining and Drinking > Cidery",
    "Dining and Drinking > Creperie",
    "Dining and Drinking > Dessert Shop",
    "Dining and Drinking > Distillery",
    "Dining and Drinking > Donut Shop",
    "Dining and Drinking > Food Court",
    "Dining and Drinking > Food Stand",
    "Dining and Drinking > Food Truck",
    "Dining and Drinking > Juice Bar",
    "Dining and Drinking > Meadery",
    "Dining and Drinking > Night Market",
    "Dining and Drinking > Restaurant",
    "Dining and Drinking > Smoothie Shop",
    "Dining and Drinking > Snack Place",
    "Dining and Drinking > Vineyard",
    "Dining and Drinking > Winery"
]

# Category groupings for business type selection
CATEGORY_GROUPS = {
    "Any Dining & Drinking": DINING_DRINKING_CATEGORIES,
    "Coffee & Cafe": [
        "Dining and Drinking > Cafe, Coffee, and Tea House",
        "Dining and Drinking > Donut Shop",
        "Dining and Drinking > Bakery",
        "Dining and Drinking > Breakfast Spot"
    ],
    "Restaurant": [
        "Dining and Drinking > Restaurant",
        "Dining and Drinking > Food Court",
        "Dining and Drinking > Cafeteria"
    ],
    "Bar & Nightlife": [
        "Dining and Drinking > Bar",
        "Dining and Drinking > Brewery",
        "Dining and Drinking > Distillery",
        "Dining and Drinking > Winery",
        "Dining and Drinking > Cidery",
        "Dining and Drinking > Vineyard",
        "Dining and Drinking > Meadery"
    ],
    "Quick Service": [
        "Dining and Drinking > Food Stand",
        "Dining and Drinking > Food Truck",
        "Dining and Drinking > Snack Place",
        "Dining and Drinking > Juice Bar",
        "Dining and Drinking > Smoothie Shop"
    ],
    "Dessert & Sweets": [
        "Dining and Drinking > Dessert Shop",
        "Dining and Drinking > Bakery",
        "Dining and Drinking > Creperie"
    ]
}

# App title and description
st.title("OPPLA \n The app for your optimal places!")
st.markdown("""
Find optimal locations for dining & drinking businesses using real-time Foursquare data.
Analyze competition, demographics, metro accessibility, and rental prices.
""")

# Sidebar for user inputs
st.sidebar.header("Analysis Parameters")

# API Status
if FOURSQUARE_API_KEY and FOURSQUARE_API_KEY != "YOUR_API_KEY_HERE":
    st.sidebar.success("🔗 Foursquare API Connected")
    st.sidebar.info(f"API calls used: {st.session_state['foursquare_calls']}")
else:
    st.sidebar.error("⚠️ Add FOURSQUARE_API_KEY to secrets.toml")

# Business Type Selection
business_types = list(CATEGORY_GROUPS.keys())
business_type = st.sidebar.selectbox(
    "Business Type", 
    business_types, 
    help="Select the type of dining/drinking business"
)

# City
cities = ["Madrid"]
city = st.sidebar.selectbox("City", cities)

# Segments
price_segment = st.sidebar.radio(
    "Price Segment", 
    ["Low-Cost", "Luxury"], 
    help="Affects real estate scoring preference"
)
audience_segment = st.sidebar.radio(
    "Target Audience", 
    ["Anyone", "Young", "Aged"], 
    help="Affects demographic scoring"
)

# Analysis Parameters
st.sidebar.subheader("Search Parameters")
search_radius = st.sidebar.slider(
    "Foursquare Search Radius (m)", 
    200, 1000, 400, 
    help="Radius for venue search around each grid point"
)
competitor_radius = st.sidebar.slider(
    "Competitor Analysis Radius (m)", 
    100, 500, 200, 
    help="Radius for competitor density analysis"
)

# Weight sliders
st.sidebar.subheader("Criteria Weights")
competition_weight = st.sidebar.slider("Competition Analysis", 0.0, 1.0, 0.35)
demo_weight = st.sidebar.slider("Demographics", 0.0, 1.0, 0.25)
metro_weight = st.sidebar.slider("Metro Accessibility", 0.0, 1.0, 0.25)
price_weight = st.sidebar.slider("Rental Affordability", 0.0, 1.0, 0.15)

# Normalize weights
total_weight = competition_weight + demo_weight + metro_weight + price_weight
if total_weight > 0:
    competition_weight /= total_weight
    demo_weight /= total_weight
    metro_weight /= total_weight
    price_weight /= total_weight

# Foursquare API functions
@lru_cache(maxsize=2000)
def get_foursquare_venues_cached(lat_lng_str, radius, limit):
    """Get all dining & drinking venues from Foursquare"""
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
        "categories": "13000",  # Dining and Drinking category ID
        "fields": "name,categories,location,fsq_category_labels,popularity,price,rating,stats"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        st.session_state["foursquare_calls"] += 1
        
        if response.status_code == 200:
            data = response.json()
            # Filter by specific dining & drinking subcategories
            filtered_results = []
            for venue in data.get("results", []):
                fsq_labels = venue.get("fsq_category_labels", [])
                if any(label in DINING_DRINKING_CATEGORIES for label in fsq_labels):
                    filtered_results.append(venue)
            
            return {"results": filtered_results}
        else:
            return {"results": []}
    except Exception as e:
        st.error(f"API Error: {e}")
        return {"results": []}

def analyze_location_competition(lat, lng, target_categories, search_radius, competitor_radius):
    """Analyze competition and business environment for a location"""
    
    cache_key = f"{lat:.4f},{lng:.4f},{search_radius},{competitor_radius}"
    if cache_key in st.session_state["foursquare_cache"]:
        return st.session_state["foursquare_cache"][cache_key]
    
    # Get all venues in search area
    all_venues = get_foursquare_venues_cached(f"{lat},{lng}", search_radius, 50)
    venues = all_venues.get("results", [])
    
    # Get competitors in smaller radius
    competitor_venues = get_foursquare_venues_cached(f"{lat},{lng}", competitor_radius, 30)
    competitor_data = competitor_venues.get("results", [])
    
    # Filter competitors by target categories
    direct_competitors = []
    for venue in competitor_data:
        venue_labels = venue.get("fsq_category_labels", [])
        if any(label in target_categories for label in venue_labels):
            direct_competitors.append(venue)
    
    # Calculate metrics
    total_venues = len(venues)
    competitor_count = len(direct_competitors)
    
    # Popularity analysis
    popularity_scores = [v.get("popularity", 0) for v in venues if v.get("popularity", 0) > 0]
    avg_popularity = np.mean(popularity_scores) if popularity_scores else 0
    
    # Price analysis
    price_levels = [v.get("price", 2) for v in venues if v.get("price")]
    avg_price_level = np.mean(price_levels) if price_levels else 2
    
    # Rating analysis
    ratings = [v.get("rating", 0) for v in venues if v.get("rating", 0) > 0]
    avg_rating = np.mean(ratings) if ratings else 0
    
    # Category diversity
    all_categories = set()
    for venue in venues:
        for label in venue.get("fsq_category_labels", []):
            all_categories.add(label)
    category_diversity = len(all_categories)
    
    # Competition score (lower competition = higher score)
    # Optimal competition: some venues for foot traffic, but not too many direct competitors
    foot_traffic_score = min(total_venues / 30, 1.0)  # Normalize to 30 venues max
    competition_penalty = max(0, 1 - (competitor_count / 8))  # Penalize >8 direct competitors
    
    # Quality and popularity boost
    popularity_boost = min(avg_popularity / 80, 1.0)  # Normalize popularity
    quality_boost = min(avg_rating / 10, 1.0)  # Normalize rating
    
    # Final competition score
    competition_score = (
        foot_traffic_score * 0.4 +      # Want some nearby venues
        competition_penalty * 0.3 +      # But not too many competitors
        popularity_boost * 0.2 +         # High popularity area
        quality_boost * 0.1              # High quality establishments
    )
    
    result = {
        'competition_score': competition_score,
        'total_venues': total_venues,
        'direct_competitors': competitor_count,
        'avg_popularity': avg_popularity,
        'avg_rating': avg_rating,
        'avg_price_level': avg_price_level,
        'category_diversity': category_diversity,
        'venue_details': venues[:10]  # Store top 10 venues for details
    }
    
    st.session_state["foursquare_cache"][cache_key] = result
    return result

# Load static data (demographics, metro, real estate)
@st.cache_data
def load_static_data():
    metro = gpd.read_file('metro_stations.geojson').to_crs(epsg=4326)
    demographics = gpd.read_file('demographics.geojson').to_crs(epsg=4326)
    real_estate = gpd.read_file('real_estate.geojson').to_crs(epsg=4326)
    return metro, demographics, real_estate

metro, demographics, real_estate = load_static_data()

# City bounds
city_bounds = {
    "Madrid": {"min_lon": -3.8, "max_lon": -3.5, "min_lat": 40.3, "max_lat": 40.5}
}

# Main analysis function
@st.cache_data
def run_foursquare_analysis(_metro, _demographics, _real_estate, business_type, 
                          price_segment, audience_segment, weights, _city, 
                          search_radius, competitor_radius):
    
    # Create analysis grid
    min_lon, max_lon = city_bounds[_city]["min_lon"], city_bounds[_city]["max_lon"]
    min_lat, max_lat = city_bounds[_city]["min_lat"], city_bounds[_city]["max_lat"]
    grid_size = 0.008  # Approximately 800m grid

    grid_points = []
    for lon in np.arange(min_lon, max_lon, grid_size):
        for lat in np.arange(min_lat, max_lat, grid_size):
            center_lon = lon + grid_size / 2
            center_lat = lat + grid_size / 2
            grid_points.append({'lon': center_lon, 'lat': center_lat})

    # Limit grid size for API efficiency
    if len(grid_points) > MAX_CALLS_PER_ANALYSIS:
        sample_indices = np.random.choice(len(grid_points), MAX_CALLS_PER_ANALYSIS, replace=False)
        grid_points = [grid_points[i] for i in sample_indices]
    
    st.info(f"Analyzing {len(grid_points)} locations...")
    
    # Get target categories for selected business type
    target_categories = CATEGORY_GROUPS[business_type]
    
    # Analyze each location
    results = []
    progress_bar = st.progress(0)
    
    for idx, point in enumerate(grid_points):
        progress_bar.progress((idx + 1) / len(grid_points))
        
        lat, lon = point['lat'], point['lon']
        centroid_point = Point(lon, lat)
        
        # 1. Competition Analysis (Foursquare)
        competition_data = analyze_location_competition(
            lat, lon, target_categories, search_radius, competitor_radius
        )
        
        # 2. Demographics
        demo_match = _demographics[_demographics.geometry.contains(centroid_point)]
        if not demo_match.empty:
            if audience_segment == "Young":
                demo_score = demo_match.get('young_population', demo_match.get('young_pop_dens', 0)).iloc[0]
            elif audience_segment == "Aged":
                demo_score = demo_match.get('aged_population', demo_match.get('old_pop_dens', 0)).iloc[0]
            else:
                demo_score = demo_match['total_pop_density'].iloc[0]
        else:
            demo_score = _demographics['total_pop_density'].mean()
        
        # 3. Metro Accessibility
        min_distance = _metro.geometry.distance(centroid_point).min()
        metro_score = 1 / (1 + min_distance * 100)  # Convert to more reasonable scale
        
        # 4. Real Estate
        price_match = _real_estate[_real_estate.geometry.contains(centroid_point)]
        if not price_match.empty:
            rental_price = price_match['sqm_rental'].iloc[0]
        else:
            rental_price = _real_estate['sqm_rental'].mean()
        
        price_score = 1 / rental_price if price_segment == "Low-Cost" else rental_price / 100
        
        results.append({
            'lon': lon,
            'lat': lat,
            'competition_score': competition_data['competition_score'],
            'demo_score': demo_score,
            'metro_score': metro_score,
            'price_score': price_score,
            'total_venues': competition_data['total_venues'],
            'direct_competitors': competition_data['direct_competitors'],
            'avg_popularity': competition_data['avg_popularity'],
            'avg_rating': competition_data['avg_rating'],
            'category_diversity': competition_data['category_diversity'],
            'rental_price': rental_price
        })
        
        # Rate limiting
        time.sleep(0.05)
    
    progress_bar.progress(1.0)
    progress_bar.empty()
    
    # Normalize and calculate final scores
    results_df = pd.DataFrame(results)
    
    # Normalize scores
    scaler = MinMaxScaler()
    score_columns = ['competition_score', 'demo_score', 'metro_score', 'price_score']
    results_df[score_columns] = scaler.fit_transform(results_df[score_columns])
    
    # Calculate weighted final score
    results_df['final_score'] = (
        weights['competition'] * results_df['competition_score'] +
        weights['demo'] * results_df['demo_score'] +
        weights['metro'] * results_df['metro_score'] +
        weights['price'] * results_df['price_score']
    )
    
    return results_df.sort_values('final_score', ascending=False)

# UI Controls
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Run Analysis", use_container_width=True):
        if not FOURSQUARE_API_KEY or FOURSQUARE_API_KEY == "YOUR_API_KEY_HERE":
            st.error("Please add FOURSQUARE_API_KEY to secrets.toml")
        else:
            weights = {
                'competition': competition_weight,
                'demo': demo_weight,
                'metro': metro_weight,
                'price': price_weight
            }
            
            with st.spinner("Analyzing locations with Foursquare data..."):
                st.session_state["ranked_locations"] = run_foursquare_analysis(
                    metro, demographics, real_estate, business_type,
                    price_segment, audience_segment, weights, city,
                    search_radius, competitor_radius
                )
            
            # Create map
            results = st.session_state["ranked_locations"]
            m = folium.Map(location=(40.40, -3.65), zoom_start=11, tiles="CartoDB positron")
            
            # Add top 30 locations to map
            top_locations = results.head(30)
            
            for _, row in top_locations.iterrows():
                # Color by score percentile
                score_percentile = (row['final_score'] - results['final_score'].min()) / (results['final_score'].max() - results['final_score'].min())
                
                if score_percentile > 0.9:
                    color, radius = 'darkgreen', 10
                elif score_percentile > 0.7:
                    color, radius = 'green', 8
                elif score_percentile > 0.5:
                    color, radius = 'orange', 6
                else:
                    color, radius = 'red', 4
                
                popup_html = f"""
                <div style="width:250px">
                <b>Score: {row['final_score']:.3f}</b><br>
                📍 {row['lat']:.4f}, {row['lon']:.4f}<br>
                🏪 Total Venues: {row['total_venues']}<br>
                🎯 Direct Competitors: {row['direct_competitors']}<br>
                ⭐ Avg Rating: {row['avg_rating']:.1f}/10<br>
                📊 Popularity: {row['avg_popularity']:.0f}<br>
                💰 Rent: €{row['rental_price']:.0f}/m²<br>
                🎨 Categories: {row['category_diversity']}
                </div>
                """
                
                folium.CircleMarker(
                    location=(row['lat'], row['lon']),
                    radius=radius,
                    color=color,
                    fill=True,
                    fillOpacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)
            
            st.session_state["map"] = m
            st.success(f"Analysis complete! Found {len(results)} locations.")

with col2:
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state["ranked_locations"] = None
        st.session_state["map"] = None
        st.rerun()

# Display results
if st.session_state.get("ranked_locations") is not None:
    results = st.session_state["ranked_locations"]
    
    st.subheader("🎯 Top Locations for " + business_type)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Locations Analyzed", len(results))
    with col2:
        avg_competitors = results['direct_competitors'].mean()
        st.metric("Avg Competitors", f"{avg_competitors:.1f}")
    with col3:
        avg_venues = results['total_venues'].mean()
        st.metric("Avg Total Venues", f"{avg_venues:.1f}")
    with col4:
        st.metric("API Calls Used", st.session_state['foursquare_calls'])
    
    # Top results table
    top_10 = results.head(10)[['lon', 'lat', 'final_score', 'total_venues', 'direct_competitors', 
                              'avg_rating', 'avg_popularity', 'rental_price']].round(3)
    
    top_10.columns = ['Longitude', 'Latitude', 'Final Score', 'Total Venues', 
                     'Direct Competitors', 'Avg Rating', 'Popularity', 'Rent €/m²']
    
    st.dataframe(top_10, use_container_width=True)
    
    # Map
    if st.session_state.get("map"):
        st.subheader("📍 Location Map")
        st_folium(st.session_state["map"], width=700, height=500)
    
    # Download
    st.download_button(
        label="📥 Download Results",
        data=results.to_csv(index=False),
        file_name=f"oppla_foursquare_{business_type.lower().replace(' ', '_')}_{city}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("**OPPLA** - Location Intelligence powered by Foursquare Places API 🚀")