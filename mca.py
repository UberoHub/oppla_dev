import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load GeoJSON files
pois = gpd.read_file('madrid_pois.geojson')  # Point layer: geometry, category
metro = gpd.read_file('madrid_metro_stops.geojson')  # Point layer: geometry, station_name
demographics = gpd.read_file('madrid_demographics.geojson')  # Polygon layer: geometry, population, avg_income
real_estate = gpd.read_file('madrid_real_estate.geojson')  # Polygon layer: geometry, price_per_sqm

# Ensure consistent CRS (WGS84)
pois = pois.to_crs(epsg=4326)
metro = metro.to_crs(epsg=4326)
demographics = demographics.to_crs(epsg=4326)
real_estate = real_estate.to_crs(epsg=4326)

# Step 1: Create a grid over Madrid (e.g., 500m x 500m)
# Define Madrid's bounding box (approximate)
min_lon, max_lon = -3.8, -3.5
min_lat, max_lat = 40.3, 40.5
grid_size = 0.005  # Approx 500m in degrees

# Generate grid cells
grid_cells = []
grid_centroids = []
for lon in np.arange(min_lon, max_lon, grid_size):
    for lat in np.arange(min_lat, max_lat, grid_size):
        # Create a rectangular cell (box)
        cell = box(lon, lat, lon + grid_size, lat + grid_size)
        # Centroid for scoring
        centroid = Point(lon + grid_size / 2, lat + grid_size / 2)
        grid_cells.append(cell)
        grid_centroids.append({'geometry': centroid, 'lon': centroid.x, 'lat': centroid.y})

# Create GeoDataFrame for grid
grid_gdf = gpd.GeoDataFrame(grid_centroids, geometry='geometry', crs='EPSG:4326')
grid_cells_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs='EPSG:4326')

# Step 2: Score each criterion
scores = []
for idx, centroid in grid_gdf.iterrows():
    cell = grid_cells_gdf.iloc[idx].geometry
    centroid_point = centroid.geometry

    # POI Density: Count POIs within the grid cell
    poi_count = len(pois[pois.geometry.within(cell)])

    # Demographics: Spatial join to get demographics for the cell's centroid
    # Find the neighborhood containing the centroid
    demo_match = demographics[demographics.geometry.contains(centroid_point)]
    demo_score = demo_match['total_pop_density'].iloc[0] if not demo_match.empty else demographics['total_pop_density'].mean()

    # Metro Accessibility: Distance to nearest metro station
    min_distance = metro.geometry.distance(centroid_point).min()
    metro_score = 1 / (1 + min_distance)  # Inverse distance for higher score

    # Real Estate: Spatial join to get price for the cell's centroid
    price_match = real_estate[real_estate.geometry.contains(centroid_point)]
    price = price_match['sqm_rental'].iloc[0] if not price_match.empty else real_estate['sqm_rental'].mean()
    price_score = 1 / price  # Inverse price for higher score

    scores.append({
        'lon': centroid.lon,
        'lat': centroid.lat,
        'poi_score': poi_count,
        'demo_score': demo_score,
        'metro_score': metro_score,
        'price_score': price_score
    })

# Step 3: Normalize scores
scores_df = pd.DataFrame(scores)
scaler = MinMaxScaler()
normalized_scores = scaler.fit_transform(scores_df[['poi_score', 'demo_score', 'metro_score', 'price_score']])
scores_df[['poi_score', 'demo_score', 'metro_score', 'price_score']] = normalized_scores

# Step 4: Weighted sum
weights = {'poi_score': 0.3, 'demo_score': 0.3, 'metro_score': 0.2, 'price_score': 0.2}
scores_df['final_score'] = (
        weights['poi_score'] * scores_df['poi_score'] +
        weights['demo_score'] * scores_df['demo_score'] +
        weights['metro_score'] * scores_df['metro_score'] +
        weights['price_score'] * scores_df['price_score']
)

# Step 5: Rank locations
ranked_locations = scores_df.sort_values(by='final_score', ascending=False)

# Save results as CSV or GeoJSON
ranked_locations.to_csv('ranked_locations.csv')
# Optionally save as GeoJSON
ranked_gdf = gpd.GeoDataFrame(
    ranked_locations,
    geometry=[Point(lon, lat) for lon, lat in zip(ranked_locations['lon'], ranked_locations['lat'])],
    crs='EPSG:4326'
)
ranked_gdf.to_file('ranked_locations.geojson', driver='GeoJSON')

# Step 6: Visualize with Streamlit
st.title("Optimal Business Locations in Madrid")
st.map(ranked_locations[['lat', 'lon']].rename(columns={'lat': 'latitude', 'lon': 'longitude'}))
st.write("Top 5 Locations:")
st.dataframe(ranked_locations[['lon', 'lat', 'final_score']].head())
