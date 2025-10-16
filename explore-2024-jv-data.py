import os, sys
from matplotlib import pyplot as plt
import time
import geopandas as gpd
import numpy as np
from pyproj import Transformer
import geopy.distance
import pickle


# Global vars
PATH_TO_GPKG = "../jv-data/tillRise_2024_v2.gpkg"

# Now read the gpkg file, and create a thing similar to csv_entries
# but called gpkg_entries
gdf = gpd.read_file(PATH_TO_GPKG)
polys = []
activities = []
comments = []
date_stamps = []
# The activities are the following: 'grazed', None, 'light grazed', 'Betad', 'Svagt betad', 'betad'.
# We lump together: 'grazed' <-- 'grazed', 'Betad', 'betad'
#                   'light grazed' <-- 'light grazed', 'Svagt betad'
#                   'no activity' <-- None
for idx, row in gdf.iterrows():
    poly_data = row['geometry']
    if poly_data.geom_type == 'Polygon':
        polys.append(poly_data)
        activity = row['event']
        if activity is None:
            activity = 'no activity'
        elif activity == 'Betad' or activity == 'betad':
            activity = 'grazed'
        elif activity == 'Svagt betad':
            activity = 'light grazed'
        activities.append(activity)
        comments.append(row['comment'])
        # row['timestamp']is a Timestamp object and has format yyyy-mm-dd hh:mm:ss,
        # and we only want the date part
        date_stamps.append(row['timestamp'].date())
    elif poly_data.geom_type == 'MultiPolygon':
        for sub_poly in poly_data.geoms:
            polys.append(sub_poly)
            activity = row['event']
            if activity is None:
                activity = 'no activity'
            elif activity == 'Betad' or activity == 'betad':
                activity = 'grazed'
            elif activity == 'Svagt betad':
                activity = 'light grazed'
            activities.append(activity)
            comments.append(row['comment'])
            # row['timestamp']is a Timestamp object and has format yyyy-mm-dd hh:mm:ss,
            # and we only want the date part
            date_stamps.append(row['timestamp'].date())

# Some assertions
assert None not in activities
assert len(polys) == len(activities) == len(comments) == len(date_stamps)

# Convert polygon geometries into lon-lat coordinates
transformer = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
lon_lats_all = []
for idx, poly in enumerate(polys):
    # The type of poly above is <class 'shapely.geometry.polygon.Polygon'>.
    # We want to go over all its coordinates and convert them to lon-lat
    lon_lats = []
    for point in poly.exterior.coords:
        # Due to always_xy=True, the order of output is (lon, lat)
        # according to the documentation of pyproj
        lon, lat = transformer.transform(point[0], point[1])
        lon_lats.append((lon, lat))
    lon_lats_np = np.array(lon_lats)
    lon_lats_all.append(lon_lats_np)

# Collect some statistics
uniques_and_counts = {}
for activity in activities:
    if activity not in uniques_and_counts:
        uniques_and_counts[activity] = 1
    else:
        uniques_and_counts[activity] += 1
print("Unique activities and their counts: ", uniques_and_counts)
# print unique dates and their counts
unique_dates = set(date_stamps)
date_counts = {date: date_stamps.count(date) for date in unique_dates}
print("Date counts: ", date_counts)

# Save dataset as .pkl. The dataset is a list, where each element is a tuple
# with 3 elements: comment, actvity, and polygon.
dataset_to_save = []
for idx, poly in enumerate(lon_lats_all):
    # Note: The date stamp below is not a series of dates, but simply the
    # inspection date of the polygon.
    dataset_to_save.append((comments[idx], activities[idx], poly, date_stamps[idx]))
with open('../jv-data/2024-dataset.pkl', 'wb') as f:
    pickle.dump(dataset_to_save, f)

# Analyze the actual polygons a bit (e.g. check where in Sweden they are, what total area they span, etc.)
all_min_lons = []
all_max_lons = []
all_min_lats = []
all_max_lats = []
for poly in lon_lats_all:
    min_lon = np.min(poly[:,0])
    max_lon = np.max(poly[:,0])
    min_lat = np.min(poly[:,1])
    max_lat = np.max(poly[:,1])
    all_min_lons.append(min_lon)
    all_max_lons.append(max_lon)
    all_min_lats.append(min_lat)
    all_max_lats.append(max_lat)
    
# Print out the minimum and maximum latitudes and longitudes based on the four lists
min_min_lat = np.min(all_min_lats)
max_max_lat = np.max(all_max_lats)
min_min_lon = np.min(all_min_lons)
max_max_lon = np.max(all_max_lons)

# Based on the above "meta bounding box", figure out the height and with in km.
# This is useful for setting the size of the satellite images to download.
lat_span_km = geopy.distance.distance((min_min_lat, min_min_lon), (max_max_lat, min_min_lon)).kilometers
lon_span_km = geopy.distance.distance((min_min_lat, min_min_lon), (min_min_lat, max_max_lon)).kilometers
print(f"Latitude span (km): {lat_span_km:.2f}")
print(f"Longitude span (km): {lon_span_km:.2f}")

# Load high-resolution country boundaries
print("Plotting polygons...")
sweden = gpd.read_file("../country-borders/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp")  # Update with your actual path

# Filter to only Sweden
sweden = sweden[sweden['SOVEREIGNT'] == 'Sweden']

# Create 3-col-1-row plot and plot Sweden's boundary in both
fig, ax = plt.subplots(1, 1, figsize=(15, 7))
sweden.boundary.plot(ax=ax, color="black", linewidth=1)

# Scatter plot min-coords
ax.scatter(all_min_lons, all_min_lats, color='red', s=10, label='Current Polygons')
ax.grid(True)
ax.set_title('2024 polygons')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Save figure
plt.savefig('polygons_2024.png')
plt.cla()
plt.clf()
plt.close('all')

print("Number of polygons: ", len(lon_lats_all))