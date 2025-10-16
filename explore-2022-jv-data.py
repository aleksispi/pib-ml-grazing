import sys
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import Transformer
import geopy.distance
import re
import pickle

# Global vars
PATH_TO_CSV = "../jv-data/grdkod52_53_RED_2022_fieldresult.csv"
PATH_TO_GPKG = "../jv-data/grdkod52_53_RED_2022.gpkg"
MIN_AREA_TO_KEEP = 0.0  # Minimum area of a polygon bounding box to keep it (in km2)

# Read the csv file
df = pd.read_csv(PATH_TO_CSV, encoding='latin1')

# Extract all values in df
values = df.iloc[:, 0].values  # somewhat unclear why this gives ALL columns, but it does...
csv_entries = []
for idx, value in enumerate(values):
    string_split = value.split(";")
    entities = []
    for entity in string_split:
        # Some of the entities have the format '"text"', so remove the double quotes
        entity = entity.replace('"', '')
        entities.append(entity)
    csv_entries.append(entities)
assert len(csv_entries) == len(df)

# Now read the gpkg file, and create a thing similar to csv_entries
# but called gpkg_entries
gdf = gpd.read_file(PATH_TO_GPKG)
# Extract all values in gdf
ids = gdf.iloc[:, 0].values
codes = gdf.iloc[:, 1].values
polys = gdf.iloc[:, 2].values
gpkg_entries = []
for idx, code in enumerate(codes):
    gpkg_entries.append([ids[idx], code, polys[idx]])
assert len(gpkg_entries) == len(gdf)

# Assert that all ids in csv_entries are unique, and similarly for gpkg_entries.
csv_ids = [csv_entry[0] for csv_entry in csv_entries]
gpkg_ids = [gpkg_entry[0] for gpkg_entry in gpkg_entries]
assert len(csv_ids) == len(set(csv_ids))
assert len(gpkg_ids) == len(set(gpkg_ids))

# Next, get the subset of gpkg_entries that are in csv_entries, based on the id.
# Also, only keep those csv_entries that are in gpkg_entries
csv_subset = []
gpkg_subset = []
for csv_entry in csv_entries:
    csv_id = csv_entry[0]
    for gpkg_entry in gpkg_entries:
        gpkg_id = gpkg_entry[0]
        if csv_id == gpkg_id:
            gpkg_subset.append(gpkg_entry)
            csv_subset.append(csv_entry)
            break
assert len(gpkg_subset) == len(csv_subset)
# Also assert that all ids match between csv_subset and gpkg_subset.
csv_ids = [csv_entry[0] for csv_entry in csv_subset]
gpkg_ids = [gpkg_entry[0] for gpkg_entry in gpkg_subset]
assert csv_ids == gpkg_ids

# Convert polygon geometries into lon-lat coordinates
transformer = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
new_gpkg_subset = []  # may be expanded when splitting MultiPolygons into Polygons
new_csv_subset = []  # may be expanded when splitting MultiPolygons into Polygons
for some_idx, gpkg_entry in enumerate(gpkg_subset):
    curr_polys = []  # MultiPolygon as list of Polygon
    if gpkg_entry[2].geom_type == 'Polygon':
        curr_polys.append(gpkg_entry[2])
    elif gpkg_entry[2].geom_type == 'MultiPolygon':
        for sub_poly in gpkg_entry[2].geoms:
            curr_polys.append(sub_poly)
    for poly in curr_polys:
        # The type of poly above is <class 'shapely.geometry.polygon.Polygon'>.
        # We want to go over all its coordinates and convert them to lon-lat
        lon_lats = []
        for point in poly.exterior.coords:
            # Due to always_xy=True, the order of output is (lon, lat)
            # according to the documentation of pyproj
            lon, lat = transformer.transform(point[0], point[1])
            lon_lats.append((lon, lat))
        lon_lats_np = np.array(lon_lats)
        
        # Check if the area of the bounding box is larger than MIN_AREA_TO_KEEP.
        # If not, skip this polygon.
        min_lon = np.min(lon_lats_np[:,0])
        max_lon = np.max(lon_lats_np[:,0])
        min_lat = np.min(lon_lats_np[:,1])
        max_lat = np.max(lon_lats_np[:,1])
        area = geopy.distance.distance((min_lat, min_lon), (max_lat, max_lon)).kilometers
        if area < MIN_AREA_TO_KEEP:
            continue

        new_gpkg_subset.append([gpkg_entry[0], gpkg_entry[1], lon_lats_np])
        new_csv_subset.append(csv_subset[some_idx])
assert len(new_gpkg_subset) == len(new_csv_subset)
gpkg_subset = new_gpkg_subset
csv_subset = new_csv_subset

# Collect some statistics
nbr_certain_grazed = 0
nbr_unclear_activity = 0
nbr_harvest = 0
nbr_no_activity = 0
date_stamps = []
for csv_entry in csv_subset:
    if csv_entry[1] == "Aktivitet bete":
        nbr_certain_grazed += 1
    if csv_entry[1] == "Aktivitet skörd":
        nbr_harvest += 1
    if csv_entry[1] == "Osäker men aktivitet":
        nbr_unclear_activity += 1
    if csv_entry[1] == "Ingen aktivitet":
        nbr_no_activity += 1
    # Also get date stamp if available
    dcim_path = csv_entry[3]
    # The date is always the first 8 digit characters in the dcim_path,
    # so use regex to extract the first 8 digits
    if 'DCIM' in dcim_path:
        date_stamp = re.search(r'\d{8}', dcim_path).group()
    else:
        date_stamp = None
    date_stamps.append(date_stamp)
# print unique dates and their counts
unique_dates = set(date_stamps)
date_counts = {date: date_stamps.count(date) for date in unique_dates}
print("Date counts: ", date_counts)
print("Number of entries in the csv subset: ", len(csv_subset))
print("Different kinds of activities there exist: ")
activities = set()
for csv_entry in csv_subset:
    activities.add(csv_entry[1])
print(activities)
prct_certain_grazed = 100*nbr_certain_grazed/len(csv_subset)
prct_unclear_activity = 100*nbr_unclear_activity/len(csv_subset)
prct_harvest = 100*nbr_harvest/len(csv_subset)
prct_no_activity = 100*nbr_no_activity/len(csv_subset)
print(f"Number of certain grazed entries (total, percentage): {nbr_certain_grazed}, {prct_certain_grazed}")  
print(f"Number of unclear activity entries (total, percentage): {nbr_unclear_activity}, {prct_unclear_activity}")
print(f"Number of harvest entries (total, percentage): {nbr_harvest}, {prct_harvest}")
print(f"Number of no activity entries (total, percentage): {nbr_no_activity}, {prct_no_activity}")
print(f"Sum of percentages: {prct_certain_grazed + prct_unclear_activity + prct_harvest + prct_no_activity}")

# Save csv_entry and gpkg_entry as .pkl files
with open("../jv-data/csv_subset_2022.pkl", "wb") as f:
    pickle.dump(csv_subset, f)
with open("../jv-data/2022-dataset.pkl", "wb") as f:
    pickle.dump(gpkg_subset, f)

# Analyze the actual polygons a bit (e.g. check where in Sweden they are, what total area they span, etc.)
all_min_lons = []
all_max_lons = []
all_min_lats = []
all_max_lats = []
for gpkg_entry in gpkg_subset:
    poly = gpkg_entry[2]
    min_lon = np.min(poly[:,0])
    max_lon = np.max(poly[:,0])
    min_lat = np.min(poly[:,1])
    max_lat = np.max(poly[:,1])
    all_min_lons.append(min_lon)
    all_max_lons.append(max_lon)
    all_min_lats.append(min_lat)
    all_max_lats.append(max_lat)

# Check which is the largest and smallest bounding box, respectively.
smallest_area = 999999
largest_area = 0
for gpkg_entry in gpkg_subset:
    poly = gpkg_entry[2]
    min_lon = np.min(poly[:,0])
    max_lon = np.max(poly[:,0])
    min_lat = np.min(poly[:,1])
    max_lat = np.max(poly[:,1])
    area = geopy.distance.distance((min_lat, min_lon), (max_lat, max_lon)).kilometers
    if area < smallest_area:
        smallest_area = area
        smallest_poly = poly
    if area > largest_area:
        largest_area = area
        largest_poly = poly
print(f"Smallest area (km2): {smallest_area}, largest area (km2): {largest_area}")

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
ax.set_title('2022 polygons')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Save figure
plt.savefig('polygons_2022.png')
plt.cla()
plt.clf()
plt.close('all')

print("Number of polygons: ", len(gpkg_subset))