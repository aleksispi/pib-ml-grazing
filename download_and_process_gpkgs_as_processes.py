import os, sys
import openeo
import argparse
import shutil
import numpy as np
from utils import create_square_bounding_box, time_measurement

# Add argument parser for reading the arguments POLY_IDX_START and POLY_IDX_END
# from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--POLY_IDX_START", type=int, default=0, help="Index of first polygon to process")
parser.add_argument("--POLY_IDX_END", type=int, default=999, help="End index of polygons to process (exclusive)")
args = parser.parse_args()

# Global vars
CONNECTION = 'des'  # 'des' (Digital Earth Sweden) or 'cop' (Copernicus)
COLLECTION = 'SENTINEL2_L2A'# 'SENTINEL2_L2A' or 'SENTINEL1_GRD'  # Collection to use
MARGIN_PER_SIDE = 0.85#  # Margin around polygon when extracting bounding box (if negative, means percentage of polygon size, otherwise this specifies the size in absolut terms, so each polygon gets a same-sized image to it)
PKL_PATH = "../jv-data/2024-dataset.pkl"  # Path to .pkl file (2024 data)
#PKL_PATH = "../jv-data/2022-dataset.pkl"  # Path to .pkl file (2022 data)
#PKL_PATH = "../jv-data/2024-dataset-more.pkl"  # Path to .pkl file (2024 data)
POLY_IDX_START = args.POLY_IDX_START  # Index of first polygon to process
POLY_IDX_END = args.POLY_IDX_END  # End index of polygons to process (exclusive)
SAVE_PATH = "../data_and_polygons_2024"  # Where to save the data and polygons
DATE_SPAN = ['2024-04-01', '2024-10-21']  # Date span over which to get the data

# Load the geometries
geometries = np.load(PKL_PATH, allow_pickle=True)
# The below [2] indexing is due to how saved in explore-2022-jv-data.py and explore-2024-jv-data.py
geometries = [geometry[2] for geometry in geometries]
# Get only the one between POLY_IDX_START and POLY_IDX_END.
geometries = geometries[POLY_IDX_START:POLY_IDX_END]
if POLY_IDX_END < 0:
    POLY_IDX_END = len(geometries) + POLY_IDX_START + 1

# Set up connection to the EO data service and specify the collection and bands to use
if CONNECTION == 'des':
    connection = openeo.connect("https://openeo.digitalearth.se")
    print("EO service URL:", "https://openeo.digitalearth.se")
    print("testuser")
    if COLLECTION == 'SENTINEL2_L2A':
        collection = "s2_msi_l2a"
    bands = {"b01": 60, "b02": 10, "b03": 10, "b04": 10, "b05": 20, "b06": 20, "b07": 20, "b08": 10, "b8a": 20, "b09": 60, "b11": 20, "b12": 20}
    # Separate bands into bands_10x10, bands_20x20, bands_60x60, based on values in the bands dictionary
    bands_10x10 = [key for key, value in bands.items() if value == 10]
    bands_20x20 = [key for key, value in bands.items() if value == 20]
    bands_60x60 = [key for key, value in bands.items() if value == 60]
    band_types = {"10x10": bands_10x10, "20x20": bands_20x20, "60x60": bands_60x60}
elif CONNECTION == 'cop':
    connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
    collection = COLLECTION
    if collection == 'SENTINEL1_GRD':
        bands = ['VV', 'VH']
        band_types = {"10x10": bands}
    elif collection == 'SENTINEL2_L2A':
        bands = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B11": 20, "B12": 20}
        bands_10x10 = [key for key, value in bands.items() if value == 10]
        bands_20x20 = [key for key, value in bands.items() if value == 20]
        bands_60x60 = [key for key, value in bands.items() if value == 60]
        band_types = {"10x10": bands_10x10, "20x20": bands_20x20, "60x60": bands_60x60}
connection.authenticate_oidc()

# Begin processing the polygons
print("Processing polygons from the .pkl file...")
poly_idxs = np.arange(POLY_IDX_START, POLY_IDX_END)
with time_measurement("Putting jobs for all polygons"):  
    for i, poly_idx in enumerate(poly_idxs):

        # Extract the current polygon, where first coordinate is longitude and second is latitude
        polygon = geometries[i]

        # Based on the above polygon, create the minimum bounding box containing the polygon
        # (but with some extra space around the polygon)
        min_lon = np.min(polygon[:, 0])
        max_lon = np.max(polygon[:, 0])
        min_lat = np.min(polygon[:, 1])
        max_lat = np.max(polygon[:, 1])

        # Need to ensure we get a square satellite image in the end
        # In the below, when negative, treated instead as relative margin on each side
        min_lat, min_lon, max_lat, max_lon, _, _ = create_square_bounding_box(
            min_lat, min_lon, max_lat, max_lon, width_km=MARGIN_PER_SIDE
        )

        # Get Sentinel 2 data for the bounding box
        with time_measurement(f" Processing polygon idx {poly_idx}/{POLY_IDX_END}..."):

            # Download the data
            for key, bands in band_types.items():
                job_designation = f"Area_{poly_idx+1}" + "_" + key

                # Load the data cube
                cube = connection.load_collection(
                    collection,
                    spatial_extent={
                        "west": min_lon,
                        "south": min_lat,
                        "east": max_lon,
                        "north": max_lat,
                    },
                    temporal_extent=DATE_SPAN,
                    bands=bands
                )

                # Apply the 'sar_backscatter' process
                elev_model = "COPERNICUS_30"  # None, "COPERNICUS_30" or "ASTER"
                if collection == 'SENTINEL1_GRD':
                    cube = cube.process(
                        process_id="sar_backscatter",
                        arguments={
                            "data": cube,                      # Input data cube
                            "coefficient": "sigma0-ellipsoid", # Use supported coefficient (default "gamma0-terrain" is not allowed)
                            "elevation_model": elev_model,     # Use default DEM (if available)
                            "mask": False,                     # Optional: Add mask band (default False)
                            "contributing_area": False,        # Optional: Add contributing area band (default False)
                            "local_incidence_angle": False,    # Optional: Add local incidence angle band (default False)
                            "ellipsoid_incidence_angle": False,# Optional: Add ellipsoidal incidence angle band (default False)
                            "noise_removal": True,             # Optional: Remove noise (default True)
                            "options": {}                      # Optional: Backend-specific options
                        }
                    )

                job = cube.create_job(
                    out_format="netCDF",
                    options={"max_files": 200},
                    title=f"JV-2022-Polys-Sen1 {job_designation}",
                    description=f"Downloads the bounding square of the given polygon nr {poly_idx}",
                )
                
                # Time to do the actual download

                # Base the filenames on the polygon number i and the key (10x10, 20x20, 60x60)
                file_name_nc = os.path.join(SAVE_PATH, f"Area_{poly_idx+1}" + "_" + key + '_image.nc')
                # Continue if it already exists.
                if os.path.exists(file_name_nc):
                    print(f"Skipping {file_name_nc} as it already exists.")
                    continue

                # Download the data
                result = job.start_and_wait().download_results()
            
                # Save the .nc file to disk
                nc_file_path = None
                for path, info in result.items():
                    if '.nc' in str(path):
                        nc_file_path = path
                        break
                if nc_file_path:
                    shutil.move(nc_file_path, file_name_nc)
                    print(f"Saved {file_name_nc}")
                else:
                    print("No .nc file found in the results.")
                    sys.exit()

            # Also save the associated polygon (in original coordinates) as an .npy file
            np.save(os.path.join(SAVE_PATH, f"Area_{poly_idx+1}" + '_polygon.npy'), polygon)

