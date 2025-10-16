import os, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
import torch
import numpy as np
import config
from utils import read_sentinel1_nc, align_poly_with_image, create_square_bounding_box
from classes import MLP5
import geopy.distance
import pickle
import random
import re
import datetime

# Global vars
LOAD_PATH = "../sen1_data_and_polygons_2024"
CSV_PATH = '../jv-data/csv_subset_all_2022.npy'  # Path to csv npy file with activity labels
PKL_SAVE_NAME = 'ml_dataset_seed1_sen1.pkl'  # Name of the pickle file to save the dataset to
DEVICE = 'cpu'
SEED = 1
START_DATE = '2024-04-01'  # Start date for the timeseries
END_DATE = '2024-10-21'  # End date for the timeseries (exclusive)
START_IDX = 0  # Start index for the timeseries
END_IDX = -1  # End index for the timeseries (exclusive) -- if -1, then use all
if '2022' in START_DATE:
    IDXS_TO_USE = [0, 1, 2, 6, 10, 26, 27, 29, 33, 40, 42, 48, 50, 51, 52, 53, 54, 57, 60, 61, 62, 71, 73, 80, 81, 82, 88, 91, 98, 99, 100, 103, 114, 115, 116, 129, 135, 137, 138, 146, 147,
                   149, 155, 156, 160, 162, 163, 164, 172, 174, 175, 182, 213, 222, 226, 229, 232, 233, 247, 278, 282, 7, 11, 12, 14, 18, 25, 28, 30, 31, 32, 34, 35, 38, 43, 44, 46, 66, 67,
                   68, 69, 70, 72, 76, 77, 84, 85, 87, 94, 95, 105, 106, 107, 108, 109, 110, 111, 112, 117, 118, 122, 123, 126, 128, 131, 134, 145, 148, 150, 151, 152, 153, 154, 157, 161,
                   167, 171, 178, 183, 199, 204, 209, 211, 217, 218, 242, 243, 244, 248, 252, 257, 258, 261, 262, 265, 268, 274, 275, 276, 299, 300, 322, 324]
else:  # 2024 case below
    IDXS_TO_USE = [5, 6, 11, 12, 18, 19, 20, 33, 44, 48, 52, 58, 59, 63, 64, 65, 68, 69, 70, 71, 72, 74, 85, 99, 102, 103, 109, 112, 116, 118, 119, 121, 125, 129, 131, 133, 136, 139, 142,
                   160, 164, 165, 166, 168, 173, 177, 178, 179, 181, 183, 184, 187, 189, 191, 193, 194, 195, 196, 197, 198, 200, 201, 202, 203, 204, 205, 212, 213, 218, 219, 227, 231, 259,
                   270, 314, 346, 349, 354, 355, 356, 370, 371, 372, 373, 374, 375, 390, 397, 398, 399, 400, 407, 408, 412, 1, 3, 4, 7, 8, 9, 10, 14, 15, 16, 17, 22, 23, 24, 25, 30, 31, 32,
                   37, 38, 39, 40, 41, 42, 46, 47, 49, 55, 56, 62, 80, 83, 84, 86, 88, 89, 90, 92, 95, 96, 98, 100, 108, 110, 111, 113, 114, 120, 126, 127, 130, 138, 141, 143, 145, 146, 147,
                   148, 149, 151, 153, 154, 155, 156, 157, 161, 162, 167, 170, 171, 172, 174, 186, 188, 199, 206, 207, 208, 209, 210, 211, 217, 221, 222, 226, 228, 229, 230, 232, 233, 234,
                   235, 236, 237, 238, 240, 244, 246, 247, 253, 254, 256, 262, 263, 264, 265, 272, 277, 278, 279, 288, 289, 290, 291, 294, 296, 298, 300, 305, 306, 307, 311, 320, 322, 327,
                   331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 342, 344, 345, 348, 351, 352, 353, 357, 358, 359, 360, 361, 364, 365, 366, 377, 379, 380, 382, 384, 386, 387, 389, 391,
                   392, 396, 401, 402, 403, 404, 405, 406, 414, 417, 418, 423, 429, 432]
ACTIVITIES_TO_USE = None  # If set as specific list of activities, then only use those activities -- ['Osäker men aktivitet', 'Ingen aktivitet', 'Aktivitet skörd', 'Aktivitet bete']
TRAIN_FRAC = 0.8  # Fraction of data to use for training

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# List all the files in LOAD_PATH in name sorted order (note that the sorting criteria is
# with respect to the integer at filename.split('/')[-1].split('_')[1]).
files = [os.path.join(LOAD_PATH, filepath) for filepath in os.listdir(LOAD_PATH)]
files = sorted(files, key=lambda x: int(x.split('/')[-1].split('_')[1]))
# Only keep the files that include the substring "10x10"
files = [file for file in files if "10x10" in file]
print("Number of areas: ", len(files))

# Read activity labels
if '2022' in LOAD_PATH:
    # JV 2022 data
    csv_content = np.load(CSV_PATH, allow_pickle=True)#pickle.load(open(CSV_PATH, "rb"))
    activity_labels = [csv_entry[1] for csv_entry in csv_content]
    
    # Also read timestamps for when inspections were done (if available)
    date_stamps = []
    for csv_entry in csv_content:
        # Also get date stamp if available
        dcim_path = csv_entry[3]
        # The date is always the first 8 digit characters in the dcim_path,
        # so use regex to extract the first 8 digits
        if 'DCIM' in dcim_path:
            date_stamp = re.search(r'\d{8}', dcim_path).group()
        else:
            date_stamp = None
        date_stamps.append(date_stamp)
else:
    # JV 2024 data
    dataset2024 = np.load("../jv-data/2024-dataset-all.pkl", allow_pickle=True)
    activity_labels = [dataset2024[i][1] for i in range(len(dataset2024))]
    date_stamps = [dataset2024[i][3] for i in range(len(dataset2024))]

# At this stage, date_stamps entries expected to be in the format 'yyyymmdd' or if not date is
# in the entry the value is None. So in case entries are datetime objects, convert them to strings.
# So e.g. datetime.date(2024, 10, 2) becomes '20241002'.
for idx, date_stamp in enumerate(date_stamps):
    if date_stamp is not None and isinstance(date_stamp, datetime.date):
        date_stamp = date_stamp.strftime('%Y%m%d')
        date_stamps[idx] = date_stamp

# Setup and load COT prediction model (Pirinen et al. 2024) -- actually not used, but is loaded here
# so it looks similar as for the Sen2 data, to make all randomness identical!
COT_MODEL_LOAD_PATH = ['../cot-model/skogs_models/2023-08-10_11-49-01/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-49-22/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-49-49/model_it_2000000',
                    '../cot-model/skogs_models/2023-08-10_11-50-44/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-51-11/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-51-36/model_it_2000000',
                    '../cot-model/skogs_models/2023-08-10_11-51-49/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-52-02/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-52-24/model_it_2000000',
                    '../cot-model/skogs_models/2023-08-10_11-52-47/model_it_2000000']
models = []
for model_load_path in COT_MODEL_LOAD_PATH:
    model = MLP5(11, 1, apply_relu=True)
    model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
    model.to(DEVICE)
    models.append(model)

# Iterate over all the files
if END_IDX == -1:
    END_IDX = len(files)

# Via the below loop, we want to end-up with an ML-ready dataset.
ml_dataset = []
for file_idx in range(START_IDX, END_IDX):

    # Print progress
    print(f"Processing file {file_idx+1}/{END_IDX}...")

    # Skip certain indices or activities
    if IDXS_TO_USE is not None and file_idx not in IDXS_TO_USE:
        continue
    if ACTIVITIES_TO_USE is not None and activity_labels[file_idx] not in ACTIVITIES_TO_USE:
        continue

    # Read image timeseries, dates and polygon of an area
    im_series, dates = read_sentinel1_nc(files[file_idx], True)
    filename_im = files[file_idx]
    poly = np.load(filename_im.replace('_image.nc', '_polygon.npy').replace('10x10_', ''))

    # Also get label of the activity, if available
    activity_label = activity_labels[file_idx]
    if activity_label in ["Ingen aktivitet", "no activity"]: 
        label = 0
    elif activity_label in ["Aktivitet bete", "grazed"]:
        label = 1
    else:
        print("GOT HEREE")
        print("Unknown activity label: ", activity_label)
        sys.exit()
        label = -1

    # Also get time stamp of the inspection, if available
    date_stamp = date_stamps[file_idx]

    # Convert into date time object; note that the starting point is a string 'yyyymmdd'.
    if date_stamp is not None:
        date_stamp = date_stamp[:4] + '-' + date_stamp[4:6] + '-' + date_stamp[6:]
        date_stamp = datetime.datetime.strptime(date_stamp, '%Y-%m-%d')

    # Based on dates, only keep the images within the specified time range
    idxs_keep = [i for i, date in enumerate(dates) if date >= START_DATE and date < END_DATE]
    im_series = im_series[:, :, :, idxs_keep]
    dates = [dates[i] for i in idxs_keep]
    H, W, C, T = im_series.shape
    im_height_km = H * 10 / 1000  # 10 m resolution is the highest, thus this is the formula used
    im_width_km = W * 10 / 1000

    # Extract square bounding box around the polygon (with a margin)
    min_lon_orig = np.min(poly[:,0])
    max_lon_orig = np.max(poly[:,0])
    min_lat_orig = np.min(poly[:,1])
    max_lat_orig = np.max(poly[:,1])
    min_lat, min_lon, max_lat, max_lon, lat_span_km, lon_span_km = create_square_bounding_box(min_lat_orig, min_lon_orig, max_lat_orig, max_lon_orig, width_km=0)
    lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).km
    lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).km

    # Get polygon aligned with the image
    margin_per_side = (np.max([im_height_km, im_width_km]) - np.max([lon_span_km, lat_span_km])) / 2 / np.max([lon_span_km, lat_span_km])
    poly_aligned = align_poly_with_image(poly, H, W, margin=margin_per_side)
    poly_aligned_int = np.floor(poly_aligned).astype(int)

    # Zoom in on the center of the images in im_series.
    if config.IMG_ZOOMED_SIZE is not None:
        assert config.IMG_ZOOMED_SIZE <= H and config.IMG_ZOOMED_SIZE <= W
        h_start = H//2-config.IMG_ZOOMED_SIZE//2
        w_start = W//2-config.IMG_ZOOMED_SIZE//2
        im_series = im_series[h_start : H//2+config.IMG_ZOOMED_SIZE//2, w_start : W//2+config.IMG_ZOOMED_SIZE//2, :, :]

    # Recompute some things based on the new image size
    H, W, _, _ = im_series.shape
    poly_aligned[:,0] -= w_start
    poly_aligned[:,1] -= h_start
    poly_aligned_int[:,0] -= w_start
    poly_aligned_int[:,1] -= h_start
    # Now that we made im_series smaller, we need to ensure poly_aligned_int is within the new bounds.
    poly_aligned[:,0] = np.clip(poly_aligned[:,0], 0, W)
    poly_aligned[:,1] = np.clip(poly_aligned[:,1], 0, H)
    poly_aligned_int[:,0] = np.clip(poly_aligned_int[:,0], 0, W-1)
    poly_aligned_int[:,1] = np.clip(poly_aligned_int[:,1], 0, H-1)

    # Append im_series and label to the dataset, as well as dates and the polygon mask and xy-coordinate-poly.
    ml_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))

# Shuffle dataset
np.random.shuffle(ml_dataset)
all_polys_raw = [entry[-1] for entry in ml_dataset]

# Check distances between polygons

# First get a square corresponding to each polygon
poly_squares = []
for idx, poly in enumerate(all_polys_raw):
    # Extract square bounding box around the polygon (with a margin). Used later for visualization
    min_lon_orig = np.min(poly[:,0])
    max_lon_orig = np.max(poly[:,0])
    min_lat_orig = np.min(poly[:,1])
    max_lat_orig = np.max(poly[:,1])
    min_lat, min_lon, max_lat, max_lon, lat_span_km, lon_span_km = create_square_bounding_box(min_lat_orig, min_lon_orig, max_lat_orig, max_lon_orig, width_km=0.85)
    lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).km
    lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).km
    poly_squares.append(np.array((min_lat, min_lon, max_lat, max_lon)))

# Now compute mid point distances between all polygon squares
all_dists = np.empty((len(poly_squares), len(poly_squares)))
all_dists[:] = np.nan
for i in range(len(poly_squares)):
    for j in range(len(poly_squares)):

        # Skip if same polygon
        if i == j:
            continue

        # Extract polygon squares
        poly1 = poly_squares[i]
        poly2 = poly_squares[j]
        mid1 = np.array([(poly1[0] + poly1[2]) / 2, (poly1[1] + poly1[3]) / 2])
        mid2 = np.array([(poly2[0] + poly2[2]) / 2, (poly2[1] + poly2[3]) / 2])
        
        # Compute distance dist_ij between centers of the squares
        dist_ij = geopy.distance.distance(mid1, mid2).km
        
        # Add to distance matrix
        all_dists[i, j] = dist_ij

# At this stage, split into training and validation sets. Try to keep class balances similar
# in both sets. Also, based on all_dists, ensure that if a data point lands in the training set,
# then all that are less than 0.85 km away also end up in the training set.
label_counts = np.array([0, 0])
for _, label, _, _, _, _ in ml_dataset:
    if label != -1:
        label_counts[label] += 1
nbr_to_take = np.min(label_counts)
nbr_train = int(nbr_to_take * TRAIN_FRAC)
nbr_val = nbr_to_take - nbr_train
train_dataset = []
val_dataset = []
idx = 0
idxs_used = []
for im_series, label, dates, poly_aligned_int, date_stamp, poly in ml_dataset:
    if idx in idxs_used:
        idx += 1
        continue
    if label == 0:
        label_reduction_0 = 1
        label_reduction_1 = 0
        if label_counts[0] <= nbr_val:
            val_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))
            idxs_used.append(idx)
            # Now also add all those that are less than 0.85 km away to the validation set
            curr_dists = all_dists[idx, :]
            close_idxs = np.argwhere(curr_dists < 0.85).flatten()
            for close_idx in close_idxs:
                if close_idx not in idxs_used:
                    im_series, label, dates, poly_aligned_int, date_stamp, poly = ml_dataset[close_idx]
                    val_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))
                    idxs_used.append(close_idx)
                    if label == 0:
                        label_reduction_0 += 1
                    elif label == 1:
                        label_reduction_1 += 1

        else:
            train_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))
            idxs_used.append(idx)
            # Now also add all those that are less than 0.85 km away to the training set
            curr_dists = all_dists[idx, :]
            close_idxs = np.argwhere(curr_dists < 0.85).flatten()
            for close_idx in close_idxs:
                if close_idx not in idxs_used:
                    im_series, label, dates, poly_aligned_int, date_stamp, poly = ml_dataset[close_idx]
                    train_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))
                    idxs_used.append(close_idx)
                    if label == 0:
                        label_reduction_0 += 1
                    elif label == 1:
                        label_reduction_1 += 1
        label_counts[0] -= label_reduction_0
        label_counts[1] -= label_reduction_1
    elif label == 1:
        label_reduction_0 = 0
        label_reduction_1 = 1
        if label_counts[1] <= nbr_val:
            val_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))
            idxs_used.append(idx)
            # Now also add all those that are less than 0.85 km away to the validation set
            curr_dists = all_dists[idx, :]
            close_idxs = np.argwhere(curr_dists < 0.85).flatten()
            for close_idx in close_idxs:
                if close_idx not in idxs_used:
                    im_series, label, dates, poly_aligned_int, date_stamp, poly = ml_dataset[close_idx]
                    val_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))
                    idxs_used.append(close_idx)
                    if label == 0:
                        label_reduction_0 += 1
                    elif label == 1:
                        label_reduction_1 += 1                        
        else:
            train_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))
            idxs_used.append(idx)
            # Now also add all those that are less than 0.85 km away to the training set
            curr_dists = all_dists[idx, :]
            close_idxs = np.argwhere(curr_dists < 0.85).flatten()
            for close_idx in close_idxs:
                if close_idx not in idxs_used:
                    im_series, label, dates, poly_aligned_int, date_stamp, poly = ml_dataset[close_idx]
                    train_dataset.append((im_series, label, dates, poly_aligned_int, date_stamp, poly))
                    idxs_used.append(close_idx)
                    if label == 0:
                        label_reduction_0 += 1
                    elif label == 1:
                        label_reduction_1 += 1
        label_counts[0] -= label_reduction_0
        label_counts[1] -= label_reduction_1
    idx += 1
assert len(train_dataset) + len(val_dataset) == len(ml_dataset)

# Finally, use pickle to save the dataset to file
# If not 2022 in LOAD_PATH, then add the year 2024 to the save name!
if '2022' not in LOAD_PATH:
    PKL_SAVE_NAME = PKL_SAVE_NAME.replace(".pkl", "_2024.pkl")
with open(PKL_SAVE_NAME.replace(".pkl", "_train.pkl"), "wb") as f:
    pickle.dump(train_dataset, f)
with open(PKL_SAVE_NAME.replace(".pkl", "_val.pkl"), "wb") as f:
    pickle.dump(val_dataset, f)

print("Done!")