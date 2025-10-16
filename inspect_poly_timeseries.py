import os, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import numpy as np
import config
from utils import read_image_and_polygon, align_poly_with_image, create_square_bounding_box, read_sentinel1_nc, mlp_inference
from classes import MLP5
import geopy.distance
import skimage.draw as skdraw

# Global vars
BANDS_ALREADY_CONCATENATED = False  # If True, then process made simpler
LOAD_PATH = "../data_and_polygons_2022"
LOAD_PATH = "../data_and_polygons_2024"
CSV_PATH = '../jv-data/csv_subset_all_2022.npy'  # Path to csv-pkl file with activity labels
DEVICE = 'cpu'
# The below model weights can be directly downloaded from this link:
# https://drive.google.com/drive/mobile/folders/14xTbLHPxaPznemG7ShE0DMC9zJsNU_hr?usp=sharing
# (See also the COT model repo here: https://github.com/aleksispi/ml-cloud-opt-thick)
COT_MODEL_LOAD_PATH = ['../cot-model/skogs_models/2023-08-10_11-49-01/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-49-22/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-49-49/model_it_2000000',
				       '../cot-model/skogs_models/2023-08-10_11-50-44/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-51-11/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-51-36/model_it_2000000',
				       '../cot-model/skogs_models/2023-08-10_11-51-49/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-52-02/model_it_2000000', '../cot-model/skogs_models/2023-08-10_11-52-24/model_it_2000000',
				       '../cot-model/skogs_models/2023-08-10_11-52-47/model_it_2000000']
VISUALIZE_RESULTS = True  # True --> Visualize the results
NEXT_TIMESERIES_WHEN_CLOUD_FREE = True  # True --> Skip to next timeseries when no clouds are detected
POLY_CLOUD_FRAC_THRESH = 0.01  # If fraction of cloudy pixels within polygon is below this, then consider as cloud-free
DISCARD_CLOUDY_POLYS = True  # True --> Discard cloudy polygons (based on POLY_CLOUD_FRAC_THRESH)
START_DATE = '2024-04-01'  # Start date for the timeseries
END_DATE = '2024-10-21'  # End date for the timeseries (exclusive)
START_IDX = 0  # Start index for the timeseries
END_IDX = -1  # End index for the timeseries (exclusive) -- if -1, then use all
IDXS_TO_USE = None  # If set as specific list of indices, then only use those indices
ACTIVITIES_TO_USE = None  # ['Ingen aktivitet', 'Aktivitet bete']  # If set as specific list of activities, then only use those activities -- ['Osäker men aktivitet', 'Ingen aktivitet', 'Aktivitet skörd', 'Aktivitet bete']
USE_ALSO_SEN1 = False  # If True, then also use Sentinel 1 data


# Make some things into lists
if not isinstance(COT_MODEL_LOAD_PATH, list):
    MODEL_LOAD_PATH = [COT_MODEL_LOAD_PATH]
if not isinstance(config.THRESHOLD_THICKNESS_IS_CLOUD, list):
    config.THRESHOLD_THICKNESS_IS_CLOUD = [config.THRESHOLD_THICKNESS_IS_CLOUD]

# List all the files in LOAD_PATH in name sorted order (note that the sorting criteria is
# with respect to the integer at filename.split('/')[-1].split('_')[1]).
files = [os.path.join(LOAD_PATH, filepath) for filepath in os.listdir(LOAD_PATH)]
files = sorted(files, key=lambda x: int(x.split('/')[-1].split('_')[1]))
if not BANDS_ALREADY_CONCATENATED:
    # Only keep the files that include the substring "10x10" and for which the 20x20 and 60x60 counterparts do exist
    files20 = [file for file in files if "20x20" in file]
    files60 = [file for file in files if "60x60" in file]
    files = [file for file in files if "10x10" in file]
    files = [file for file in files if file.replace("10x10", "20x20") in files20 and file.replace("10x10", "60x60") in files60]
else:
    # Let files only be the filenames containing the "image" substring.
    files = [file for file in files if "image" in file]
print("Number of areas: ", len(files))

# Read also the Sentinel 1 data if needed
if USE_ALSO_SEN1:
    load_path_components = LOAD_PATH.split('/')
    last_part = load_path_components[-1]
    last_part_sen1 = last_part.replace("data_", "sen1_data_")
    load_path_sen1 = '/'.join(load_path_components[:-1] + [last_part_sen1])
    files_sen1 = [os.path.join(load_path_sen1, filepath) for filepath in os.listdir(load_path_sen1)]
    files_sen1 = sorted(files_sen1, key=lambda x: int(x.split('/')[-1].split('_')[1]))
    # Remove all poly files since those are duplicates relative to the Sentinel 2 data
    files_sen1 = [file for file in files_sen1 if "image" in file]
    # Note that all are 10x10 files, so no need to filter out any.
    print("Number of areas (Sentinel 1): ", len(files_sen1))

# Read activity labels
if '2022' in LOAD_PATH:
    # JV 2022 data
    csv_content = np.load(CSV_PATH, allow_pickle=True)
    activity_labels = [csv_entry[1] for csv_entry in csv_content]
else:
    # JV 2024 data
    dataset2024 = np.load("../jv-data/2024-dataset-all.pkl", allow_pickle=True)
    activity_labels = [dataset2024[i][1] for i in range(len(dataset2024))]

# Setup and load COT prediction model (Pirinen et al. 2024)
models = []
for model_load_path in COT_MODEL_LOAD_PATH:
    model = MLP5(11, 1, apply_relu=True)
    model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
    model.to(DEVICE)
    models.append(model)

# Iterate over all the files
if END_IDX == -1:
    END_IDX = len(files)

for file_idx in range(START_IDX, END_IDX):

    # Skip certain indices or activities
    if IDXS_TO_USE is not None and file_idx not in IDXS_TO_USE:
        continue
    if ACTIVITIES_TO_USE is not None and activity_labels[file_idx] not in ACTIVITIES_TO_USE:
        continue

    # Read image timeseries and polygon of an area
    if BANDS_ALREADY_CONCATENATED:
        im_series = np.load(files[file_idx], allow_pickle=True)
        poly = np.load(files[file_idx].replace("image", "polygon"))
        dates = np.load(files[file_idx].replace("image", "dates"), allow_pickle=True)
    else:
        im_series, poly, dates = read_image_and_polygon(files[file_idx])

    # Also get the Sentinel 1 data if needed
    if USE_ALSO_SEN1:
        im_series_sen1 = read_sentinel1_nc(files_sen1[file_idx])
    else:
        # Create dummy array for convenience, that has shape (H, W, 2, T) where T is the same as for the Sentinel 2 data
        H, W, C, T = im_series.shape
        im_series_sen1 = np.zeros((H, W, 2, T))

    # NOTE: The date stamps of im_series may differ from those of im_series_sen1.
    # OBS: For now, just make sure both equally long time series are used.
    curr_min_len = min(im_series.shape[-1], im_series_sen1.shape[-1])
    im_series = im_series[:, :, :, :curr_min_len]
    im_series_sen1 = im_series_sen1[:, :, :, :curr_min_len]
    dates = dates[:curr_min_len]

    # Based on dates, only keep the images within the specified time range
    idxs_keep = [i for i, date in enumerate(dates) if date >= START_DATE and date < END_DATE]
    im_series = im_series[:, :, :, idxs_keep]
    im_series_sen1 = im_series_sen1[:, :, :, idxs_keep]
    dates = [dates[i] for i in idxs_keep]
    H, W, C, T = im_series.shape
    im_height_km = H * 10 / 1000  # 10 m resolution is the highest, thus this is the formula used
    im_width_km = W * 10 / 1000

    # Extract square bounding box around the polygon (with a margin). Used later for visualization
    min_lon_orig = np.min(poly[:,0])
    max_lon_orig = np.max(poly[:,0])
    min_lat_orig = np.min(poly[:,1])
    max_lat_orig = np.max(poly[:,1])
    min_lat, min_lon, max_lat, max_lon, lat_span_km, lon_span_km = create_square_bounding_box(min_lat_orig, min_lon_orig, max_lat_orig, max_lon_orig, width_km=0)
    lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).km
    lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).km

    # Load means and stds based on the synthetic SMHI data on which the COT model was trained
    means = torch.Tensor(np.array([0.64984976, 0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])).to(DEVICE)
    stds = torch.Tensor(np.array([0.3596485, 0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])).to(DEVICE)
    means = means[[1,2,3,4,5,6,7,8,9,11,12]]  # due to how COT model works, remove band 1 and band 10
    stds = stds[[1,2,3,4,5,6,7,8,9,11,12]]  # due to how COT model works, remove band 1 and band 10
    # Note: in im_series, band 10 is already not there
    # since it is not present in the Sentinel 2 L2A data
    im_series = im_series[:, :, 1:, :]  # due to how COT model works, remove band 1 here

    # Get polygon aligned with the image
    margin_per_side = (np.max([im_height_km, im_width_km]) - np.max([lon_span_km, lat_span_km])) / 2 / np.max([lon_span_km, lat_span_km])
    poly_aligned = align_poly_with_image(poly, H, W, margin=margin_per_side)
    poly_aligned_int = np.floor(poly_aligned).astype(int)

    # Iterate over the timeseries
    for t in range(T):
        # Perform prediction
        print("Processing image (%d, %d, %d)" % (file_idx, t, T))
        img = (im_series[:, :, :, t] - 1000) / 10000  # -1k and then 10k division, done after Jan 2022
        pred_map, pred_map_binary_list, pred_map_binary_list_thin = mlp_inference(img, means, stds, models, H*W, config.THRESHOLD_THICKNESS_IS_CLOUD,
                                                                                  config.THRESHOLD_THICKNESS_IS_CLOUD, 1, DEVICE)
        pred_map_binary = pred_map_binary_list[0]
        pred_map_binary_thin = pred_map_binary_list_thin[0]
        
        # Check which pixels within the polygon are deemed as cloudy
        # Note that poly[:,0] represents longitude (width direction)
        # and poly[:,1] represents latitude (height direction)
        poly_mask = np.zeros((H, W), dtype=bool)
        rr, cc = skdraw.polygon(H-poly_aligned_int[:,1], poly_aligned_int[:,0])
        rr = np.clip(rr, 0, H-1)
        cc = np.clip(cc, 0, W-1)
        poly_mask[rr, cc] = True
        pred_map_binary_poly = pred_map_binary[poly_mask]
        frac_binary_poly = np.sum(pred_map_binary_poly) / len(pred_map_binary_poly)
        pred_cloudy_poly = frac_binary_poly > POLY_CLOUD_FRAC_THRESH

        # Print fraction of pixels deemed as cloudy, both in total and within the polygon
        frac_binary = np.sum(pred_map_binary) / (H*W)
        pred_cloudy = frac_binary > 0
        print("Fraction of pixels deemed as cloudy (total, polygon):", frac_binary*100, frac_binary_poly*100)

        if DISCARD_CLOUDY_POLYS and pred_cloudy_poly:
            if VISUALIZE_RESULTS:
                print("Polygon is deemed cloudy, continuing to next time step")
            continue

        # Visualize results
        if VISUALIZE_RESULTS:
            fig = plt.figure(figsize=(16, 16))
            H, W, C = img.shape

            # Extract and show RGB image
            rgb_img = img[:, :, [2, 1, 0]] / np.max(img[:, :, [2, 1, 0]])
            ax = fig.add_subplot(2,2,1)
            plt.imshow(rgb_img)
            plt.title(dates[t] + ", %.2f x %.2f km" % (im_height_km, im_width_km) + ", " + activity_labels[file_idx])
            # Also show aligned polygon on top of the image
            plt.plot(poly_aligned[:,0], H-poly_aligned[:,1], color='red', linewidth=2)
            # Set ticks and tick labels
            # Need to update this to display the correct lon-lat coordinates in the ticks
            min_lat, min_lon, max_lat, max_lon, _, _ = create_square_bounding_box(min_lat_orig, min_lon_orig, max_lat_orig, max_lon_orig, width_km=-margin_per_side)
            nbr_x_ticks = len(ax.get_xticks())
            ax.set_xticklabels(np.round(np.linspace(min_lon, max_lon, nbr_x_ticks), 4))
            nbr_y_ticks = len(ax.get_yticks())
            ax.set_yticklabels(np.round(np.linspace(max_lat, min_lat, nbr_y_ticks), 4))
            plt.xticks(rotation=45)
            #plt.grid(color='black', linestyle='-', linewidth=0.5)

            # COT map
            ax = fig.add_subplot(2,2,2)
            plt.title('pred (min, max)=(%.3f, %.3f)' % (np.nanmin(pred_map), np.nanmax(pred_map)))
            pred_map[np.isnan(pred_map)] = 0
            plt.imshow(pred_map, vmin=0, vmax=1, cmap='gray')
            plt.plot(poly_aligned[:,0], H-poly_aligned[:,1], color='red', linewidth=2)
            nbr_x_ticks = len(ax.get_xticks())
            ax.set_xticklabels(np.round(np.linspace(min_lon, max_lon, nbr_x_ticks), 4))
            nbr_y_ticks = len(ax.get_yticks())
            ax.set_yticklabels(np.round(np.linspace(max_lat, min_lat, nbr_y_ticks), 4))
            plt.xticks(rotation=45)

            # Binary cloud map
            ax = fig.add_subplot(2,2,3)
            plt.imshow(0.0 + 2*pred_map_binary + pred_map_binary_thin, vmin=0, vmax=2, cmap='gray')
            if pred_cloudy:
                plt.title('binary pred, cloudy (%.1f prct-tot, %.1f prct-poly)' % (100*frac_binary, 100*frac_binary_poly))
            else:
                plt.title('binary pred, clear (%.1f prct)' % (100*frac_binary))
            plt.plot(poly_aligned[:,0], H-poly_aligned[:,1], color='red', linewidth=2)
            nbr_x_ticks = len(ax.get_xticks())
            ax.set_xticklabels(np.round(np.linspace(min_lon, max_lon, nbr_x_ticks), 4))
            nbr_y_ticks = len(ax.get_yticks())
            ax.set_yticklabels(np.round(np.linspace(max_lat, min_lat, nbr_y_ticks), 4))
            plt.xticks(rotation=45)

            # Also compute the NDVI for the current image and show at (2,2,4).
            # OBS: If we want to visualize Sentinel 1 data, we skip NDVI computation
            # and instead go for the Sentinel 1 data visualization.
            if USE_ALSO_SEN1:
                # Extract and show Sentinel 1 data
                img_sen1 = im_series_sen1[:, :, 1, t]  # 0 or 1
                # For Sentinel-1, do min-max normalization, or do percentile normalization
                if False:
                    img_sen1 = (img_sen1 - np.min(img_sen1)) / (np.max(img_sen1) - np.min(img_sen1))
                else:
                    # Percentile normalization
                    vec_sen1 = img_sen1.flatten()
                    p_low = np.percentile(vec_sen1, 2)
                    p_high = np.percentile(vec_sen1, 98)
                    img_sen1 = (img_sen1 - p_low) / (p_high - p_low)
                    img_sen1 = np.clip(img_sen1, 0, 1)
                ax = fig.add_subplot(2,2,4)
                plt.imshow(img_sen1, vmin=0, vmax=1, cmap='gray')
                plt.plot(poly_aligned[:,0], H-poly_aligned[:,1], color='red', linewidth=2)
                plt.title('Sentinel 1')
            else:
                red = img[:,:,2]
                nir = img[:,:,6]
                ndvi = (nir - red) / (nir + red)
                ax = fig.add_subplot(2,2,4)
                plt.imshow(ndvi, vmin=-1, vmax=1, cmap='RdYlGn')
                plt.plot(poly_aligned[:,0], H-poly_aligned[:,1], color='red', linewidth=2)
                plt.title('NDVI')
            nbr_x_ticks = len(ax.get_xticks())
            ax.set_xticklabels(np.round(np.linspace(min_lon, max_lon, nbr_x_ticks), 4))
            nbr_y_ticks = len(ax.get_yticks())
            ax.set_yticklabels(np.round(np.linspace(max_lat, min_lat, nbr_y_ticks), 4))
            plt.xticks(rotation=45)

            # Save figure
            plt.savefig(f"results_{file_idx}_{t}.png")
            plt.cla()
            plt.clf()
            plt.close('all')
            
            # If this was a cloud-free image and NEXT_TIMESERIES_WHEN_CLOUD_FREE is True, then break
            # i.e. continue to the next timeseries
            if NEXT_TIMESERIES_WHEN_CLOUD_FREE:
                break
                
        # If this was a clear image and NEXT_TIMESERIES_WHEN_CLOUD_FREE is True, then continue
        # processing the next timeseries
        if NEXT_TIMESERIES_WHEN_CLOUD_FREE and not pred_cloudy:
            continue

print("Done processing all images")