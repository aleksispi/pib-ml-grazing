import os, sys
import matplotlib.pyplot as plt
from datetime import datetime
import netCDF4
import cv2
import random
import torch
import torch.nn as nn
from pyproj import Transformer
import datetime
from pprint import pprint
import numpy as np
import config
import pickle
import sklearn.metrics
import geopy.distance
from contextlib import contextmanager
from datetime import datetime, timezone
from scipy.special import expit
from classes import LSTMClassifier, StatCollector


@contextmanager
def time_measurement(msg):
    thrown_exception: Exception = None
    try:
        ts_start = datetime.now(timezone.utc)
        yield  # Execute the block of code inside the 'with' statement
    except Exception as e:
        thrown_exception = e
        raise
    finally:
        ts_end = datetime.now(timezone.utc)
        duration = (ts_end - ts_start).total_seconds()
        if thrown_exception:
            print(
                f"{msg} raised exception {type(thrown_exception)}"
                f" {thrown_exception}after {duration}s"
            )
        else:
            print(f"{msg} took {duration}s")

def read_sentinel1_nc(filename, return_date=False):
    # Read a Sentinel-1 netCDF file and return the data as a numpy array
    # The data is in dB, and the dimensions are (H, W, C, T), where
    # H is the height, W is the width, C is the number of channels, and
    # T is the number of timestamps.
    nc = netCDF4.Dataset(filename)
    bands = []
    for variable in nc.variables.keys():
        if 'V' in variable or 'v' in variable:
            bands.append(nc.variables[variable][:, :, :][np.newaxis])
    im = np.concatenate(bands, axis=0)
    im = np.transpose(im, (2, 3, 0, 1))
    
    if return_date:
        dates = netCDF4.num2date(nc.variables['t'], nc.variables['t'].units)
        dates = [date.strftime('%Y-%m-%d') for date in dates]
        return im, dates

    return im

def read_image_and_polygon(filename_im, band_upsampling_approach='bilinear', return_poly=True):
    # Read image, polygon and dates of an area
    # For band_upsampling_approach, the possibilities are:
    # - 'nearest': Use cv2.resize with interpolation=cv2.INTER_NEAREST
    # - 'bilinear': Use cv2.resize with interpolation=cv2.INTER_LINEAR
    # - 'bicubic': Use cv2.resize with interpolation=cv2.INTER_CUBIC
    
    # Update filename_im to match the fact that there are 10x10, 20x20 and 60x60
    # counterparts of the same area, and that the filename is the same for all
    filenames_im = [filename_im, filename_im.replace('10x10', '20x20'), filename_im.replace('10x10', '60x60')]

    # Process similarly for the different band resolutions
    ims_all_bands = []
    for i, filename_im in enumerate(filenames_im):
        im_nc = netCDF4.Dataset(filename_im)
        # im_nc above is a netCDF4.Dataset object, now we want to extract the data from it
        # and convert it to a numpy array
        # Create an np array that has dimension H x W x C x T, where C is the number of bands,
        # and T is the number of timestamps. We want to do this by fetching all variables in
        # the .nc file that contain the word "B" or "b".
        bands = []
        for variable in im_nc.variables.keys():
            if 'B' in variable or 'b' in variable:
                bands.append(im_nc.variables[variable][:, :, :][np.newaxis])

        # Each of the C elements in bands have shape (1, T, H, W).
        # Now from this use np to create im that has shape (H, W, C, T).
        im = np.concatenate(bands, axis=0)
        im = np.transpose(im, (2, 3, 0, 1))
        
        # Append to ims_all_bands
        ims_all_bands.append(im)

    # Next, we want to upsample the 20x20 and 60x60 bands to 10x10,
    # based on band_upsampling_approach. Use cv2 for this.
    H_upsampled, W_upsampled, _, T = ims_all_bands[0].shape
    size_upsampled = max(H_upsampled, W_upsampled)  # We want to upsample to a square image
    for i in range(len(ims_all_bands)):
        # Use cv2 to resize the image to the size of the 10x10 image
        curr_img = ims_all_bands[i]
        H, W, C, T = curr_img.shape
        # curr_img has shape H x W x C x T; iterate over the last dimension
        curr_img_upsampled = np.zeros((size_upsampled, size_upsampled, C, T))
        for t in range(T):
            if band_upsampling_approach == 'nearest':
                curr_img_upsampled[:, :, :, t] = cv2.resize(curr_img[:, :, :, t], (size_upsampled, size_upsampled), interpolation=cv2.INTER_NEAREST)
            elif band_upsampling_approach == 'bilinear':
                curr_img_upsampled[:, :, :, t] = cv2.resize(curr_img[:, :, :, t], (size_upsampled, size_upsampled), interpolation=cv2.INTER_LINEAR)
            elif band_upsampling_approach == 'bicubic':
                curr_img_upsampled[:, :, :, t] = cv2.resize(curr_img[:, :, :, t], (size_upsampled, size_upsampled), interpolation=cv2.INTER_CUBIC)
            else:
                raise ValueError(f"Unknown band_upsampling_approach: {band_upsampling_approach}")
        ims_all_bands[i] = curr_img_upsampled

    # We now want to create im, which contains ALL bands (that currently reside in ims_all_bands)
    # and the order of the bands in im should match the variable bands above.
    ims_10x10 = ims_all_bands[0]
    ims_20x20 = ims_all_bands[1]
    ims_60x60 = ims_all_bands[2]
    bands = {"b01": 60, "b02": 10, "b03": 10, "b04": 10, "b05": 20, "b06": 20, "b07": 20, "b08": 10, "b8a": 20, "b09": 60, "b11": 20, "b12": 20}
    bands_10x10 = [key for key, value in bands.items() if value == 10]
    bands_20x20 = [key for key, value in bands.items() if value == 20]
    bands_60x60 = [key for key, value in bands.items() if value == 60]
    im = []
    for band_name, _ in bands.items():
        if band_name in bands_10x10:
            im.append(ims_10x10[:, :, bands_10x10.index(band_name), :][:, :, np.newaxis, :])
        elif band_name in bands_20x20:
            im.append(ims_20x20[:, :, bands_20x20.index(band_name), :][:, :, np.newaxis, :])
        elif band_name in bands_60x60:
            im.append(ims_60x60[:, :, bands_60x60.index(band_name), :][:, :, np.newaxis, :])
        else:
            raise ValueError(f"Unknown band_name: {band_name}")
    im = np.concatenate(im, axis=2)

    # Load the polygon
    filename_im = filenames_im[0]
    if return_poly:
        poly = np.load(filename_im.replace('_image.nc', '_polygon.npy').replace('10x10_', ''))
    else:
        poly = None

    # Get the timestamps from im_nc in date format i.e. YYYY-MM-DD
    dates = netCDF4.num2date(im_nc.variables['t'], im_nc.variables['t'].units)
    dates = [date.strftime('%Y-%m-%d') for date in dates]

    # Return the image, polygon and the dates
    return im, poly, dates    

def align_poly_with_image(poly, im_height, im_width, margin=0.0, clip_if_poly_outside=True):
    # In the below im_width and im_height correspond to a slightly larger image than
    # the minimum bounding box of the polygon (because of the margin). Thus
    # we need to adjust the polygon by moving it slightly to the right and down,
    # and when scaling with the image width and height, we need to scale it down.
    
    # Based on min_lat, min_lon, max_lat, max_lon, calculate the distance in km
    # of the latitudal span and the longitudinal span
    min_lon = np.min(poly[:,0])
    max_lon = np.max(poly[:,0])
    min_lat = np.min(poly[:,1])
    max_lat = np.max(poly[:,1])
    lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).kilometers
    lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).kilometers
    if lat_span_km > lon_span_km:
        # latitude corresponds to the y-axis, longitude to the x-axis
        # thus in this case we are considering an image which is taller than it is wide
        im_width_tight = im_width / (1 + 2 * margin) * lon_span_km / lat_span_km
        im_height_tight = im_height / (1 + 2 * margin)
    else:
        im_height_tight = im_height / (1 + 2 * margin) * lat_span_km / lon_span_km
        im_width_tight = im_width / (1 + 2 * margin)

    # Begin aligning the polygon with the image
    poly_aligned = poly.copy()
    poly_aligned[:,0] = poly_aligned[:,0] - np.min(poly_aligned[:,0])
    poly_aligned[:,1] = poly_aligned[:,1] - np.min(poly_aligned[:,1])
    poly_aligned[:,0] = poly_aligned[:,0] / np.max(poly_aligned[:,0]) * im_width_tight  # poly[:,0] is the longitude (x-axis)
    poly_aligned[:,1] = poly_aligned[:,1] / np.max(poly_aligned[:,1]) * im_height_tight  # poly[:,1] is the latitude (y-axis)
    
    # Now move poly_aligned so that it is centered in the image
    poly_aligned[:,0] = poly_aligned[:,0] + (im_width - im_width_tight) / 2
    poly_aligned[:,1] = poly_aligned[:,1] + (im_height - im_height_tight) / 2
    
    # Assert some dimensions etc
    assert poly_aligned.shape[1] == 2
    if clip_if_poly_outside:
        poly_aligned[:,0] = np.clip(poly_aligned[:,0], 0, im_width)
        poly_aligned[:,1] = np.clip(poly_aligned[:,1], 0, im_height)

    # Return the aligned polygon
    return poly_aligned

def create_square_bounding_box(min_lat, min_lon, max_lat, max_lon, width_km=-0.25):
    """
    Creates a square bounding box around a center point.

    Args:
        min_lat (float): Smallest endpoint latitude.
        min_lon (float): Smallest endpoint longitude.
        max_lat (float): Largest endpoint latitude.
        max_lon (float): Largest endpoint longitude.
        width_km (float): Width of the square bounding box in kilometers.
                          If non-positive, then interpret it instead as a relative
                          margin on each side around the center point (adaptively set
                          based on original min_lat, min_lon, max_lat, max_lon).

    Returns:
        tuple: (min_lat, min_lon, max_lat, max_lon) representing the square bounding box.
    """

    if width_km <= 0:
        # Based on min_lat, min_lon, max_lat, max_lon, calculate the distance in km
        # of the latitudal span and the longitudinal span
        lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).kilometers
        lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).kilometers

        # Since we want a square bounding box, we need to set the width_km to the
        # maximum of the two spans, and also add the relative margin
        margin = -width_km * 2  # Times 2 since it is a margin on each side
        width_km = max(lat_span_km, lon_span_km)
        width_km += margin * width_km

    # Calculate the distance in kilometers for the given width
    half_width_km = width_km / 2

    # Calculate the coordinates of the corners of the square bounding box
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    center_point = geopy.Point(center_lat, center_lon)
    north_point = geopy.distance.distance(kilometers=half_width_km).destination(center_point, 0)
    south_point = geopy.distance.distance(kilometers=half_width_km).destination(center_point, 180)
    east_point = geopy.distance.distance(kilometers=half_width_km).destination(center_point, 90)
    west_point = geopy.distance.distance(kilometers=half_width_km).destination(center_point, 270)

    # Extract latitude and longitude values for square bounding box
    min_lat = south_point.latitude
    min_lon = west_point.longitude
    max_lat = north_point.latitude
    max_lon = east_point.longitude

    # Calculate the distance in km of the latitudal span and the longitudinal span
    lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).kilometers
    lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).kilometers
    try:
        assert np.abs(lat_span_km - width_km) < 1e-2 and np.abs(lon_span_km - width_km) < 1e-2
    except:
        print(f"lat_span_km: {lat_span_km}, lon_span_km: {lon_span_km}, width_km: {width_km}")
        raise ValueError("The latitudal and longitudinal spans do not match the width_km")

    # Return the square bounding box
    return min_lat, min_lon, max_lat, max_lon, lat_span_km, lon_span_km

def _mlp_post_filter(pred_map_binary_list, pred_map_binary_thin_list, pred_map, thresh_thin_cloud, post_filt_sz):
	if post_filt_sz == 1:
		return pred_map_binary_list, pred_map_binary_thin_list
	H, W = pred_map.shape
	for list_idx, pred_map_binary in enumerate(pred_map_binary_list):
		tmp_map = np.zeros_like(pred_map)
		tmp_map_thin = np.zeros_like(pred_map)
		count_map = np.zeros_like(pred_map)
		for i_start in range(post_filt_sz):
			for j_start in range(post_filt_sz):
				for i in range(i_start, H // post_filt_sz):
					for j in range(j_start, W // post_filt_sz):
						count_map[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
						curr_patch = pred_map_binary[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz]
						curr_patch_thin = pred_map_binary_thin_list[min(list_idx, len(thresh_thin_cloud) - 1)][i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz]
						if np.count_nonzero(curr_patch) >= np.prod(curr_patch.shape) // 2:
							tmp_map[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
						if np.count_nonzero(curr_patch_thin) >= np.prod(curr_patch_thin.shape) // 2:
							tmp_map_thin[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
		tmp_map[count_map == 0] = 0
		count_map[count_map == 0] = 1
		tmp_map /= count_map
		assert np.min(tmp_map) >= 0 and np.max(tmp_map) <= 1
		pred_map_binary = tmp_map >= 0.50
		pred_map_binary_list[list_idx] = pred_map_binary

		tmp_map_thin[count_map == 0] = 0
		tmp_map_thin /= count_map
		assert np.min(tmp_map_thin) >= 0 and np.max(tmp_map_thin) <= 1
		pred_map_binary_thin = tmp_map_thin >= 0.50
		pred_map_binary_thin_list[min(list_idx, len(thresh_thin_cloud) - 1)] = pred_map_binary_thin

		# 'Aliasing effect' after this filtering can cause BOTH thin and regular cloud to be active at the same time -- give prevalence to regular
		pred_map_binary_thin_list[0][pred_map_binary_list[0]] = 0

	return pred_map_binary_list, pred_map_binary_thin_list

# Setup MLP-computation function
def mlp_inference(img, means, stds, models, batch_size, thresh_cloud, thresh_thin_cloud, post_filt_sz, device='cpu', predict_also_cloud_binary=False):
	H, W, input_dim = img.shape
	img_torch = torch.reshape((torch.Tensor(img).to(device) - means) / stds, [H * W, input_dim])
	pred_map_tot = 0.0
	pred_map_binary_tot = 0.0
	for model in models:
		pred_map = np.zeros(H * W)
		pred_map_binary = np.zeros(H * W)
		for i in range(0, H * W, batch_size):
			curr_pred = model(img_torch[i : i + batch_size, :])
			pred_map[i : i + batch_size] = curr_pred[:, 0].cpu().detach().numpy()
			if predict_also_cloud_binary:
				pred_map_binary[i : i + batch_size] = curr_pred[:, 1].cpu().detach().numpy()
		pred_map = np.reshape(pred_map, [H, W])
		if predict_also_cloud_binary:
			pred_map_binary = np.reshape(expit(pred_map_binary), [H, W]) >= 0.5
		else:
			pred_map_binary = np.zeros_like(pred_map)#pred_map >= thresh_cloud[-1] <<--- overwritten anyway
			
		# Average model predictions
		pred_map_tot += pred_map / len(models)
		pred_map_binary_tot += pred_map_binary.astype(float) / len(models)
		
	# Return final predictions
	pred_map = pred_map_tot
	if predict_also_cloud_binary:
		pred_map_binary = pred_map_binary_tot >= 0.5
	else:
		pred_map_binary_list = []
		pred_map_binary_thin_list = []
		for thresh in thresh_cloud:
			pred_map_binary_list.append(pred_map_tot >= thresh)
		for thresh in thresh_thin_cloud:
			# Below: A thin cloud is a thin cloud only if it is above the thin thresh AND below the regular cloud thresh
			pred_map_binary_thin_list.append(np.logical_and(pred_map_tot >= thresh, pred_map_tot < thresh_cloud[0]))

	# Potentially do post-processing on the cloud/not cloud (binary)
	# prediction, so that it becomes more spatially coherent
	pred_map_binary_list, pred_map_binary_thin_list = _mlp_post_filter(pred_map_binary_list, pred_map_binary_thin_list, pred_map, thresh_thin_cloud, post_filt_sz)

	# Return
	return pred_map, pred_map_binary_list, pred_map_binary_thin_list

def load_dataset(config):
    # Load Sen2 dataset
    dataset_train = pickle.load(open(os.path.join(config.DATASET_PKL_BASE + '_train.pkl'), 'rb'))
    dataset_val = pickle.load(open(os.path.join(config.DATASET_PKL_BASE + '_val.pkl'), 'rb'))
    dataset_train_2024 = pickle.load(open(os.path.join(config.DATASET_PKL_BASE + '_2024_train.pkl'), 'rb'))
    dataset_val_2024 = pickle.load(open(os.path.join(config.DATASET_PKL_BASE + '_2024_val.pkl'), 'rb'))
    dataset_train += dataset_train_2024
    dataset_val += dataset_val_2024
    
    # Optionally concatenate the Sentinel-1 dataset
    if config.CONCAT_SEN1:
        
        # Load Sen1 dataset
        
        dataset_train_sen1 = pickle.load(open(os.path.join(config.DATASET_PKL_BASE + '_sen1_train.pkl'), 'rb'))
        dataset_val_sen1 = pickle.load(open(os.path.join(config.DATASET_PKL_BASE + '_sen1_val.pkl'), 'rb'))
        dataset_train_2024_sen1 = pickle.load(open(os.path.join(config.DATASET_PKL_BASE + '_sen1_2024_train.pkl'), 'rb'))
        dataset_val_2024_sen1 = pickle.load(open(os.path.join(config.DATASET_PKL_BASE + '_sen1_2024_val.pkl'), 'rb'))
        dataset_train_sen1 += dataset_train_2024_sen1
        dataset_val_sen1 += dataset_val_2024_sen1
            
        # Do some sanity checks between sen1 and sen2 datasets
        assert len(dataset_train) == len(dataset_train_sen1)
        assert len(dataset_val) == len(dataset_val_sen1)
        for i in range(len(dataset_train)):
            poly = dataset_train[i][-1]
            poly_sen1 = dataset_train_sen1[i][-1]
            assert poly.shape == poly_sen1.shape and np.allclose(poly, poly_sen1)
        for i in range(len(dataset_val)):
            poly = dataset_val[i][-1]
            poly_sen1 = dataset_val_sen1[i][-1]
            assert poly.shape == poly_sen1.shape and np.allclose(poly, poly_sen1)

        # Ensure that the each element is a tuple with 8 elements
        # ((im_series, label, dates, poly_mask, poly_aligned_int, pred_maps_binary, date_stamp, poly))
        for i in range(len(dataset_train_sen1)):
            dataset_train_sen1[i] = list(dataset_train_sen1[i])
            if len(dataset_train_sen1[i]) < 8:
                # In this case, due to Sentinel-1, poly_mask and pred_maps_binary are missing.
                # We add dummy values at these positions of the tuple.
                curr_tuple = dataset_train_sen1[i]
                dummy_poly_mask = np.zeros((curr_tuple[0].shape[0], curr_tuple[0].shape[1]))
                curr_tuple.insert(3, dummy_poly_mask)
                dummy_pred_maps_binary = np.zeros((curr_tuple[0].shape[0], curr_tuple[0].shape[1], curr_tuple[0].shape[3]))
                curr_tuple.insert(5, dummy_pred_maps_binary)
                dataset_train_sen1[i] = tuple(curr_tuple)
        for i in range(len(dataset_val_sen1)):
            dataset_val_sen1[i] = list(dataset_val_sen1[i])
            if len(dataset_val_sen1[i]) < 8:
                # In this case, due to Sentinel-1, poly_mask and pred_maps_binary are missing.
                # We add None values at these positions of the tuple.
                curr_tuple = dataset_val_sen1[i]
                dummy_poly_mask = np.zeros((curr_tuple[0].shape[0], curr_tuple[0].shape[1]))
                curr_tuple.insert(3, dummy_poly_mask)
                dummy_pred_maps_binary = np.zeros((curr_tuple[0].shape[0], curr_tuple[0].shape[1], curr_tuple[0].shape[3]))
                curr_tuple.insert(5, dummy_pred_maps_binary)
                dataset_val_sen1[i] = tuple(curr_tuple)

        # Now it is time to concatenate the Sentinel-1 data to the Sentinel-2 data
        # We do this simply by appending full Sentinel-1 raw data + dates
        # to the input image series of the Sentinel-2 data.
        for i in range(len(dataset_train)):
            dataset_train[i] = list(dataset_train[i])
            dataset_train[i].append((dataset_train_sen1[i][0], dataset_train_sen1[i][2]))
            dataset_train[i] = tuple(dataset_train[i])
        for i in range(len(dataset_val)):
            dataset_val[i] = list(dataset_val[i])
            dataset_val[i].append((dataset_val_sen1[i][0], dataset_val_sen1[i][2]))
            dataset_val[i] = tuple(dataset_val[i])
    else:
        # For consistency, add dummy tuple to each element in the dataset
        for i in range(len(dataset_train)):
            dataset_train[i] = list(dataset_train[i])
            dataset_train[i].append((np.zeros((dataset_train[i][0].shape[0], dataset_train[i][0].shape[1],
                                               dataset_train[i][0].shape[2], dataset_train[i][0].shape[3])), []))
            dataset_train[i] = tuple(dataset_train[i])
        for i in range(len(dataset_val)):
            dataset_val[i] = list(dataset_val[i])
            dataset_val[i].append((np.zeros((dataset_val[i][0].shape[0], dataset_val[i][0].shape[1],
                                             dataset_train[i][0].shape[2], dataset_val[i][0].shape[3])), []))
            dataset_val[i] = tuple(dataset_val[i])

    # In the below, we ensure that the train and val sets have no spatial overlap

    # First get a square corresponding to each polygon
    poly_squares_train = []
    for idx in range(len(dataset_train)):
        poly = dataset_train[idx][7]
        # Extract square bounding box around the polygon (with a margin).
        min_lon_orig = np.min(poly[:,0])
        max_lon_orig = np.max(poly[:,0])
        min_lat_orig = np.min(poly[:,1])
        max_lat_orig = np.max(poly[:,1])
        min_lat, min_lon, max_lat, max_lon, _, _ = create_square_bounding_box(min_lat_orig, min_lon_orig, max_lat_orig, max_lon_orig, width_km=0.85)
        poly_squares_train.append(np.array((min_lat, min_lon, max_lat, max_lon)))
    poly_squares_val = []
    for idx in range(len(dataset_val)):
        poly = dataset_val[idx][7]
        # Extract square bounding box around the polygon (with a margin).
        min_lon_orig = np.min(poly[:,0])
        max_lon_orig = np.max(poly[:,0])
        min_lat_orig = np.min(poly[:,1])
        max_lat_orig = np.max(poly[:,1])
        min_lat, min_lon, max_lat, max_lon, _, _ = create_square_bounding_box(min_lat_orig, min_lon_orig, max_lat_orig, max_lon_orig, width_km=0.85)
        poly_squares_val.append(np.array((min_lat, min_lon, max_lat, max_lon)))

    # Now compute mid point distances between all polygon squares in the respective splits
    all_dists = np.empty((len(poly_squares_train), len(poly_squares_val)))
    all_dists[:] = np.nan
    for i in range(len(poly_squares_train)):
        for j in range(len(poly_squares_val)):

            # Extract polygon squares
            poly1 = poly_squares_train[i]
            poly2 = poly_squares_val[j]
            mid1 = np.array([(poly1[0] + poly1[2]) / 2, (poly1[1] + poly1[3]) / 2])
            mid2 = np.array([(poly2[0] + poly2[2]) / 2, (poly2[1] + poly2[3]) / 2])

            # Compute distance dist_ij between centers of the squares
            dist_ij = geopy.distance.distance(mid1, mid2).km

            # Add to distance matrix
            all_dists[i, j] = dist_ij

    # Get indices of validation points that are less than 0.85 km from some training point.
    vals_too_close = np.any(all_dists < 0.85, axis=0)

    # Finally, remove those validation points that are too close to some training point.
    dataset_val_new = []
    for i in range(len(dataset_val)):
        if not vals_too_close[i]:
            dataset_val_new.append(dataset_val[i])
    dataset_val = dataset_val_new
    print("nbr_train and nbr_val", len(dataset_train), len(dataset_val))

    # Go over all elements in dataset_train and then exlude everything that is not within the date range
    # specified by config.START_DATE and config.END_DATE. Each element of dataset_train is a tuple, so we must convert them
    # to lists first.
    #
    # OBS: Since the dataset might contain data from different years, the dates will only care about days and
    # moths, NOT years. I.e. we care about seasonality but not the actual year.
    start_date = datetime.strptime(config.START_DATE, '%Y-%m-%d')
    end_date = datetime.strptime(config.END_DATE, '%Y-%m-%d')

    # Adjust start_date and end_date to ignore the year
    start_date = (start_date.month, start_date.day)
    end_date = (end_date.month, end_date.day)

    # Function to compare month and day only
    def is_within_date_range(date_str, start, end):
        date = datetime.strptime(date_str, '%Y-%m-%d')
        date_tuple = (date.month, date.day)
        if start <= end:
            return start <= date_tuple < end
        else:
            # Handle ranges that span across the year boundary (e.g., Dec 20 to Jan 10)
            return date_tuple >= start or date_tuple < end

    # Process dataset_train
    for i in range(len(dataset_train)):
        dataset_train[i] = list(dataset_train[i])
    for i in range(len(dataset_train)):
        dates = dataset_train[i][2]
        idxs_to_keep = []
        for j, date in enumerate(dates):
            if is_within_date_range(date, start_date, end_date):
                idxs_to_keep.append(j)
        dataset_train[i][0] = dataset_train[i][0][:, :, :, idxs_to_keep]
        dataset_train[i][2] = [dates[j] for j in idxs_to_keep]
        dataset_train[i][5] = dataset_train[i][5][:, :, idxs_to_keep]

    # Process dataset_val
    for i in range(len(dataset_val)):
        dataset_val[i] = list(dataset_val[i])
    for i in range(len(dataset_val)):
        dates = dataset_val[i][2]
        idxs_to_keep = []
        for j, date in enumerate(dates):
            if is_within_date_range(date, start_date, end_date):
                idxs_to_keep.append(j)
        dataset_val[i][0] = dataset_val[i][0][:, :, :, idxs_to_keep]
        dataset_val[i][2] = [dates[j] for j in idxs_to_keep]
        dataset_val[i][5] = dataset_val[i][5][:, :, idxs_to_keep]
    
    # TODO: THE BELOW IS A RELIC FROM WHEN STILL USING POS ENC AS OPTION -- SHOULD BE REMOVED
    # Append dummy-zeros of same shape as above to each element in the dataset.
    # They are not used anywhere.
    for i in range(len(dataset_train)):
        dataset_train[i].append(np.zeros((len(dataset_train[i][2]), 512)))
    for i in range(len(dataset_val)):
        dataset_val[i].append(np.zeros((len(dataset_val[i][2]), 512)))

    # Return the datasets
    return dataset_train, dataset_val

def compute_dataset_means_stds(dataset, nbr_channels_sen1):
    means = np.zeros(12)
    stds = np.zeros(12)
    means_sen1 = np.zeros(nbr_channels_sen1)
    stds_sen1 = np.zeros(nbr_channels_sen1)
    for i in range(len(dataset)):
        im_series = dataset[i][0]
        means += np.mean(im_series, axis=(0, 1, 3))
        stds += np.std(im_series, axis=(0, 1, 3))
        im_series_sen1 = dataset[i][8][0]
        means_sen1 += np.mean(im_series_sen1, axis=(0, 1, 3))
        stds_sen1 += np.std(im_series_sen1, axis=(0, 1, 3))
    means /= len(dataset)
    stds /= len(dataset)
    means_sen1 /= len(dataset)
    stds_sen1 /= len(dataset)
    means = torch.Tensor(means[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]).to(config.DEVICE)
    stds = torch.Tensor(stds[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]).to(config.DEVICE)
    means_sen1 = torch.Tensor(means_sen1[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]).to(config.DEVICE)
    stds_sen1 = torch.Tensor(stds_sen1[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]).to(config.DEVICE)
    return means, stds, means_sen1, stds_sen1

def create_batch(dataset, i, means, stds, means_sen1, stds_sen1, band_idxs_use, apply_data_aug=False, config_in=None):
    if config_in is not None:
         config = config_in
    # Setup the batch
    im_series_batch = []
    label_batch = []
    poly_batch = []
    cloud_batch = []
    poly_for_viz_batch = []
    dates_batch = []
    sen1_series_batch = []
    sen1_dates_batch = []
    nbr_data_points = len(dataset)
    current_batch_size = min(i + config.BATCH_SIZE, nbr_data_points) - i
    for j in range(i, i + current_batch_size):
        curr_im_series = torch.Tensor(np.transpose(dataset[j][0], (3, 2, 0, 1))).to(config.DEVICE)
        curr_label = dataset[j][1]
        curr_poly = torch.Tensor(dataset[j][3]).to(config.DEVICE)
        curr_poly_for_viz = np.copy(dataset[j][4])
        curr_cloud_mask = torch.Tensor(np.transpose(dataset[j][5], (2, 0, 1))).to(config.DEVICE)
        curr_dates = dataset[j][2]
        curr_sen1_series = torch.Tensor(np.transpose(dataset[j][8][0], (3, 2, 0, 1))).to(config.DEVICE)
        curr_sen1_dates = dataset[j][8][1]

        # Apply some random horizontal flip, vertical flip.
        if apply_data_aug:
            _, _, H, W = curr_im_series.shape
            if np.random.rand() > 0.5:
                # Vertical flip
                curr_im_series = torch.flip(curr_im_series, [2])
                curr_poly = torch.flip(curr_poly, [0])
                curr_cloud_mask = torch.flip(curr_cloud_mask, [1])
                # When flipping curr_poly_viz, note that [:, 0]
                # corresponds to the x-axis and [:, 1] to the y-axis.
                curr_poly_for_viz[:, 1] = H - curr_poly_for_viz[:, 1]
                curr_sen1_series = torch.flip(curr_sen1_series, [2])
            if np.random.rand() > 0.5:
                # Horizontal flip
                curr_im_series = torch.flip(curr_im_series, [3])
                curr_poly = torch.flip(curr_poly, [1])
                curr_cloud_mask = torch.flip(curr_cloud_mask, [2])
                curr_poly_for_viz[:, 0] = W - curr_poly_for_viz[:, 0]
                curr_sen1_series = torch.flip(curr_sen1_series, [3])

            # Apply some random cropping
            H_frac = 4 / 5
            W_frac = 4 / 5
            new_H = random.randint(int(H_frac * H), H)
            new_W = random.randint(int(W_frac * W), W)
            x = random.randint(0, H - new_H)
            y = random.randint(0, W - new_W)
            curr_im_series_orig = torch.clone(curr_im_series)
            curr_im_series = curr_im_series[:, :, x:x+new_H, y:y+new_W]
            curr_im_series_sen1_orig = torch.clone(curr_sen1_series)
            curr_sen1_series = curr_sen1_series[:, :, x:x+new_H, y:y+new_W]
            curr_cloud_mask_orig = torch.clone(curr_cloud_mask)
            curr_cloud_mask = curr_cloud_mask[:, x:x+new_H, y:y+new_W]
            curr_poly_orig = torch.clone(curr_poly)
            new_poly = curr_poly[x:x+new_H, y:y+new_W]
            # If too much of actual polygon ends up outside the crop, then use original-sized input
            if torch.sum(new_poly) < 0.5 * torch.sum(curr_poly_orig) * H * W / (new_H * new_W):
                curr_im_series = curr_im_series_orig
                curr_sen1_series = curr_im_series_sen1_orig
                curr_cloud_mask = curr_cloud_mask_orig
                curr_poly = curr_poly_orig
            else:
                curr_im_series = nn.functional.interpolate(curr_im_series, (H, W), mode='bilinear', align_corners=False)
                curr_sen1_series = nn.functional.interpolate(curr_sen1_series, (H, W), mode='bilinear', align_corners=False)
                curr_cloud_mask = nn.functional.interpolate(curr_cloud_mask.unsqueeze(0).float(), (H, W), mode='nearest').squeeze(0).squeeze(0)
                curr_poly = nn.functional.interpolate(new_poly.unsqueeze(0).unsqueeze(0).float(), (H, W), mode='nearest').squeeze(0).squeeze(0)
                # Also update curr_poly_for_viz
                curr_poly_for_viz[:, 0] = (curr_poly_for_viz[:, 0] - y) * W / new_W
                curr_poly_for_viz[:, 1] = H - curr_poly_for_viz[:, 1]
                curr_poly_for_viz[:, 1] = (curr_poly_for_viz[:, 1] - x) * H / new_H
                curr_poly_for_viz[:, 1] = H - curr_poly_for_viz[:, 1]

            # Independently drop out timesteps in the input tensor
            if np.random.rand() >= 0.5:  # Only do this half of the time
                idxs_to_keep = np.random.rand(curr_im_series.shape[0]) >= config.TIMESTEP_DROPOUT_PROB
                while np.sum(idxs_to_keep) <= 3:  # Make sure we keep at least 4 timesteps
                    idxs_to_keep = np.random.rand(curr_im_series.shape[0]) >= config.TIMESTEP_DROPOUT_PROB
                curr_im_series = curr_im_series[idxs_to_keep, :, :, :]
                curr_cloud_mask = curr_cloud_mask[idxs_to_keep, :, :]
                # Also do it for sen1 data (note that the number of timesteps can be different!)
                idxs_to_keep_sen1 = np.random.rand(curr_sen1_series.shape[0]) >= config.TIMESTEP_DROPOUT_PROB
                while np.sum(idxs_to_keep_sen1) <= 3:  # Make sure we keep at least 4 timesteps
                    idxs_to_keep_sen1 = np.random.rand(curr_sen1_series.shape[0]) >= config.TIMESTEP_DROPOUT_PROB
                curr_sen1_series = curr_sen1_series[idxs_to_keep_sen1, :, :, :]

        # Normalize the input tensor
        curr_im_series = (curr_im_series - means) / stds
        curr_sen1_series = (curr_sen1_series - means_sen1) / stds_sen1

        # Optionally append the cloud mask to the input tensor
        if config.CLOUD_MASK_INPUT:
            tmp = curr_cloud_mask.view(1, curr_cloud_mask.shape[0], 1, curr_cloud_mask.shape[1], curr_cloud_mask.shape[2])
            curr_im_series = torch.cat((curr_im_series, tmp), dim=2)
            # TODO: SHOULD THE CLOUD MASK BE ADDED TO THE SEN1 DATA AS WELL, SO THAT THIS PART OF
            # THE ML MODEL UNDERSTANDS THAT A CERTAIN PART OF SEN2 IS CLOUDED?

        # Optionally append the polygon mask to the input tensor
        if config.POLY_MASK_INPUT:
            # Ensure that the poly_tensor has the same number of timesteps as the input_tensor
            tmp = curr_poly.repeat(1, curr_im_series.shape[1], 1, 1, 1)
            # Now append the poly_tensor to the input_tensor
            curr_im_series = torch.cat((curr_im_series, tmp), dim=2)
            tmp = curr_poly.repeat(1, curr_sen1_series.shape[1], 1, 1, 1)
            curr_sen1_series = torch.cat((curr_sen1_series, tmp), dim=2)

        # Poly is a binary mask, where 1 indicates the polygon and 0 indicates non-polygon and has shape H x W.
        if config.MASK_OUT_NONPOLY:
            curr_im_series *= curr_poly
            curr_sen1_series *= curr_poly

        # Append to the batch
        im_series_batch.append(curr_im_series.squeeze(0)[:, band_idxs_use, :, :])
        label_batch.append(curr_label)
        poly_batch.append(curr_poly)
        cloud_batch.append(curr_cloud_mask)
        poly_for_viz_batch.append(curr_poly_for_viz)
        dates_batch.append(curr_dates)
        sen1_series_batch.append(curr_sen1_series.squeeze(0))
        sen1_dates_batch.append(curr_sen1_dates)

    # Return the batch
    label_batch = torch.Tensor(np.array(label_batch)).to(config.DEVICE).long()
    return im_series_batch, label_batch, poly_batch, cloud_batch, poly_for_viz_batch, dates_batch, sen1_series_batch, sen1_dates_batch

def pad_input_sequences(im_series_batch, sen1_series_batch, config_in=None):
    if config_in is not None:
        config = config_in
    current_batch_size = len(im_series_batch)
    seq_lens = [im_series_batch[j].shape[0] for j in range(current_batch_size)]
    padded_seqs = nn.utils.rnn.pad_sequence([im_series_batch[j] for j in range(current_batch_size)], batch_first=True)
    # Also pad the sen1 data
    seq_lens_sen1 = [sen1_series_batch[j].shape[0] for j in range(current_batch_size)]
    padded_seqs_sen1 = nn.utils.rnn.pad_sequence([sen1_series_batch[j] for j in range(current_batch_size)], batch_first=True)
    return padded_seqs, seq_lens, padded_seqs_sen1, seq_lens_sen1

def get_model_prediction(padded_seqs, seq_lens, padded_seqs_sen1, seq_lens_sen1, model, train_mode=True, config_in=None):
    if config_in is not None:
        config = config_in
    if train_mode:
        current_batch_size = len(padded_seqs)
        predictions = []
        for curr_model in model:
            if config.CONCAT_SEN1:
                prediction, _ = curr_model([padded_seqs, padded_seqs_sen1], [seq_lens, seq_lens_sen1])
            else:
                prediction, _ = curr_model(padded_seqs, seq_lens)
            predictions.append(prediction)
        # Average the predictions from the ensemble of models
        tmp_var = []
        for aa in range(current_batch_size):
            all_preds = []
            for bb in range(len(predictions)):
                all_preds.append(predictions[bb][aa])
            mean_pred = torch.mean(torch.stack(all_preds), dim=0)
            tmp_var.append(mean_pred)
        # Final prediction done at final hidden state of LSTM
        prediction = torch.stack([tmp_var[j][-1, :] for j in range(current_batch_size)])
        if config.EVAL_ONLY:
            return prediction
        else:
            # In this case, return two things: prediction as above, and also the predictions at all time steps.
            all_predictions = []
            for aa in range(current_batch_size):
                all_preds = []
                for bb in range(len(predictions)):
                    all_preds.append(predictions[bb][aa])
                all_predictions.append(torch.stack(all_preds).squeeze(0))
            return prediction, all_predictions
    else:  # Validation mode
        # Get the model's prediction
        predictions_val_curr = []
        for curr_model in model:
            if config.CONCAT_SEN1:
                prediction_val, _ = curr_model([padded_seqs, padded_seqs_sen1], [None, None])
            else:
                prediction_val, _ = curr_model(padded_seqs)
            predictions_val_curr.append(prediction_val)
        # Average the final-step predictions (predictions done at end of LSTM processing)
        # Median gives better results than mean
        last_step_preds = [predictions_val_curr[k][:, -1, :] for k in range(len(predictions_val_curr))]
        stacked_preds = torch.stack(last_step_preds)
        # We are interested here in checking the agreement between the models.
        # We obtain this by argmaxing along the last dimension.
        hard_preds = torch.argmax(stacked_preds, dim=2)
        most_common_pred = torch.mode(hard_preds, dim=0).values
        nbr_agree = torch.sum(hard_preds == most_common_pred)
        agreement_frac = nbr_agree / stacked_preds.shape[0]
        stacked_all_step_preds = torch.stack(predictions_val_curr).squeeze(1)
        if config.CONCAT_SEN1:  # only median of final step predictions
            prediction_val, _ = torch.median(stacked_preds, dim=0)
        else:  # Here we do a median over the last x time steps, as kind of a smoothing
            prediction_all_steps, _ = torch.median(stacked_all_step_preds, dim=0)
            prediction_val, _ = torch.median(prediction_all_steps[-config.PRED_MEDIAN_LAST_X:, :], dim=0)
            prediction_val.unsqueeze_(0)
        return prediction_val, agreement_frac, stacked_all_step_preds

def track_stats(sc, predicted_classes, gt_classes, train_mode=True, agreement_fracs=None):
    if train_mode:
        appendix = ''
    else:
        appendix = '_val'
    acc = np.mean(predicted_classes == gt_classes)
    sc.s('Accuracy' + appendix).collect(acc)
    if np.count_nonzero(gt_classes == 1) > 0:
        acc_grazing = np.mean(predicted_classes[gt_classes == 1] == 1)
        sc.s('Accuracy_grazing' + appendix).collect(acc_grazing)
    if np.count_nonzero(gt_classes == 0) > 0:
        acc_no_activity = np.mean(predicted_classes[gt_classes == 0] == 0)
        sc.s('Accuracy_no_activity' + appendix).collect(acc_no_activity)
    f1 = sklearn.metrics.f1_score(gt_classes, predicted_classes, average='macro', zero_division=0)
    prec = sklearn.metrics.precision_score(gt_classes, predicted_classes, average='macro', zero_division=0)
    rec = sklearn.metrics.recall_score(gt_classes, predicted_classes, average='macro', zero_division=0)
    both_prec = sklearn.metrics.precision_score(gt_classes, predicted_classes, average=None, zero_division=0)
    both_rec = sklearn.metrics.recall_score(gt_classes, predicted_classes, average=None, zero_division=0)
    if f1 > 0:
        sc.s('F1_score' + appendix).collect(f1)
    if prec > 0:
        sc.s('Precision' + appendix).collect(prec)
    if len(both_prec) == 2:
        prec_grazing = both_prec[1]
        prec_no_activity = both_prec[0]
        if prec_grazing > 0:
            sc.s('Precision_grazing' + appendix).collect(prec_grazing)
        if prec_no_activity > 0:
            sc.s('Precision_no_activity' + appendix).collect(prec_no_activity)
    if rec > 0:
        sc.s('Recall' + appendix).collect(rec)
    if len(both_rec) == 2:
        rec_grazing = both_rec[1]
        rec_no_activity = both_rec[0]
        if rec_grazing > 0:
            sc.s('Recall_grazing' + appendix).collect(rec_grazing)
        if rec_no_activity > 0:
            sc.s('Recall_no_activity' + appendix).collect(rec_no_activity)
    if appendix == '_val' and agreement_fracs is not None:
        # Print model uncertainty stuff
        agreement_fracs = np.array(agreement_fracs)
        correct_preds = predicted_classes == gt_classes
        correct_preds_when_gt_is_graz = (correct_preds==1) & (gt_classes==1)
        incorrect_preds_when_gt_is_graz = (correct_preds==0) & (gt_classes==1)
        correct_preds_when_gt_is_non_graz = (correct_preds==1) & (gt_classes==0)
        incorrect_preds_when_gt_is_non_graz = (correct_preds==0) & (gt_classes==0)
        agrees_correct = agreement_fracs[correct_preds]
        mean_agree_correct = np.mean(agrees_correct)
        agrees_incorrect = agreement_fracs[~correct_preds]
        mean_agree_inc = np.mean(agrees_incorrect)
        agrees_graz_correct = agreement_fracs[correct_preds_when_gt_is_graz]
        mean_agree_graz_correct = np.mean(agrees_graz_correct)
        agrees_graz_incorrect = agreement_fracs[incorrect_preds_when_gt_is_graz]
        mean_agree_graz_incorrect = np.mean(agrees_graz_incorrect)
        agrees_non_graz_correct = agreement_fracs[correct_preds_when_gt_is_non_graz]
        mean_agree_non_graz_correct = np.mean(agrees_non_graz_correct)
        agrees_non_graz_incorrect = agreement_fracs[incorrect_preds_when_gt_is_non_graz]
        mean_agree_non_graz_incorrect = np.mean(agrees_non_graz_incorrect)
        print("Mean agreement (correct, incorrect, graz correct, graz incorrect, non-graz correct, non-graz incorrect): (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)"
                % (mean_agree_correct, mean_agree_inc, mean_agree_graz_correct, mean_agree_graz_incorrect, mean_agree_non_graz_correct, mean_agree_non_graz_incorrect))
        
        # Do a 3-row-2-column histogram plot of the agreement fractions with matplotlib
        _, axs = plt.subplots(2, 3, figsize=(16, 10))
        font_sz = 14
        tick_sz = 13
        line_thick = 2
        actual_agree_frac = True
        
        for i in range(2):
            for j in range(3):
                axs[i, j].tick_params(axis='both', which='major', labelsize=tick_sz)
                axs[i, j].grid(True)
                axs[i, j].set_ylabel('Amount', fontsize=font_sz)
                if actual_agree_frac:
                    axs[i, j].set_xlabel('Agreement fraction', fontsize=font_sz)
                    axs[i, j].set_ylim([0, 35])
                    axs[i, j].set_xlim([0.5, 1])
                else:
                    axs[i, j].set_xlabel('Mean dist to mean prob', fontsize=font_sz)
                    axs[i, j].set_ylim([0, 10])
                    axs[i, j].set_xlim([0, 0.5])

        axs[0, 0].hist(agrees_correct, bins=20, align='mid')
        axs[0, 0].set_title('Correct predictions', fontsize=font_sz)
        axs[0, 0].axvline(x=mean_agree_correct, color='r', linestyle='--', linewidth=line_thick)
        
        axs[1, 0].hist(agrees_incorrect, bins=20, align='mid')
        axs[1, 0].set_title('Incorrect predictions', fontsize=font_sz)
        axs[1, 0].axvline(x=mean_agree_inc, color='r', linestyle='--', linewidth=line_thick)
        
        axs[0, 1].hist(agrees_graz_correct, bins=20, align='mid')
        axs[0, 1].set_title('Correct predictions | grazing examples', fontsize=font_sz)
        axs[0, 1].axvline(x=mean_agree_graz_correct, color='r', linestyle='--', linewidth=line_thick)
        
        axs[1, 1].hist(agrees_graz_incorrect, bins=10, align='mid')
        axs[1, 1].set_title('Incorrect predictions | grazing examples', fontsize=font_sz)
        axs[1, 1].axvline(x=mean_agree_graz_incorrect, color='r', linestyle='--', linewidth=line_thick)
        
        axs[0, 2].hist(agrees_non_graz_correct, bins=10, align='mid')
        axs[0, 2].set_title('Correct predictions | non-grazing examples', fontsize=font_sz)
        axs[0, 2].axvline(x=mean_agree_non_graz_correct, color='r', linestyle='--', linewidth=line_thick)
        
        axs[1, 2].hist(agrees_non_graz_incorrect, bins=10, align='mid')
        axs[1, 2].set_title('Incorrect predictions | non-grazing examples', fontsize=font_sz)
        axs[1, 2].axvline(x=mean_agree_non_graz_incorrect, color='r', linestyle='--', linewidth=line_thick)

        plt.tight_layout()
        plt.savefig(os.path.join(sc.log_dir, 'agreement_fractions.png'))
        plt.close()

def visualize_results(tot_ctr, label_dict, stat_train_dir, im_series_batch, label_batch, poly_for_viz_batch, predicted_classes, dataset_val,
                      im_series_batch_val, label_batch_val, poly_for_viz_batch_val, predicted_classes_val, config_in=None):
    if config_in is not None:
        config = config_in
    if config.EVAL_ONLY:                
        # Plot images of ALL validation examples in this case.
        for val_idx in range(len(dataset_val)):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            idx = random.randint(0, config.BATCH_SIZE - 1)
            im_series = im_series_batch[idx]  # T x C x H x W
            tstep = random.randint(0, im_series.shape[0] - 1)
            im_rgb = im_series[tstep, [3,2,1], :, :].cpu().detach().numpy()  # 3 x H x W
            # go to shape (H, W, 3)
            im_rgb = np.transpose(im_rgb, (1, 2, 0))
            label = label_batch[idx].cpu().detach().numpy().item()
            poly_for_viz = poly_for_viz_batch[idx]
            im_rgb = (im_rgb - np.min(im_rgb)) / (np.max(im_rgb) - np.min(im_rgb))
            H = im_rgb.shape[0]
            plt.imshow(im_rgb)
            plt.title("Label: %s, Prediction: %s" % (label_dict[label], label_dict[predicted_classes[idx]]))
            plt.plot(poly_for_viz[:,0], H-poly_for_viz[:,1], color='red', linewidth=2)

            # Now the same but for val example.
            plt.subplot(1, 2, 2)
            idx = val_idx
            im_series = im_series_batch_val[idx]
            tstep = random.randint(0, im_series.shape[3] - 1)
            im_rgb = im_series[:, :, [3,2,1], tstep]
            label = label_batch_val[idx]
            poly_for_viz = poly_for_viz_batch_val[idx]
            im_rgb = (im_rgb - np.min(im_rgb)) / (np.max(im_rgb) - np.min(im_rgb))
            plt.imshow(im_rgb)
            plt.title("Label: %s, Prediction: %s" % (label_dict[label], label_dict[predicted_classes_val[idx]]))
            plt.plot(poly_for_viz[:,0], H-poly_for_viz[:,1], color='red', linewidth=2)

            # Save figure in stat_train_dir.
            plt.savefig(os.path.join(stat_train_dir, f"viz_{val_idx}.png"))
            plt.cla()
            plt.clf()
            plt.close('all')
    else:
        # Do a plot with left | right, where left = train example, right = val example.
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        idx = random.randint(0, config.BATCH_SIZE - 1)
        im_series = im_series_batch[idx]  # T x C x H x W
        tstep = random.randint(0, im_series.shape[0] - 1)
        im_rgb = im_series[tstep, [3,2,1], :, :].cpu().detach().numpy()  # 3 x H x W
        # go to shape (H, W, 3)
        im_rgb = np.transpose(im_rgb, (1, 2, 0))
        label = label_batch[idx].cpu().detach().numpy().item()
        poly_for_viz = poly_for_viz_batch[idx]
        im_rgb = (im_rgb - np.min(im_rgb)) / (np.max(im_rgb) - np.min(im_rgb))
        H = im_rgb.shape[0]
        plt.imshow(im_rgb)
        plt.title("Label: %s, Prediction: %s" % (label_dict[label], label_dict[predicted_classes[idx]]))
        plt.plot(poly_for_viz[:,0], H-poly_for_viz[:,1], color='red', linewidth=2)

        # Now the same but for val example.
        plt.subplot(1, 2, 2)
        idx = random.randint(0, len(dataset_val) - 1)
        im_series = im_series_batch_val[idx]
        tstep = random.randint(0, im_series.shape[3] - 1)
        im_rgb = im_series[:, :, [3,2,1], tstep]
        label = label_batch_val[idx]
        poly_for_viz = poly_for_viz_batch_val[idx]
        im_rgb = (im_rgb - np.min(im_rgb)) / (np.max(im_rgb) - np.min(im_rgb))
        plt.imshow(im_rgb)
        plt.title("Label: %s, Prediction: %s" % (label_dict[label], label_dict[predicted_classes_val[idx]]))
        plt.plot(poly_for_viz[:,0], H-poly_for_viz[:,1], color='red', linewidth=2)

        # Save figure in stat_train_dir.
        plt.savefig(os.path.join(stat_train_dir, f"viz_{tot_ctr}.png"))
        plt.cla()
        plt.clf()
        plt.close('all')

def setup_model_and_optimizer(H, W, nbr_channels, config_in=None):
    if config_in is not None:
        config = config_in
    if config.MODEL_LOAD_PATH is None:
        if config.MODEL_TYPE == 'LSTM' or config.MODEL_TYPE == 'biLSTM':
            model = LSTMClassifier(nbr_channels, 2, config.CNN_OUT_DIM, config.HIDDEN_DIM_LSTM, config.NBR_LSTM_LAYERS, H, W, config.MODEL_TYPE=='biLSTM', two_branches=config.CONCAT_SEN1)
        else:
            print("Model type set by config.MODEL_TYPE not defined, try something else.")
            sys.exit()
        model.to(config.DEVICE)
    else:  # Currently only implemented for LSTM models
        model = []
        model_load_paths = config.MODEL_LOAD_PATH
        if not isinstance(model_load_paths, list):
            model_load_paths = [model_load_paths]
        for model_load_path in model_load_paths:
            curr_model = LSTMClassifier(nbr_channels, 2, config.CNN_OUT_DIM, config.HIDDEN_DIM_LSTM, config.NBR_LSTM_LAYERS, H, W, config.MODEL_TYPE=='biLSTM', two_branches=config.CONCAT_SEN1)
            curr_model.load_state_dict(torch.load(model_load_path, map_location=config.DEVICE))
            curr_model.to(config.DEVICE)
            model.append(curr_model)
    if not isinstance(model, list):
        model = [model]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model[0].parameters(), lr=config.LR)
    return model, criterion, optimizer

def setup_stat_collector(stat_train_dir, print_iter=10, also_val_stats=True, config_in=None):
    if config_in is not None:
        config = config_in
    sc = StatCollector(stat_train_dir, config.NBR_EPOCHS, print_iter)
    sc.register('CE_loss', {'type': 'avg', 'freq': 'step'})
    sc.register('Accuracy', {'type': 'avg', 'freq': 'epoch'})
    sc.register('Accuracy_grazing', {'type': 'avg', 'freq': 'epoch'})
    sc.register('Accuracy_no_activity', {'type': 'avg', 'freq': 'epoch'})
    sc.register('F1_score', {'type': 'avg', 'freq': 'epoch'})
    sc.register('Precision', {'type': 'avg', 'freq': 'epoch'})
    sc.register('Recall', {'type': 'avg', 'freq': 'epoch'})
    sc.register('Precision_grazing', {'type': 'avg', 'freq': 'epoch'})
    sc.register('Recall_grazing', {'type': 'avg', 'freq': 'epoch'})
    sc.register('Precision_no_activity', {'type': 'avg', 'freq': 'epoch'})
    sc.register('Recall_no_activity', {'type': 'avg', 'freq': 'epoch'})
    # Add the same stats but with a '_val' suffix
    if also_val_stats:
        sc.register('CE_loss_val', {'type': 'avg', 'freq': 'step'})
        sc.register('Accuracy_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('Accuracy_grazing_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('Accuracy_no_activity_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('F1_score_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('Precision_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('Recall_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('Precision_grazing_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('Recall_grazing_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('Precision_no_activity_val', {'type': 'avg', 'freq': 'epoch'})
        sc.register('Recall_no_activity_val', {'type': 'avg', 'freq': 'epoch'})
    return sc
