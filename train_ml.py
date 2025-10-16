import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
import torch
import numpy as np
import random
import datetime
from shutil import copyfile
from utils import load_dataset, compute_dataset_means_stds, create_batch, setup_model_and_optimizer
from utils import setup_stat_collector, pad_input_sequences, get_model_prediction, track_stats, visualize_results
import config


# Here we load models if in evaluation mode
if config.EVAL_ONLY:
    assert isinstance(config.MODEL_LOAD_PATH_BASE, str), "Set config.MODEL_LOAD_PATH_BASE as string pointing to saved model folder (in /log)"
    model_load_paths = []
    for SEED in config.SEED_LIST:
        model_load_paths.append(config.MODEL_LOAD_PATH_BASE + '/' + str(SEED) + '/train_stats/model_weights.pth')
    config.MODEL_LOAD_PATH = model_load_paths
else:
    config.MODEL_LOAD_PATH = None

# Create log directory where to store runs for all seeds
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(config.BASE_PATH_LOG, timestamp)
os.makedirs(log_dir, exist_ok=False)
copyfile("train_ml.py", os.path.join(log_dir, "train_ml.py"))
copyfile("config.py", os.path.join(log_dir, "config.py"))
copyfile("classes.py", os.path.join(log_dir, "classes.py"))
copyfile("utils.py", os.path.join(log_dir, "utils.py"))

# Loop over all seeds
for SEED in config.SEED_LIST:

    # Create a folder for the current seed
    curr_folder = str(SEED)
    stat_train_dir = os.path.join(log_dir, curr_folder, "train_stats")
    os.makedirs(stat_train_dir, exist_ok=False)

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
    dataset_train, dataset_val = load_dataset(config)

    # Get some dimensions
    H, W, nbr_channels, _ = dataset_train[0][0].shape
    _, _, nbr_channels_sen1, _ = dataset_train[0][8][0].shape
    if config.CLOUD_MASK_INPUT:
        nbr_channels += 1
    if config.POLY_MASK_INPUT:
        nbr_channels += 1
    nbr_channels -= (12 - len(config.BANDS_TO_USE))  # Adjust for the bands used
    nbr_train = len(dataset_train)

    # Setup band dictionary
    band_mapping = {'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4, 'B06': 5, 'B07': 6, 'B08': 7, 'B8A': 8, 'B09': 9, 'B11': 10, 'B12': 11}
    # Extract relevant indices based on config.BANDS_TO_USE
    band_idxs_use = [band_mapping[band] for band in config.BANDS_TO_USE]
    if config.POLY_MASK_INPUT:
        band_idxs_use.append(12)

    # Setup label dictionary
    label_dict = {0: 'No activity', 1: 'Active grazing'}

    # Note that the first element of each element in dataset_train has shape H x W x C x T.
    # Based on this, we want to compute mean and std for each channel over the whole dataset_train.
    means, stds, means_sen1, stds_sen1 = compute_dataset_means_stds(dataset_train, nbr_channels_sen1)

    # Model, loss function, optimizer
    model, criterion, optimizer = setup_model_and_optimizer(H, W, nbr_channels, config)

    # Setup StatCollector
    sc = setup_stat_collector(stat_train_dir, 10, True, config)

    # Go over all the data points in the dataset_train and train the model
    tot_ctr = 0
    dataset_train_orig = dataset_train.copy()
    for ep_idx in range(config.NBR_EPOCHS):

        # Shuffle the dataset_train
        random.shuffle(dataset_train_orig)
        dataset_train = dataset_train_orig.copy()

        # Run an epoch
        for i in range(0, nbr_train, config.BATCH_SIZE):

            # Create current batch via individual data points
            im_series_batch, label_batch, poly_batch, cloud_batch, poly_for_viz_batch, \
            dates_batch, sen1_series_batch, sen1_dates_batch = create_batch(dataset_train, i, means, stds, means_sen1,
                                                                            stds_sen1, band_idxs_use, apply_data_aug=True,
                                                                            config_in=config)

            # Before feeding to model, prepare inputs by padding (since each element in the batch
            # can have different number of timesteps) and then pack the sequence
            padded_seqs, seq_lens, padded_seqs_sen1, seq_lens_sen1 = pad_input_sequences(im_series_batch, sen1_series_batch, config)

            # Get the model's prediction
            if config.EVAL_ONLY:
                prediction = get_model_prediction(padded_seqs, seq_lens, padded_seqs_sen1, seq_lens_sen1, model, True, config)
                all_predictions = None
            else:
                curr_batch_size = len(im_series_batch)
                prediction, all_predictions = get_model_prediction(padded_seqs, seq_lens, padded_seqs_sen1, seq_lens_sen1, model, True, config)
                shortest_seq = min(seq_lens)
                assert shortest_seq == np.min([len(tmp_pred) for tmp_pred in all_predictions])
                sec_half = int(shortest_seq / 2)
                # Now create a list of length sec_half, where each element has the same shape
                # as the prediction tensor, but which correspond to the earler timestep predictions.
                preds_sec_half = []
                for j in range(sec_half, shortest_seq - 1):  # -1 because the very last timestep already has a loss component
                    curr_j_preds = []
                    for k in range(curr_batch_size):
                        curr_len = seq_lens[k]
                        assert curr_len == all_predictions[k].shape[0]
                        curr_j_preds.append(all_predictions[k][j + curr_len - shortest_seq, :])
                    preds_sec_half.append(torch.stack(curr_j_preds, dim=0))

            # Get the predicted classes for the batch, and the actual classes
            predicted_classes = torch.argmax(prediction, dim=1).cpu().detach().numpy()
            gt_classes = label_batch.cpu().detach().numpy()
            
            # Track some stats
            track_stats(sc, predicted_classes, gt_classes)

            # Calculate loss and do backpropagation
            loss = criterion(prediction, label_batch)
            if all_predictions is not None:
                # OBS: This part is not part of the techincal report, but after the report we found that
                # doing the below is slightly benecifial during training. What happens here is hat we also
                # add a loss component for those PRED_MEDIAN_LAST_X (=4 by default) last probability estimates,
                # and not only for the very last step of the LSTM. This is in accoradence with the inference
                # procedure, wherein we take the median of the last 4 predictions, rather than only the last
                # prediction.
                loss *= 0.5
                loss_sec_half = 0
                tmp_ctr = 0
                for j in range(len(preds_sec_half)):
                    if j >= len(preds_sec_half) - config.PRED_MEDIAN_LAST_X:  # only add loss for the last X timesteps
                        loss_sec_half += criterion(preds_sec_half[j], label_batch)
                        tmp_ctr += 1
                tmp_ctr = max(1, tmp_ctr)  # max operation ensures that we do not divide by zero
                loss += 0.5 * loss_sec_half / tmp_ctr
            if not config.EVAL_ONLY:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sc.s('CE_loss').collect(loss.cpu().detach().numpy())

            # Occassionally validate the model on the validation set and track associated stats.
            if tot_ctr % 100 == 0:
                with torch.no_grad():

                    # Create current batch
                    im_series_batch_val = [dataset_val[j][0] for j in range(len(dataset_val))]
                    label_batch_val = [dataset_val[j][1] for j in range(len(dataset_val))]
                    poly_batch_val = [dataset_val[j][3] for j in range(len(dataset_val))]
                    poly_for_viz_batch_val = [dataset_val[j][4] for j in range(len(dataset_val))]
                    cloud_batch_val = [dataset_val[j][5] for j in range(len(dataset_val))]
                    im_series_sen1_batch_val = [dataset_val[j][8][0] for j in range(len(dataset_val))]

                    # Based on the above, create the actual batches
                    label_val = np.array(label_batch_val)
                    poly_val = np.stack(poly_batch_val, axis=0)

                    # Convert the numpy arrays to torch tensors
                    output_tensor_val = torch.Tensor(label_val).to(config.DEVICE).long()
                    poly_tensor_val = torch.Tensor(poly_val[:, :, :, np.newaxis, np.newaxis]).to(config.DEVICE)
                    poly_tensor_val = poly_tensor_val.permute(0, 4, 3, 1, 2)

                    # We do one element at a time, since the time series lengths may differ
                    predictions_val = []
                    agreement_fracs = []
                    poly_areas = []  # track relative polygon areas
                    series_lens = []  # track the lengths of the series
                    for j in range(len(im_series_batch_val)):
                        im_series_val = im_series_batch_val[j][np.newaxis, :, :, :, :]
                        cloud_masks_val = cloud_batch_val[j][np.newaxis, :, :, np.newaxis, :]
                        input_tensor_val = torch.Tensor(im_series_val).to(config.DEVICE).permute(0, 4, 3, 1, 2)
                        cloud_tensor_val = torch.Tensor(cloud_masks_val).to(config.DEVICE).permute(0, 4, 3, 1, 2)
                        curr_poly_tensor_val = poly_tensor_val[j]
                        im_series_sen1_val = im_series_sen1_batch_val[j][np.newaxis, :, :, :, :]
                        input_sen1_tensor_val = torch.Tensor(im_series_sen1_val).to(config.DEVICE).permute(0, 4, 3, 1, 2)

                        # Track relative polygon area and series length
                        poly_areas.append(np.sum(curr_poly_tensor_val.cpu().detach().numpy()) / (curr_poly_tensor_val.shape[2] * curr_poly_tensor_val.shape[3]))
                        series_lens.append(input_tensor_val.shape[1])

                        # Normalize the input tensor
                        input_tensor_val = (input_tensor_val - means) / stds
                        input_sen1_tensor_val = (input_sen1_tensor_val - means_sen1) / stds_sen1

                        # Optionally append the cloud mask to the input tensor
                        if config.CLOUD_MASK_INPUT:
                            input_tensor_val = torch.cat((input_tensor_val, cloud_tensor_val), dim=2)
                            # TODO: SHOULD THE CLOUD MASK BE ADDED TO THE SEN1 DATA AS WELL, SO THAT THIS PART OF
                            # THE ML MODEL UNDERSTANDS THAT A CERTAIN PART OF SEN2 IS CLOUDED?

                        # Optionally append the polygon mask to the input tensor
                        if config.POLY_MASK_INPUT:
                            # Ensure that the poly_tensor has the same number of timesteps as the input_tensor
                            tmp = curr_poly_tensor_val[np.newaxis, :, :, :, :]
                            tmp = tmp.repeat(1, input_tensor_val.shape[1], 1, 1, 1)
                            # Now append the poly_tensor to the input_tensor
                            input_tensor_val = torch.cat((input_tensor_val, tmp), dim=2)
                            # Also for sen1
                            tmp = curr_poly_tensor_val[np.newaxis, :, :, :, :]
                            tmp = tmp.repeat(1, input_sen1_tensor_val.shape[1], 1, 1, 1)
                            input_sen1_tensor_val = torch.cat((input_sen1_tensor_val, tmp), dim=2)

                        # Poly is a binary mask, where 1 indicates the polygon and 0 indicates non-polygon
                        # and has shape H x W.
                        if config.MASK_OUT_NONPOLY:
                            input_tensor_val *= curr_poly_tensor_val[np.newaxis, :, :, :, :]
                            input_sen1_tensor_val *= curr_poly_tensor_val[np.newaxis, :, :, :, :]

                        # At this stage, remove bands that are not to be used
                        input_tensor_val = input_tensor_val[:, :, band_idxs_use, :, :]

                        # Get the model's prediction + the fraction of agreement between the models (1.0 = all models agree, for example)
                        prediction_val, agreement_frac, stack_preds_all_steps_val = get_model_prediction(input_tensor_val, None, input_sen1_tensor_val,
                                                                                                         None, model, False, config)
                        predictions_val.append(prediction_val)
                        agreement_fracs.append(agreement_frac.cpu().detach().numpy())

                    # Get the predicted classes for the batch, and the actual classes
                    prediction_val = torch.cat(predictions_val, dim=0)
                    predicted_classes_val = torch.argmax(prediction_val, dim=1).cpu().detach().numpy()
                    gt_classes_val = output_tensor_val.cpu().detach().numpy()

                    # Track some stats
                    if config.EVAL_ONLY:
                        track_stats(sc, predicted_classes_val, gt_classes_val, train_mode=False, agreement_fracs=agreement_fracs)
                    else:
                        track_stats(sc, predicted_classes_val, gt_classes_val, train_mode=False)

            # Print some stats
            if tot_ctr % 10 == 0:
                sc.print()
                sc.save()
                print("Epoch (%d, / %d), Iter (%d / %d), Loss: %f" % (ep_idx, config.NBR_EPOCHS, i, nbr_train, loss))

            # Visualize some results
            if config.DO_VISUALIZE and tot_ctr % 2500 == 0:
                visualize_results(tot_ctr, label_dict, stat_train_dir, im_series_batch, label_batch,
                                  poly_for_viz_batch, predicted_classes, dataset_val, im_series_batch_val,
                                  label_batch_val, poly_for_viz_batch_val, predicted_classes_val, config)

            if config.EVAL_ONLY:
                print("Done with evaluation")
                sys.exit()

            # Increment the total counter
            tot_ctr += 1

    # Save model weights to log path.
    if config.SAVE_MODEL:
        torch.save(model[0].state_dict(), os.path.join(stat_train_dir, 'model_weights.pth'))

    # We are done with the current seed
    print("Done with current seed!")

# Done overall
print("Done with everything!")
