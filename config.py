# Global variables for ML training in train_ml.py
SEED_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # The code will run for these seeds, and store all associated runs in a folder named by the timestamp.
BASE_PATH_LOG = 'log'  # Base path for the log files
DATASET_PKL_BASE = 'ml_dataset_seed0'  # Base name for the dataset pickle files (for the Sentinel-2 data)
DEVICE = 'cuda'  # 'cuda' or 'cpu'. Default: 'cuda'
NBR_EPOCHS = 300  # Number of epochs to train the model. Default: 300
LR = 0.0001  # Learning rate. Default: 0.0001
BATCH_SIZE = 10  # Batch size. Default: 10
MODEL_TYPE = 'biLSTM'  # 'LSTM' or 'biLSTM'. Default: 'biLSTM'
NBR_LSTM_LAYERS = 1  # Number of LSTM layers. Default: 1
HIDDEN_DIM_LSTM = 8  # Hidden dimension of the LSTM. Default: 8
CNN_OUT_DIM = 4  # Output dimension of the CNN block. Default: 4
TIMESTEP_DROPOUT_PROB = 0.35  # Probability of dropping out timesteps in the input image series (data aug). Default: 0.35
START_DATE = '2023-04-01'  # Start date for the timeseries (obs: year does not matter, only month and day). Default: April 1st
END_DATE = '2023-10-22'  # End date for the timeseries (exclusive) (obs: year does not matter, only month and day). Default: Oct 22nd
MASK_OUT_NONPOLY = True  # If True, then mask out all non-polygon pixels in the input image. Default: True
POLY_MASK_INPUT = False  # If True, then append the binary polygon mask to the input image series in the channel dimension. Default: False
CLOUD_MASK_INPUT = False  # If True, then append the binary cloud mask to the input image series in the channel dimension. Default: False
PRED_MEDIAN_LAST_X = 4  # Number of timesteps to use for median prediction (last X timesteps, where X=4 is default). Default: 4
THRESHOLD_THICKNESS_IS_CLOUD = 0.01  # If cloud optical thickness (COT) predicted above this for a pixel, it's predicted as 'cloudy'. Default: 0.01
IMG_ZOOMED_SIZE = 46  # None --> No effect. Images typically 90x90, but if e.g. 46 we pick the 46x46 center part of the  image. Default: 46
DO_VISUALIZE = False  # If True, then occasionally visualize a timestep of the input image series + prediction. Default: False
CONCAT_SEN1 = False  # If True, then concatenate the Sentinel-1 data to the Sentinel-2 data. Default: False
# B01 = coastal aerosol, B02-B04 = BGR, B05-B07 = vegetation red edge, B08 = NIR, B8A = narrow NIR, B09 = water vapor, B11-B12 = SWIR
BANDS_TO_USE = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']  # Bands to use in the Sentinel-2 data
SAVE_MODEL = True  # If True, then save the model after training. Default: True
MODEL_LOAD_PATH_BASE = None  # 'log/2025-07-06_10-40-23'  # 'log/2025-07-06_10-40-23'  # If not None, then load the model from this path (for evaluation mode)
EVAL_ONLY = False  # True --> Skip training, simply run once on validation set and terminate to obtain results