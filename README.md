# Grazing Detection using Deep Learning and Sentinel-2 Time Series Data
<img width="1806" height="450" alt="Image" src="https://github.com/user-attachments/assets/9f05be15-dbbb-4546-b997-5a8e339221f6" />

Official code repository for the paper [Grazing Detection using Deep Learning and Sentinel-2 Time Series Data](https://arxiv.org/abs/2510.14493), summarizing the outcomes of a project conducted by [RISE Research Institutes of Sweden](https://www.ri.se/en) and the [Swedish Board of Agriculture (SBA)](https://jordbruksverket.se/languages/english/swedish-board-of-agriculture), with funding from the [Swedish National Space Agency (SNSA)](https://www.rymdstyrelsen.se/en/). Code mainly developed by [Aleksis Pirinen](https://aleksispi.github.io/) (RISE), with contributions from [Delia Fano Yela](https://delialia.github.io/) and [Smita Chakraborty](https://www.ri.se/en/person/smita-chakraborty) (both at RISE). [Erik Källman](https://www.ri.se/en/person/erik-kallman) (RISE) was the project leader. We thank the SBA, and [Niklas Boke Olen](https://scholar.google.com/citations?user=aUvgSgoAAAAJ&hl=en) in particular, for their valuable contributions, including key parts of the data used for this work.

## Data
The polygons and associated labels in this project were obtained from the SBA -- please contact their GIS department (Gis.Support@jordbruksverket.se) if you want to ask for access to their polygons and labels, including the ones we used in this project. However, satellite data can be freely downloaded from the [_Digital Earth Sweden_](https://digitalearth.se/) platform using scripts provided within this code repository, where you could of course specify regions of interest with polygons of your own.

## Setup
First, clone this repo. Then, see the included `requirements.txt` file for all Python packages used on a fresh Ubuntu 24.04.2 machine that is equipped with a modern Nvidia-5090 GPU (install the packages in the requirements file with the command: `pip install -r requirements.txt`). Note that you might want and have to change the torch-versions in the requirements file if you have another (older) GPU, or if you don't intend to use a GPU at all. **Tip:** It might be a good idea to first set up a virtual environment on your machine, prior to installing packages from `requirements.txt`.

## Overview of main files of importance in this repo

`config.py` is the main file for setting various parameters in the other files to be run.

`train_ml.py` is the main file for training an ML-based satellite image time series classifier (Sentinel 2 L2A data assumed). It assumes a dataset of time series with associated binary labels (1 = grazing activity | 0 = no grazing activity), where a single label is given per full time series. The dataset was provided as polygons from the SBA, and then associated time series satellite imagery was downloaded from [_Digital Earth Sweden_](https://digitalearth.se/) using the script `download_and_process_gpkgs_as_processes.py` (described below), and the downloaded data was in turn made ML-ready using the script `prepare_sen2_ml_dataset.py` (described below). **Note:** Models can also be _evaluated_ using `train_ml.py`, by setting the `EVAL_ONLY` flag in `config.py` to `True`, and pointing MODEL_LOAD_PATH_BASE in `config.py` to the folder containing trained model weights. Pre-trained ML models that give ~80\% accuracy can be [downloaded here](https://drive.google.com/file/d/1AG__4BXAM8y4iK9E3AFw36V8EfpTkhyo/view?usp=sharing) (extract and put into the `log` folder in the repository).

`plot_results.py` can be used for plotting some results, generated e.g. from the training statistics within `train_ml.py` (see above).

`download_and_process_gpkgs_as_processes.py` is the script that is used to download data from _Digital Earth Sweden_. Details should be quite clear by looking within the file.

`explore-2022-jv-data.py` and `explore-2024-jv-data.py` are scripts for investigating the polygons from the SBA, where we have labels for each time series. Before using any of these two scripts, download the content of [this folder](https://drive.google.com/drive/folders/11OCkuh46NwOr_mHgnVNhWOLiT8A1oFtg?usp=sharing) and place it into a folder called `country-borders` that is outside (one level "up") of this repository. You also need access to the polygon data from SBA (see _Data_ above).

`prepare_sen2_ml_dataset.py` takes the data by the SBA (which can be explored in the files `explore-2022-jv-data.py` and `explore-2024-jv-data.py`) and makes it into a format that can then be ML-trained within the file `train_ml.py` (see above). Note that [cloud predictions](https://github.com/aleksispi/ml-cloud-opt-thick) is also performed as a preprocessing step, and by default, only polygons without clouds are kept in the time series.

`inspect_poly_timeseries.py` is a file that visualizes various polygon tile time series that were downloaded using the script `download_and_process_gpkgs_as_processes.py`.

`utils.py` is a basic file with various utilities (imported by other scripts in the repo). It's best understood by checking examining the file contents.

## ML model results
Running the default `train_ml.py` code with `EVAL_MODE=True` and `MODEL_LOAD_PATH_BASE = 'log/2025-07-06_10-40-23'` in `config.py` (and having [downloaded](https://drive.google.com/file/d/1AG__4BXAM8y4iK9E3AFw36V8EfpTkhyo/view?usp=sharing) ML models, extracted them and put them into the `log` directory), should give this output on the data from the SBA:
```
Mean CE_loss                             tot:    0.22250, ma:    0.22250, last:    0.22250
Mean Accuracy                            tot:    0.90000, ma:    0.90000, last:    0.90000
Mean Accuracy_grazing                    tot:    1.00000, ma:    1.00000, last:    1.00000
Mean Accuracy_no_activity                tot:    0.83333, ma:    0.83333, last:    0.83333
Mean F1_score                            tot:    0.89899, ma:    0.89899, last:    0.89899
Mean Precision                           tot:    0.90000, ma:    0.90000, last:    0.90000
Mean Recall                              tot:    0.91667, ma:    0.91667, last:    0.91667
Mean Precision_grazing                   tot:    0.80000, ma:    0.80000, last:    0.80000
Mean Recall_grazing                      tot:    1.00000, ma:    1.00000, last:    1.00000
Mean Precision_no_activity               tot:    1.00000, ma:    1.00000, last:    1.00000
Mean Recall_no_activity                  tot:    0.83333, ma:    0.83333, last:    0.83333
Mean Accuracy_val                        tot:    0.79661, ma:    0.79661, last:    0.79661
Mean Accuracy_grazing_val                tot:    0.86667, ma:    0.86667, last:    0.86667
Mean Accuracy_no_activity_val            tot:    0.72414, ma:    0.72414, last:    0.72414
Mean F1_score_val                        tot:    0.79514, ma:    0.79514, last:    0.79514
Mean Precision_val                       tot:    0.80235, ma:    0.80235, last:    0.80235
Mean Recall_val                          tot:    0.79540, ma:    0.79540, last:    0.79540
Mean Precision_grazing_val               tot:    0.76471, ma:    0.76471, last:    0.76471
Mean Recall_grazing_val                  tot:    0.86667, ma:    0.86667, last:    0.86667
Mean Precision_no_activity_val           tot:    0.84000, ma:    0.84000, last:    0.84000
Mean Recall_no_activity_val              tot:    0.72414, ma:    0.72414, last:    0.72414

```

## Notes on minor differences relative to paper
<!--Since delivering the [technical report](https://drive.google.com/file/d/1rtJascI-Jae9VVzqEsALWPQEP05QnqbQ/view?usp=sharing) of this project, there have been two minor changes done in the code:-->
<!--* Skip using _adjacent time step shifting_ data augmentation (see Sec. 3.2.1 in the report). This data augmentation technique did in the end not improve results (as is also apparent in Table 2 in the report), so we don't use it anymore, and nor is it possible to turn it on within `config.py`.-->
* Slightly changed loss computation, so that the last 4 time step predictions are used within the loss, not only the last step prediction. Please refer to Line 137 and onwards within `train_ml.py` for more on this.

## Citation
If you use this repository and/or find our paper useful, please cite the following:

TODO: Add bib entry here.

## License
This project is released under the **MIT License**.  
Copyright (c) 2025, RISE Research Institutes of Sweden.

See the full text in [the license file](https://github.com/aleksispi/pib-ml-grazing/blob/main/license.md).
