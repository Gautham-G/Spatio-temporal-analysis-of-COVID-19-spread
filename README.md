# Spatio-temporal modelling and hotspot detection for the spreadof COVID-19 using Mobility and Exposure

Final Project for CSE 8803 (Data Science in Epidemiology)

## Setting up

1. Make sure you have anaconda/miniconda installed
2. run `bash ./scripts/py_setup.sh`  (Remove cuda dependency if you don't have it)

## Prerpocessing

1. Run `python ./script/pattern_extract.py`
2. Other preprocessing steps are in `./scripts/data_util.py`

## Training model

1. Run `python ./save_once.py` to extract all visit counts and case data. (Takes some time)
2. Run  `./train_model.py` with relevant parameters. (See `run.sh` for example of 1-4 week ahead predictions)
3. The predictions will be saved in `./pred_dir` folder.
   