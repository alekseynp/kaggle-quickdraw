# Kaggle: Quick, Draw! Doodle Recognition - 8th Place Solution

This is a stripped down repo of my 8th place solution in the 2018 Kaggle competition: [Quick, Draw! Doodle Recognition](https://www.kaggle.com/c/quickdraw-doodle-recognition/). Blog post/paper pending. Rushed summary drafted upon competition closure [here](https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73967/). 

This repo does not contain training code. Managing the raw 50 million example training set was non-trivial. I assume that anyone getting value from this code will implement something of their own.

## Setup

Weights for my best single model: `sh download_weights.sh`

Download kaggle data (requires cli setup): `sh download_test_raw.sh`

Requirements:
- `conda install --file requirements-conda.txt`
- `pip install -r requirements-pip.txt`

## Run

`python test.py`
