# %%

import pandas as pd
import numpy as np


# %%

red = pd.read_csv("../../../our_data/output/SatelliteTimeSeries/GLC24-PA-train-landsat-time-series-red.csv")
blue = pd.read_csv("../../../our_data/output/SatelliteTimeSeries/GLC24-PA-train-landsat-time-series-blue.csv")
green = pd.read_csv("../../../our_data/output/SatelliteTimeSeries/GLC24-PA-train-landsat-time-series-green.csv")
nir = pd.read_csv("../../../our_data/output/SatelliteTimeSeries/GLC24-PA-train-landsat-time-series-nir.csv")
swir1 = pd.read_csv("../../../our_data/output/SatelliteTimeSeries/GLC24-PA-train-landsat-time-series-swir1.csv")
swir2 = pd.read_csv("../../../our_data/output/SatelliteTimeSeries/GLC24-PA-train-landsat-time-series-swir2.csv")

pa_train = pd.read_csv("../../../our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv")

red.head()


# %%