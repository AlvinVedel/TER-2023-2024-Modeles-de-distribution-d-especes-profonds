# %%


import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Conv2D, MaxPooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import keras
import os
from keras.initializers import glorot_uniform
import numpy as np
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '1'




# %%

all_data = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/features_1024elu.csv", index_col=0)

all_data.head()

# %%

X = all_data[all_data["type"]=="train"].iloc[:, :1024]

pca = PCA()
X_r = pca.fit(X).transform(X)

# %%

variances_explained = pca.explained_variance_ratio_
somme = 0
nb = 0

while somme < 0.90 :
    somme += variances_explained[nb]
    nb += 1

print(nb)


# %%

X_new = all_data[all_data["type"]=="test"].iloc[:, :1024]

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

X_new_projected = pca.transform(X_new_scaled)

#print(X_new_projected)
# %%
#print(X_new_projected.shape)

df_survey_test = all_data[all_data["type"]=="test"]["surveyId"]
df_survey_train = all_data[all_data["type"]=="train"]["surveyId"]

# %%
X_r = X_r[:,:nb]
X_new_projected = X_new_projected[:,:nb]
# %%

df_bioclim = pd.DataFrame(X_new_projected)
df_bioclim_train = pd.DataFrame(X_r)

# %%

df_final_te = pd.concat([df_survey_test, df_bioclim], axis=1)
df_final_tr = pd.concat([df_survey_train, df_bioclim_train], axis=1)

# %%

df_final_tr

# %% 
df_final_te.to_csv("../../../our_data/data_ter_distribution/outputed_csv/coord_acp_bioclim_test"+str(nb)+"va.csv", index=False)
df_final_tr.to_csv("../../../our_data/data_ter_distribution/outputed_csv/coord_acp_bioclim_train"+str(nb)+"va.csv", index=False)

# %%
