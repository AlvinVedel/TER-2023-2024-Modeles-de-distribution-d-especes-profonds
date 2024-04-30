# %%

import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Conv2D, MaxPooling2D, Dense, Input, Dropout
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
from scipy.interpolate import griddata
from PIL import Image


import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# %%

def basic_block(input_tensor, filters, stride=1):
    x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if stride != 1:
        input_tensor = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same')(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def build_resnet(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = basic_block(x, filters=64, stride=1)
    x = basic_block(x, filters=64, stride=1)
    x = basic_block(x, filters=128, stride=2)
    x = basic_block(x, filters=128, stride=1)
    x = basic_block(x, filters=256, stride=2)
    x = basic_block(x, filters=256, stride=1)
    x = basic_block(x, filters=512, stride=2)
    x = basic_block(x, filters=512, stride=1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation ="sigmoid", name='last_hidden')(x)
    x = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

# Définir la taille d'entrée et le nombre de classes
input_shape = (128, 128, 3)  # 3 canaux
num_classes = 5016  # Nombre de classes pour l'entraînement sur ImageNet, peut être ajusté pour votre tâche

# Construire le modèle ResNet-18
model = build_resnet(input_shape, num_classes)

# Afficher un résumé du modèle
model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy')

# %%

train = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/all_infos.csv")

survey = train["surveyId"].to_list()

dictionnaire = {}

base_path = '/home/data/ter_meduse_log/our_data/output/SatellitePatches/pa_train_patches_rgb/'

for i in range(len(survey)):
    print(i)
    idi = str(survey[i])
    while len(idi) < 3 :
        idi = '0'+idi
    ss_r = idi[-2:]
    sss_r = idi[-4:-2]
    path = base_path+ss_r+'/'+sss_r+'/'+idi+'.jpeg' 
    image_pil = Image.open(path)

    image_np = np.array(image_pil)
    dictionnaire[survey[i]] = image_np



# %%


selection = train.iloc[:, [1] + list(range(52, len(train.columns)))]
X = np.zeros((len(selection), 128, 128, 3), dtype=np.float32)
y = []
for i in range(len(selection)):
    print(i)
    y.append(selection.iloc[i, 1:])
    X[i] = dictionnaire[selection["surveyId"][i]]

print(X[0])
print(y[0])
#X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# %%

batch_size=32

history = model.fit(X,
 y, 
 batch_size=batch_size, 
 epochs=25
 )

# %%



# %%

test = pd.read_csv("../../../our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv")
test = test[["surveyId"]]

survey_t = test["surveyId"].to_list()

base_path = '/home/data/ter_meduse_log/our_data/output/SatellitePatches/pa_test_patches_rgb/'

dictionnaire_t = {}

for i in range(len(survey_t)):
    print(i)
    idi = str(survey_t[i])
    while len(idi) < 3 :
        idi = '0'+idi
    ss_r = idi[-2:]
    sss_r = idi[-4:-2]
    path = base_path+ss_r+'/'+sss_r+'/'+idi+'.jpeg' 
    image_pil = Image.open(path)

    image_np = np.array(image_pil)
    dictionnaire_t[survey_t[i]] = image_np


X_test = []
for i in range(len(test)):
    X_test.append(dictionnaire_t[test["surveyId"][i]])
X_test = np.array(X_test)

predictions = model.predict(X_test)

# %%



last_hidden_layer = model.get_layer('last_hidden').output

feature_extractor = Model(inputs=model.input, outputs=last_hidden_layer)

#feature_extractor.summary()

# %%


features = feature_extractor.predict(X_test)
features_train = feature_extractor.predict(X)
#features_train
# %%

df_features = pd.DataFrame(features)
df_features["surveyId"] = test["surveyId"].to_list()
df_features["type"] = "test"
df_features2 = pd.DataFrame(features_train)
df_features2["surveyId"] = selection["surveyId"].to_list()
df_features2["type"] = "train"


global_features = pd.concat([df_features, df_features2])

# %%

global_features.to_csv("../../../our_data/data_ter_distribution/outputed_csv/features20_satpatch_1024sigmoid.csv")


for i in range(1, len(selection.columns)):
    print(i)
    liste_espece = []
    for j in range(len(test)):
        liste_espece.append(predictions[j][i-1])
    test[selection.columns[i]] = liste_espece

test.to_csv('../../../our_data/data_ter_distribution/outputed_csv/result/cnn_satpatch20_hidden_sigmoid1024.csv')

