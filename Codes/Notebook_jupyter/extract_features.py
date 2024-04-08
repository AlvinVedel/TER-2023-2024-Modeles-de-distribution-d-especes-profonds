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

import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#pa_train = pd.read_csv("/home/data/ter_meduse_log/our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv")
test = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/all_infos.csv")
climat_train = pd.read_csv("/home/data/ter_meduse_log/our_data/output/EnvironmentalRasters/Climate/GLC24-PA-train-bioclimatic-monthly.csv")


climat_train.head()
# %%

dictionnaire = {}
first_year = 2000
last_year=2018
print("TOTAL : ",len(climat_train))

for i in range(len(climat_train)):
    print(i)
    ligne = climat_train.iloc[i, 1:].values
    image = np.zeros((19, 12, 4))
    indice = 0
    canaux_indice = 0
    year_indice = 0
    month_indice = 0
    while indice < len(climat_train.columns)-1:
        image[year_indice][month_indice][canaux_indice] = ligne[indice]
        canaux_indice+=1
        if canaux_indice==4 :
            canaux_indice=0
            month_indice+=1
            if month_indice==12:
                month_indice=0
                year_indice+=1
        indice+=1
    dictionnaire[climat_train["surveyId"][i]] = image
        

dictionnaire
# %%

#pa_train = pd.read_csv("../../../our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv")

#train = pa_train[["surveyId", "speciesId"]]
#train.head()

# %%
selection = test.iloc[:, [1] + list(range(52, len(test.columns)))]
selection.head()
# %%
#dictionnaire[212].shape

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

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = basic_block(x, filters=64, stride=1)
    x = basic_block(x, filters=64, stride=1)
    x = basic_block(x, filters=128, stride=2)
    x = basic_block(x, filters=128, stride=1)
    x = basic_block(x, filters=256, stride=2)
    x = basic_block(x, filters=256, stride=1)
    x = basic_block(x, filters=512, stride=2)
    x = basic_block(x, filters=512, stride=1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation ='tanh', name='last_hidden')(x)
    x = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

# Définir la taille d'entrée et le nombre de classes
input_shape = (19, 12, 4)  # 4 canaux
num_classes = 5016  # Nombre de classes pour l'entraînement sur ImageNet, peut être ajusté pour votre tâche

# Construire le modèle ResNet-18
model = build_resnet(input_shape, num_classes)

# Afficher un résumé du modèle
model.summary()

# %%

model.compile(optimizer='adam', loss='binary_crossentropy')

# %%

X = []
y = []
for i in range(len(selection)):
    print(i)
    y.append(selection.iloc[i, 1:])
    X.append(dictionnaire[selection["surveyId"][i]])

print(X[0])
print(y[0])
# %%
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# %%
batch_size=32

# Calculer le nombre d'étapes par époque


history = model.fit(X,
 y, 
 batch_size=batch_size, 
 epochs=8
 )


# %%

climat_test = pd.read_csv("/home/data/ter_meduse_log/our_data/output/EnvironmentalRasters/Climate/GLC24-PA-test-bioclimatic-monthly.csv")


dictionnaire_test = {}

for i in range(len(climat_test)):
    print(i)
    ligne = climat_test.iloc[i, 1:].values
    image = np.zeros((19, 12, 4))
    indice = 0
    canaux_indice = 0
    year_indice = 0
    month_indice = 0
    while indice < len(climat_test.columns)-1:
        image[year_indice][month_indice][canaux_indice] = ligne[indice]
        canaux_indice+=1
        if canaux_indice==4 :
            canaux_indice=0
            month_indice+=1
            if month_indice==12:
                month_indice=0
                year_indice+=1
        indice+=1
    dictionnaire_test[climat_test["surveyId"][i]] = image

# %%

df_test = pd.read_csv("../../../our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv")
df_test = df_test[["surveyId"]]

df_test.head()

# %%

X_test = []
for i in range(len(df_test)):
    X_test.append(dictionnaire_test[df_test["surveyId"][i]])
X_test = np.array(X_test)

predictions = model.predict(X_test)

print(predictions)


# %%
# FEATURES EXTRACTION

last_hidden_layer = model.get_layer('last_hidden').output

feature_extractor = Model(inputs=model.input, outputs=last_hidden_layer)

feature_extractor.summary()

# %%


features = feature_extractor.predict(X_test)
features_train = feature_extractor.predict(X)
features_train
# %%

df_features = pd.DataFrame(features)
df_features["surveyId"] = df_test["surveyId"].to_list()
df_features["type"] = "test"
df_features2 = pd.DataFrame(features_train)
df_features2["surveyId"] = selection["surveyId"].to_list()
df_features2["type"] = "train"


global_features = pd.concat([df_features, df_features2])

# %%

global_features.to_csv("../../../our_data/data_ter_distribution/outputed_csv/features_1024tanh.csv")



# %%


len(selection.columns)

# %%


for i in range(1, len(selection.columns)):
    print(i)
    liste_espece = []
    for j in range(len(df_test)):
        liste_espece.append(predictions[j][i-1])
    df_test[selection.columns[i]] = liste_espece

df_test.head()

# %%

df_test.to_csv('../../../our_data/data_ter_distribution/outputed_csv/result/cnn_bioclimatic_monthly2.csv')

# %%

### TOP K
#row = df_test.iloc[0]
#print(row.nlargest(30).tolist())

# Obtenir les noms des 30 colonnes avec les plus grandes valeurs dans la première ligne
#top_30_columns = row.nlargest(30).index.tolist()
#top_30_columns


# %%
copy = df_test.iloc[:, 1:]

top_30 = []
for i in range(len(copy)):
    row = copy.iloc[i]
    top_30_columns = row.nlargest(30).index.tolist()
    top_30.append(top_30_columns)
df_test["predictions"] = top_30
df_test = df_test[["surveyId", "predictions"]]
df_test.head()



# %%

def liste_to_str(liste):
    chaine = ""
    for el in liste :
        chaine += str(el)+" "
    return chaine

df_test["predictions"] = df_test["predictions"].apply(liste_to_str)
df_test.head()

df_test.to_csv('../../../our_data/data_ter_distribution/outputed_csv/result/cnn_bioclimatic_monthly_adadelta_top30_2.csv', index=False)

