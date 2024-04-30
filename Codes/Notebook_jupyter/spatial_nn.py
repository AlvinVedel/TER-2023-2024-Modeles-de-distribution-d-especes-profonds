# %%
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import Dense
import random
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


print("imports OK")
# %%
df = pd.read_csv("/home/data/ter_meduse_log/our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv")
df = df[["lon", "lat", "speciesId"]]
df_po = pd.read_csv("/home/data/ter_meduse_log/our_data/output/PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv")
df_po = df_po[["lon", "lat", "speciesId"]]

concatenated_df = pd.concat([df, df_po])

print("df concat")
# %%
concatenated_df = concatenated_df.sample(frac=1).reset_index(drop=True)

# %%

liste_especes = list(set(concatenated_df["speciesId"]))
nb_especes = len(liste_especes)
print(nb_especes)

# %%
print(concatenated_df.head())

X = concatenated_df[["lon", 'lat']]
y = concatenated_df["speciesId"]




matrix = X.values

yt = y.values
yt = yt.astype(np.int32)
y_encoded = label_encoder.fit_transform(yt)
#yt = np.array(yt).astype(np.float32)
Xt = np.array(matrix)
Xt = Xt.astype(np.float32)



print("creation model")
# %%

print(Xt[0:5])
print(y_encoded[0:5])

# %%

model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
for i in range(50):
    model.add(layers.Dense(100, activation="relu"))
# %%
model.add(layers.Dense(10358, activation="softmax", name='output'))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy"
  )

history = model.fit(
    Xt,
    y_encoded,
    batch_size=200,
    epochs=5,
)

print("train fait")
# %%
df_test = pd.read_csv("/home/data/ter_meduse_log/our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv")

data_test = df_test.copy()
data_test = data_test[["lon", "lat"]]
X_test = []
for i in range(len(data_test)):
    new_liste = []
    for col in data_test.columns :
        new_liste.append(data_test[col][i])
    X_test.append(new_liste)
    
X_t = np.array(X_test)
X_t = X_t.astype(np.float32)

print("modifications X_test effectuées")


pred = model.predict(X_t)

print("predictions faites")


colonnes = []
for i in range(len(pred[0])):
    print(i)
    colonne_espece = []
    for j in range(len(pred)):
        colonne_espece.append(pred[j][i])
    colonnes.append(colonne_espece)
    
for i in range(len(liste_especes)) :
    df_test[liste_especes[i]] = colonnes[i]
    
print("colonnes ajoutées")    


# In[ ]:


df_final = df_test.drop(df_test.columns[1:51], axis=1)
liste_especes = df_final.columns[1:].to_list()

df = df_final.copy()

colonne_liste = []
for i in range(len(df)):
    liste = []
    print(i)
    for j in liste_especes :
        liste.append(df[j][i])
    colonne_liste.append(liste)
    
df["pred"] = colonne_liste
    
print("probas en liste")

df.to_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/probas_nn_200x30_prelu.csv", sep=',', index=False)

print("fin")
