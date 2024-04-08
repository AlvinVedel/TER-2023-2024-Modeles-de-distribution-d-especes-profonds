#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import Dense
import random
from keras.initializers import glorot_uniform

print("imports OK")


# In[8]:


df = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/all_infos.csv", index_col=0)
df_test = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/test_all_infos.csv", index_col=0)
print("dfs lu")


# In[7]:


X = df.iloc[:,0:51]
y = df.iloc[:, 51:]
X = X.drop(columns='surveyId')
X = X.drop(columns="geoUncertaintyInM")
X = X.drop(columns="HumanFootprint-Roads")


print("X et y créées")


# In[8]:


matrix = []
for i in range(len(X)):
    new_liste = []
    for col in X.columns :
        new_liste.append(X[col][i])
    matrix.append(new_liste)

yt = y.copy()
yt = np.array(yt).astype(np.float32)
Xt = np.array(matrix)
Xt = Xt.astype(np.float32)

print("conversion en array")


# In[1]:


model = keras.Sequential()
model.add(keras.Input(shape=(48,)))

for i in range(30):
    model.add(layers.Dense(1024, activation="relu"))

"""
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))

model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))

model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))


model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
"""

model.add(layers.Dense(5016, activation="softmax", name='output'))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-6),
    loss="binary_crossentropy"
  )

print("création + compilation modèle")
print(model.summary())


# In[18]:


history = model.fit(
    Xt,
    yt,
    batch_size=10,
    epochs=20,
)
print("Entrainement terminé")


# In[ ]:


data_test = df_test.copy()
data_test = data_test.drop(columns="surveyId")
data_test = data_test.drop(columns="geoUncertaintyInM")
data_test = data_test.drop(columns="HumanFootprint-Roads")
X_test = []
for i in range(len(data_test)):
    new_liste = []
    for col in data_test.columns :
        new_liste.append(data_test[col][i])
    X_test.append(new_liste)
    
X_t = np.array(X_test)
X_t = X_t.astype(np.float32)

print("modifications X_test effectuées")


# In[ ]:


pred = model.predict(X_t)

print("predictions faites")

liste_especes = y.columns.to_list()

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

df.to_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/probas_nn_1024x30_relu_sftmax.csv", sep=',', index=False)

print("fin")
