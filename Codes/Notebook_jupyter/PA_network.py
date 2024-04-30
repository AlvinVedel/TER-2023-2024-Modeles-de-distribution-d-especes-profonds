#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import Dense, Dropout
import random
from keras.initializers import glorot_uniform
from keras.metrics import Precision, Recall
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print("imports OK")


# In[8]:


df = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/all_infos.csv", index_col=0)
df["geoUncertaintyInM"] = df["geoUncertaintyInM"].fillna(df["geoUncertaintyInM"].mean())

df_test = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/test_all_infos.csv", index_col=0)
df_test["geoUncertaintyInM"] = df_test["geoUncertaintyInM"].fillna(df_test["geoUncertaintyInM"].mean())

print("dfs lu")

# %%

# NORMALISATION DES DONNEES
    
for col in df.columns[1:51] :
    print(col)
    if col in df_test.columns :
        big_array = df[col].to_list() + df_test[col].to_list()
        moy = np.mean(big_array)
        std = np.std(big_array)
        normalized_array = (big_array - moy)/std
        colonne_train = normalized_array[:len(df)]
        colonne_test = normalized_array[len(df):]
        df[col] = colonne_train
        df_test[col] = colonne_test
    else : 
        print("la colonne", col, "n'est pas dans le test")


# In[7]:

saved = []
for col in df.iloc[:, :51].columns :
    if 'Bio' in col or 'Soilgrid' in col :
        saved.append(col)

cols = ['lon', 'lat', 'Elevation', 'LandCover'] + saved
nb_var = len(cols)


X = df[cols]
y = df.iloc[:, 51:]
#X = X.drop(columns='surveyId')
print("X et y créées")
# %%
#X.iloc[:, 20:42]


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
model.add(keras.Input(shape=(nb_var,)))


for i in range(35):
    model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dense(1024*2, activation='relu'))

model.add(layers.Dense(5016, activation="sigmoid", name='output'))

# %%


model.compile(
    optimizer="Adam",
    loss="binary_crossentropy"
)

print("création + compilation modèle")
print(model.summary())





history = model.fit(
    Xt,
    yt,
    batch_size=32,
    epochs=20
)
print("Entrainement terminé")
    
model.save("selfnet.h5")



# In[ ]:


data_test = df_test.copy()
data_test = data_test.drop(columns="surveyId")
data_test = data_test[cols]
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

df.to_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/probas_nn_tabulaire_normalized.csv", sep=',', index=False)

print("fin")
