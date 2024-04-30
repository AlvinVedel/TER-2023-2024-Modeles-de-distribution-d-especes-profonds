# %%

import numpy as np
import keras
from tensorflow.keras.layers import LayerNormalization, Conv2D, MaxPooling2D, Dense, Input
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICE'] = '2'



# %%

df_tab = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/all_infos.csv", index_col=0)
df_fb = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/coord_acp_bioclim_train93va.csv")
for col in df_fb.columns :
    if col != 'surveyId' :
        df_fb = df_fb.rename(columns={col: "dim_b_"+col})
nb_b = 93
df_fb = df_fb.iloc[:, :nb_b]
df_fl = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/coord_acp_landsat_train153va.csv")
for col in df_fl.columns :
    if col != 'surveyId' :
        df_fl = df_fl.rename(columns={col: "dim_l_"+col})
nb_l=153
df_fl = df_fl.iloc[:, :nb_l]
# %%

df_fl

# %%

saved = []
for col in df_tab.iloc[:, :51].columns :
    if 'Bio' in col or 'Soilgrid' in col :
        saved.append(col)

cols = ['surveyId', 'lon', 'lat', 'Elevation', 'LandCover'] + saved


# %%

#df = pd.merge(df_tab[cols], df_fb, how='inner', on='surveyId') 
df = pd.merge(df_fb, df_fl, how='inner', on='surveyId')
#df = df_fb.copy()
df

# %%

model = keras.Sequential()
model.add(keras.Input(shape=(len(df.columns)-1,)))

model.add(keras.layers.Dense(512, activation='elu'))

model.add(keras.layers.Dense(1024, activation='elu'))

model.add(keras.layers.Dense(2048, activation='elu'))
model.add(keras.layers.Dense(2048, activation=keras.layers.PReLU()))
model.add(keras.layers.Dense(2048, activation='elu'))

model.add(keras.layers.Dense(4096, activation='elu'))
model.add(keras.layers.Dense(4096, activation=keras.layers.PReLU()))

model.add(keras.layers.Dense(6144, activation='elu'))
model.add(keras.layers.Dense(6144, activation=keras.layers.PReLU()))

model.add(keras.layers.Dense(8192, activation='elu'))
model.add(keras.layers.Dense(8192, activation=keras.layers.PReLU()))

model.add(keras.layers.Dense(10240, activation='elu'))
model.add(keras.layers.Dense(10240, activation=keras.layers.PReLU()))
model.add(keras.layers.Dense(10240, activation='elu'))

model.add(keras.layers.Dense(8192, activation='elu'))
model.add(keras.layers.Dense(8192, activation=keras.layers.PReLU()))

model.add(keras.layers.Dense(7168, activation='elu'))
model.add(keras.layers.Dense(6144, activation='elu'))

model.add(keras.layers.Dense(5016, activation="sigmoid"))

model.summary()
print("end")


model.compile("Adam", loss='binary_crossentropy')

# %%
## DONNEES TEST :

#df_test = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/test_all_infos.csv", index_col=0)
df_test_fl = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/coord_acp_landsat_test153va.csv")
for col in df_test_fl.columns :
    if col != 'surveyId' :
        df_test_fl = df_test_fl.rename(columns={col: "dim_l_"+col})
df_test_fl = df_test_fl.iloc[:, :nb_l]
df_test_fb = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/coord_acp_bioclim_test93va.csv")
for col in df_test_fb.columns :
    if col != 'surveyId' :
        df_test_fb = df_test_fb.rename(columns={col: "dim_l_"+col})
df_test_fb = df_test_fb.iloc[:, :nb_b]


df_test = pd.merge(df_test_fb, df_test_fl, how='inner', on='surveyId')

#  %%
X = df.copy()
y = df_tab.iloc[:, 51:]
X = X.drop(columns='surveyId')

# %%

y

# %%
matrix = []
for i in range(len(X)):
    new_liste = []
    for col in X.columns :
        new_liste.append(X[col][i])
    matrix.append(new_liste)

yt = y.copy()
y_train_tab = np.array(yt).astype(np.float32)   #####
Xt = np.array(matrix)
X_train_tab = Xt.astype(np.float32)    #####

data_test = df_test.copy()
data_test = data_test.drop(columns="surveyId")

X_test = []
for i in range(len(data_test)):
    new_liste = []
    for col in data_test.columns :
        new_liste.append(data_test[col][i])
    X_test.append(new_liste)
    
X_t = np.array(X_test)
X_test_tab = X_t.astype(np.float32)               #####


# %%
liste_especes = y.columns.to_list()

# %%
for e in range(5) :
    print("training1")

    history = model.fit(
        X_train_tab,
        y_train_tab,
        batch_size=32,
        epochs=5
    )

    pred = model.predict(X_test_tab)


    colonnes = []
    for i in range(len(pred[0])):
        print(i)
        colonne_espece = []
        for j in range(len(pred)):
            colonne_espece.append(pred[j][i])
        colonnes.append(colonne_espece)
        
    for i in range(len(liste_especes)) :
        df_test[liste_especes[i]] = colonnes[i]

    liste_column = ["surveyId"]+liste_especes
    df = df_test[liste_column]

    colonne_liste = []
    for i in range(len(df)):
        liste = []
        print(i)
        for j in liste_especes :
            liste.append(df[j][i])
        colonne_liste.append(liste)
        
    df["pred"] = colonne_liste
        

    df.to_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/probas_nn_meganet_tabulaire_normalized"+str(e)+".csv", sep=',', index=False)

    print("fin")


