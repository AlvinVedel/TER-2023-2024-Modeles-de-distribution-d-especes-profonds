
#%%
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import math

#%%
# Data train
train_bio_ave = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/train/GLC24-PA-train-bioclimatic-average.csv')
train_elev = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/train/GLC24-PA-train-elevation.csv')
train_human = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/train/GLC24-PA-train-human-footprint.csv')
train_land = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/train/GLC24-PA-train-landcover.csv')
train_soilgrid = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/train/GLC24-PA-train-soilgrids.csv')

abio = train_bio_ave.merge(train_elev, on='surveyId')\
        .merge(train_human, on='surveyId')\
                .merge(train_land, on='surveyId')\
                    .merge(train_soilgrid, on='surveyId')\
                    
abio = abio.replace([np.inf], np.nan)
abio = abio.replace([-np.inf], np.nan)
abio = abio.fillna(abio.mean())

# Data TEST
test_bio_ave = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/test/GLC24-PA-test-bioclimatic-average.csv')
test_elev = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/test/GLC24-PA-test-elevation.csv')
test_human = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/test/GLC24-PA-test-human-footprint.csv')
test_land = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/test/GLC24-PA-test-landcover.csv')
test_soilgrid = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/Environnemental_Data/test/GLC24-PA-test-soilgrids.csv')

abiotest = test_bio_ave.merge(test_elev, on='surveyId')\
        .merge(test_human, on='surveyId')\
                .merge(test_land, on='surveyId')\
                    .merge(test_soilgrid, on='surveyId')\
                    
abiotest = abiotest.fillna(abiotest.mean())

# Listes espèces
species = pd.read_csv('./../../MIASHS/TER-2023-2024-Modeles-de-distribution-d-especes-profonds/Data/Data_2024/GLC24-PA-metadata-train.csv')
species['speciesId'] = species['speciesId'].astype(int)


#%%
species_patch = species.groupby('speciesId')['surveyId'].agg(list).reset_index()

#Liste présence/absence
presence = []
for i in range(len(species_patch)):
    survey_id = species_patch['surveyId'].iloc[i]
    species_id = species_patch['speciesId'].iloc[i]
    # Pour chaque espèce on stock 0 ou 1 pour chaqeu surveyId
    presence_espece = abio['surveyId'].isin(survey_id).astype(int)
    # On rassemble les présences de toutes les espèces dans une liste
    presence.append((str(species_id), presence_espece))
    print(i)

#%%
predict={}

XTEST = abiotest.iloc[:,1:].values
XTRAIN = abio.iloc[:,1:].values

model = XGBClassifier()
param_grid = {
    'learning_rate': [0.1,0.5,1],
    'max_depth': [2,5,10],
    'n_estimators': [100],
    'min_child_weight' : [2,4],
    'reg_alpha': [0, 0.1],  
    'reg_lambda': [0, 0.1]
}

#%%
for i in range(4800,len(species_patch)):
    
    YTRAIN = presence[i][1]
    
    if YTRAIN.sum() > 4:
        
        ind = len(YTRAIN)
        minoritaire = YTRAIN.sum()
    
        # On cherche les meilleurs paramètres
        cv = math.floor(2+(minoritaire*50/ind))
        grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
        grid_search.fit(XTRAIN, YTRAIN)
    
        # Modèle avec poids
        xgb = XGBClassifier(scale_pos_weight=(ind - minoritaire) / minoritaire,
                            learning_rate=grid_search.best_params_['learning_rate'],
                            max_depth=grid_search.best_params_['max_depth'],
                            n_estimators=grid_search.best_params_['n_estimators'],
                            min_child_weight=grid_search.best_params_['min_child_weight'],
                            reg_alpha=grid_search.best_params_.get('reg_alpha', 0),
                            reg_lambda=grid_search.best_params_.get('reg_lambda', 1))
        
        
        # TROUVER LE SEUIL OPTIMAL
        # X_train, X_test, y_train, y_test = train_test_split(XTRAIN, YTRAIN, test_size=0.3, random_state=42)
        # xgb.fit(X_train, y_train)
        # y_pred = xgb.predict_proba(X_test)[:,1]
        # fpr, tpr, seuils = roc_curve(y_test, y_pred)    
        # seuil = seuils[np.argmin(np.sqrt(fpr**2 + (1-tpr)**2))]
        
        # Prédiction
        xgb.fit(XTRAIN, YTRAIN)
        YPRED = xgb.predict_proba(XTEST)[:,1]
        
        predict[presence[i][0]] = YPRED
        print(i)
    
    else:
        print(i)
        i+=1
    
    # SI LE NOMBRE DESPECE EST INFERIEUR A 4 : 
    # else:
    #     # Modèle avec poids
    #     xgb_unique = XGBClassifier(scale_pos_weight=(ind - minoritaire) / minoritaire, 
    #                                learning_rate=0.1,
    #                                max_depth=2,
    #                                min_child_weight= 2,
    #                                n_estimators=100,
    #                                reg_alpha= 0.1)
    #     #Seuil
    #     X_train, X_test, y_train, y_test = train_test_split(XTRAIN, YTRAIN, test_size=0.3, random_state=42)
    #     xgb_unique.fit(X_train, y_train)
    #     y_pred = xgb_unique.predict_proba(X_test)[:,1]
    #     fpr, tpr, seuils = roc_curve(y_test, y_pred)    
    #     seuil = seuils[np.argmin(np.sqrt(fpr**2 + (1-tpr)**2))]
        
    #     # Prédiction
    #     xgb_unique.fit(XTRAIN, YTRAIN)
    #     YPRED = (xgb_unique.predict_proba(XTEST)[:,1] > seuil).astype(int)
         
#%%
pa_xgb = pd.DataFrame(predict)

pa_xgb.to_csv("XGB_100n_prob_finsp.csv", index=False, sep=',')



#%%
xgb0 = pd.read_csv("./XGB_100n_prob_1000sp.csv")
xgb100 = pd.read_csv("./XGB_100n_prob_2000sp.csv")
xgb300 = pd.read_csv("./XGB_100n_prob_2500sp.csv")
xgb500 = pd.read_csv("./XGB_100n_prob_3000sp.csv")
xgb700 = pd.read_csv("./XGB_100n_prob_4000sp.csv")
xgb900 = pd.read_csv("./XGB_100n_prob_4500sp.csv")
xgb1100 = pd.read_csv("./XGB_100n_prob_4800sp.csv")
xgb1300 = pd.read_csv("./XGB_100n_prob_finsp.csv")


predict = pd.concat([xgb0,xgb100,xgb300,xgb500,xgb700,xgb900,xgb1100,xgb1300],axis=1)



#%%
def replace_with_column_name(row):
    return [col for col, value in row.items() if value > 0.7]

#%%
predict_test = predict.apply(replace_with_column_name, axis=1)
predict_test = pd.DataFrame(predict_test)
predict_test.insert(0, "surveyId", abiotest["surveyId"])
predict_test = predict_test.rename(columns={predict_test.columns[1]: "Predict"})
predict_test.head()

blind = predict_test.rename(columns={'Predict': 'predictions'})
blind['predictions'] = blind['predictions'].apply(lambda x: str(x).replace('[', '').replace(']', '').replace(',', '').replace("'",''))
blind.to_csv("XGB_100n_prob0.7_roc.csv", index=False, sep=',')




# BOUCLE POUR UNE SEULE ESPECE TES
#%%
YTRAIN = presence[112][1]
ind = len(YTRAIN)
minoritaire = YTRAIN.sum()

#%%
# On cherche les meilleurs paramètres
cv = math.floor(2+(minoritaire*100/ind))
grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_search.fit(XTRAIN, YTRAIN)


#%%
# Modèle avec poids
xgb = XGBClassifier(scale_pos_weight=(ind - minoritaire) / minoritaire,
                    learning_rate=grid_search.best_params_['learning_rate'],
                    max_depth=grid_search.best_params_['max_depth'],
                    n_estimators=grid_search.best_params_['n_estimators'],
                    min_child_weight=grid_search.best_params_['min_child_weight'],
                    reg_alpha=grid_search.best_params_.get('reg_alpha', 0),
                    reg_lambda=grid_search.best_params_.get('reg_lambda', 1))

   
#%% 
# Séparation pour trouver seuil
X_train, X_test, y_train, y_test = train_test_split(XTRAIN, YTRAIN, test_size=0.3, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict_proba(X_test)[:,1]
fpr, tpr, seuils = roc_curve(y_test, y_pred)    
seuil = seuils[np.argmin(np.sqrt(fpr**2 + (1-tpr)**2))]
    

#%%
# Prédiction
xgb.fit(XTRAIN, YTRAIN)
YPRED = (xgb.predict_proba(XTEST)[:,1] > seuil).astype(int)
        
#%%
predict[presence[4][0]] = YPRED

pa_xgb = pd.DataFrame(predict)


# %%
pa_xgb

