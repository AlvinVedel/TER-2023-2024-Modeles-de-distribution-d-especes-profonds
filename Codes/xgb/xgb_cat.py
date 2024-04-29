#%%
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, make_scorer, precision_score, recall_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

#%%
species = pd.read_csv('../../Data/Data_2024/GLC24-PA-metadata-train.csv')
species['speciesId'] = species['speciesId'].astype(int)

acp_biotest = pd.read_csv('../../../Data_TER/feature/coord_acp_bioclim_test93va.csv')
acp_biotrain = pd.read_csv('../../../Data_TER/feature/coord_acp_bioclim_train93va.csv')

acp_landtest = pd.read_csv('../../../Data_TER/feature/coord_acp_landsat_test153va.csv')
acp_landtrain = pd.read_csv('../../../Data_TER/feature/coord_acp_landsat_train153va.csv')

elev = pd.read_csv('../../Data/Data_2024/Environnemental_Data/train/GLC24-PA-train-elevation.csv')
test_elev = pd.read_csv('../../Data/Data_2024/Environnemental_Data/test/GLC24-PA-test-elevation.csv')

abio = acp_biotrain.merge(acp_landtrain, on='surveyId').merge(elev, on='surveyId')
abiotest = acp_biotest.merge(acp_landtest, on='surveyId').merge(test_elev, on='surveyId')

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
len_y = []
for i in range (len(species_patch)):
    len_y.append(presence[i][1].sum())

bins = [0, 3, 10, 100, 1000, float('inf')]

counts = pd.cut(len_y, bins, right=False).value_counts()
counts = counts.sort_index()
counts


# %%
dfcat = pd.DataFrame({'ID':species_patch['speciesId'], 'Count':len_y})

categories = pd.cut(dfcat['Count'], bins=[0, 3, 10, 100, 1000, float('inf')], labels=['1-3', '3-10', '10-100', '100-1000', '1000-inf'])
dfcat['Category'] = categories

#%%
group0 = dfcat[dfcat['Category'] == '1-3']['ID'].astype(str)
group1 = dfcat[dfcat['Category'] == '3-10']['ID'].astype(str)
group10 = dfcat[dfcat['Category'] == '10-100']['ID'].astype(str)
group100 = dfcat[dfcat['Category'] == '100-1000']['ID'].astype(str)
group1000 = dfcat[dfcat['Category'] == '1000-inf']['ID'].astype(str)


# %%
# Dataframe selon catégorie
presence0 = [(id, data) for id, data in presence if id in group0.values]
presence1 = [(id, data) for id, data in presence if id in group1.values]
presence10 = [(id, data) for id, data in presence if id in group10.values]
presence100 = [(id, data) for id, data in presence if id in group100.values]
presence1000 = [(id, data) for id, data in presence if id in group1000.values]


# %%
X = abio.iloc[:,1:].values
Y = presence100[0][1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
ind = len(Y)
minoritaire = Y.sum()
weight = (ind - minoritaire) / minoritaire

# %%
model = XGBClassifier()
param_grid = {
    'scale_pos_weight': [50,weight],
    'learning_rate': [0.01,0.1], 
    'max_depth': [1,2,4], 
    'n_estimators': [100], 
    'min_child_weight': [1,2,4],
    'reg_alpha': [0,0.2],  
    'reg_lambda': [0,0.2]
}

grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=-1)
grid_search.fit(X, Y)
#%%
print(grid_search.best_params_, grid_search.best_score_)

#%%
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


#%%
#**grid_search.best_params_
# xgb100 = XGBClassifier(n_estimators=500,
#                      learning_rate=0.1, 
#                      max_depth=4,
#                      min_child_weight=1,
#                      reg_alpha=0.2,  
#                      reg_lambda=0)
xgb100.fit(X_resampled, y_resampled)
y_pred = xgb100.predict(X_test)

TP = sum((y_pred == 1) & (y_test == 1))
FP = sum((y_pred == 1) & (y_test == 0))
TN = sum((y_pred == 0) & (y_test == 0))
FN = sum((y_pred == 0) & (y_test == 1))

print("Vrais Positifs:", TP, "    |    Faux Positifs:", FP, "     |    Vrais Négatifs:", TN, "     |    Faux Négatifs:", FN)



# %%
y_prob = xgb100.predict_proba(X_test)[:,1]

prec=[]
for i in range (100):
    y_adjust = (y_prob > i/100).astype(int)
    if sum((y_adjust == 1) & (y_test == 0)) == 0:
        prec.append(np.nan)
    else:
        TP = sum((y_adjust == 1) & (y_test == 1))
        FP = sum((y_adjust == 1) & (y_test == 0))
        FN = sum((y_adjust == 0) & (y_test == 1))
        prec.append(TP/FP)

seuil = prec.index(max(prec))/100


# %%
y_adjust = np.array([1 if y > seuil else 0 for y in y_prob])
TP = sum((y_adjust == 1) & (y_test == 1))
FP = sum((y_adjust == 1) & (y_test == 0))
TN = sum((y_adjust == 0) & (y_test == 0))
FN = sum((y_adjust == 0) & (y_test == 1))

print("Vrais Positifs:", TP, "    |    Faux Positifs:", FP, "     |    Vrais Négatifs:", TN, "     |    Faux Négatifs:", FN)


# %%
