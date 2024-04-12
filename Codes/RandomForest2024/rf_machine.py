import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Data train
train_bio_ave = pd.read_csv('GLC24-PA-train-bioclimatic-average.csv')
train_elev = pd.read_csv('GLC24-PA-train-elevation.csv')
train_human = pd.read_csv('GLC24-PA-train-human-footprint.csv')
train_land = pd.read_csv('GLC24-PA-train-landcover.csv')
train_soilgrid = pd.read_csv('GLC24-PA-train-soilgrids.csv')

abio = train_bio_ave.merge(train_elev, on='surveyId')\
        .merge(train_human, on='surveyId')\
                .merge(train_land, on='surveyId')\
                    .merge(train_soilgrid, on='surveyId')
                    
abio = abio.replace([np.inf], np.nan)
abio = abio.replace([-np.inf], np.nan)
abio = abio.fillna(abio.mean())
                    
# Listes espèces
species = pd.read_csv('GLC24-PA-metadata-train.csv')
species['speciesId'] = species['speciesId'].astype(int)

#Regroupement des surveyId
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
    

# Data TEST
test_bio_ave = pd.read_csv('GLC24-PA-test-bioclimatic-average.csv')
test_elev = pd.read_csv('GLC24-PA-test-elevation.csv')
test_human = pd.read_csv('GLC24-PA-test-human-footprint.csv')
test_land = pd.read_csv('GLC24-PA-test-landcover.csv')
test_soilgrid = pd.read_csv('GLC24-PA-test-soilgrids.csv')

abiotest = test_bio_ave.merge(test_elev, on='surveyId')\
        .merge(test_human, on='surveyId')\
                .merge(test_land, on='surveyId')\
                    .merge(test_soilgrid, on='surveyId')


# Random Forest
proba_predict={}

X_test = abiotest.iloc[:,1:].values
X_train = abio.iloc[:,1:].values
model = RandomForestClassifier(n_estimators=100)

for i in range(len(species_patch)-1):
    Y_train = presence[i][1]
    model.fit(X_train, Y_train)
    Y_pred = model.predict_proba(X_test)[:,1]
    species_name = presence[i][0]
    proba_predict[species_name] = Y_pred
    print(i)
    
pa_proba = pd.DataFrame(proba_predict)

# Transformation proba >= 0.2 en présence 
def replace_with_column_name(row):
    return [col for col, value in row.items() if value >= 0.2]

pa_predict = pa_proba.apply(replace_with_column_name, axis=1)
pa_predict = pd.DataFrame(pa_predict)

# Format kaggle
pa_predict.insert(0, "surveyId", abiotest["surveyId"])
pa_predict = pa_predict.rename(columns={pa_predict.columns[1]: "Predict"})
blind = pa_predict.rename(columns={'Predict': 'predictions'})
blind['predictions'] = blind['predictions'].apply(lambda x: str(x).replace('[', '').replace(']', '').replace(',', '').replace("'",''))
blind.to_csv("RF_100_allparam_mean_prob02.csv", index=False, sep=',')

