# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



# %%

df_train = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/all_infos.csv", index_col=0)
df_train.head()
# %%
variables_sup = df_train.iloc[:, :51]
variables_sup

# %%

train = pd.read_csv("../../../our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv")
train.head()

# %%

count_per_id = train.groupby('surveyId')['speciesId'].nunique().to_frame()
count_per_id
# %%
first_train = train.drop(columns = "speciesId")

merged_df = first_train.merge(count_per_id, left_on='surveyId', right_index=True, how='left')
merged_df = merged_df.drop_duplicates(subset='surveyId').reset_index().drop(columns="index")

merged_df.head()
# %%

merged_2 = merged_df.merge(variables_sup, left_on="surveyId", right_index=True, how='left')
merged_2

# %%

X_train = merged_df.iloc[:, :len(merged_df.columns)-2]
X_train

surveys = merged_df["surveyId"]
y_train

# %%


categorical_columns = ["region", "country"]
numeric_columns = ["lon", "lat", "year", "geoUncertaintyInM", "areaInM2"]
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns),  # Encodage one-hot pour les variables catégorielles
        ('num', 'passthrough', numeric_columns)  # Laisser les variables quantitatives inchangées
    ])

# Ajustement du pré-processeur aux données et transformation
X_train = preprocessor.fit_transform(X_train)


# %%

X_train = X_train.toarray()
y_train = np.array(y_train.values)
#X_train = np.array(X_train.values)
#y_train = np.array(y_train.values)

# %%
merged_2c = merged_2.copy()
y_train = merged_df["speciesId"]


# %%

test = merged_2.replace(-np.inf, np.nan, inplace=False)
test = test.drop(columns= ["speciesId", "region", "country", "surveyId_y", 'surveyId_x', "lon_y", "lat_y", "year_y", "geoUncertaintyInM_y"])

X_train2 = test.fillna(test.mean())
#X_train = np.array(X_train.values)
X_train2
# %%

model = xgb.XGBRegressor(objective ='reg:squarederror', max_depth=100, learning_rate=0.1, n_estimators=100)
dtrain = xgb.DMatrix(X_train2, label=y_train)


params = {
    'objective': 'reg:squarederror',  # Fonction objectif pour la régression
    'max_depth': 100,                    # Profondeur maximale de l'arbre
    'learning_rate': 0.1,              # Taux d'apprentissage
    'n_estimators': 100                # Nombre d'arbres à entraîner
}

# Entraîner le modèle XGBoost
model = xgb.train(params, dtrain)


# %%
train_predictions = model.predict(dtrain)

rmse = np.sqrt(np.mean((y_train - train_predictions) ** 2))
rmse

# %%
test = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/test_all_infos.csv", index_col=0)
#test = test.drop(columns="surveyId")
test = test.rename(columns={'lon': 'lon_x', 'lat': 'lat_x', 'year':'year_x', 'geoUncertaintyInM':'geoUncertaintyInM_x'})
test = test.replace(-np.inf, np.nan)
test = test.fillna(test.mean())
test.head()

# %%

test2 = pd.read_csv("../../../our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv")
test2 = test2[["surveyId", "areaInM2"]]

merged_2 = merged_df.merge(variables_sup, left_on="surveyId", right_index=True, how='left')
test = test.merge(test2, left_on="surveyId", right_index=True, how="left")
test = test.drop(columns = "surveyId_y")

# %%

#test_save = test.copy()
test = test_save.copy()
test

# %%
test = test.drop(columns = 'surveyId_x')
test







# %%

col = test.pop('areaInM2')
col







# %%

test.insert(4, 'areaInM2', col) 
test






# %%

# %%
test = test.replace(-np.inf, np.nan)
test = test.fillna(test.mean())

dtest = xgb.DMatrix(test)

nb_esp = model.predict(dtest)

# %%

print(np.mean(nb_esp))


# %%
nb_esp


test2 = pd.read_csv("../../../our_data/output/PresenceAbsenceSurveys/GLC24-PA-metadata-test.csv")
test2 = test2["surveyId"]
print(len(test2), len(nb_esp))

df = test2.to_frame()
df["nb_a_pred"] = nb_esp
df
# %%
df.to_csv("../../../our_data/data_ter_distribution/outputed_csv/nb_esp_regression.csv", index=False)
