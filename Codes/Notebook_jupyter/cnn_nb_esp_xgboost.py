# %%
import pandas as pd
import numpy as np

#df = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/result/cnn_bioclimatic_monthly5_20tanh.csv", index_col=0)
#df = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/result/cnn_landsat20_hidden_tanh1024.csv", index_col=0)
df = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/result/probas_nn_meganet_tabulaire_normalized1.csv")
nb_esp = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/nb_esp_regression.csv")
#df = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/result/RF_5_month_pa_proba.csv")
#%%
#abiotest =  pd.read_csv('../../../our_data/output/EnvironmentalRasters/Soilgrids/GLC24-PA-test-soilgrids.csv')
#df.insert(0, "surveyId", abiotest["surveyId"])

df.head()

# %%

df_test = nb_esp.copy()

top_X = []
copy = df.iloc[:, 1:-1]

copy.head()

# %%

for i in range(len(copy)):
    row = copy.iloc[i]
    nb = int(nb_esp["nb_a_pred"][i])+25
    top_X_columns = row.nlargest(nb).index.tolist()
    top_X.append(top_X_columns)


# %%
df["predictions"] = top_X
df = df[["surveyId", "predictions"]]

# %%

def liste_to_str(liste):
    chaine = ""
    for el in liste :
        chaine += str(el)+" "
    return chaine

df["predictions"] = df["predictions"].apply(liste_to_str)
df.head()

df.to_csv('../../../our_data/data_ter_distribution/outputed_csv/result/meganet11.csv', index=False)




# %%
