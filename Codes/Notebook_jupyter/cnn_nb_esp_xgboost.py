# %%
import pandas as pd
import numpy as np

df = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/result/cnn_bioclimatic_monthly_adadelta.csv", index_col=0)
nb_esp = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/nb_esp_regression.csv")


# %%

df_test = nb_esp.copy()

top_X = []
copy = df.iloc[:, 1:]

for i in range(len(copy)):
    row = copy.iloc[i]
    nb = int(nb_esp["nb_a_pred"][i]) + 20
    top_X_columns = row.nlargest(nb).index.tolist()
    top_X.append(top_X_columns)


# %%
df["predictions"] = top_X
df = df[["surveyId", "predictions"]]
df.head()

# %%

def liste_to_str(liste):
    chaine = ""
    for el in liste :
        chaine += str(el)+" "
    return chaine

df["predictions"] = df["predictions"].apply(liste_to_str)
df.head()

df.to_csv('../../../our_data/data_ter_distribution/outputed_csv/result/cnn_bioclimatic_monthly_topK_xgboost20_adadelta.csv', index=False)



