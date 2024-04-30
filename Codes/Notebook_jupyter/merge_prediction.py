# %%
import pandas as pd
import numpy as np

best_bioclim = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/cnn_bioclim_20tanh.csv")
best_landsat = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/cnn_landsat_20tanh.csv")

def chaine_as_list(chaine) :
    splited = chaine.split()
    length = 2*len(splited)/3
    splited = splited[:int(length)]
    return splited



best_bioclim["liste"] = best_bioclim["predictions"].apply(chaine_as_list)
best_landsat["liste"] = best_landsat["predictions"].apply(chaine_as_list)

# %%

global_df = best_bioclim.merge(best_landsat, how='inner', on='surveyId')
print(global_df)

big_liste = []
for i in range(len(global_df)):
    liste = global_df["liste_x"][i]
    for specie in global_df["liste_y"][i] :
        if specie not in liste :
            liste.append(specie)
    big_liste.append(liste)

global_df["predictions"] = big_liste

def liste_as_chaine(liste) :
    chaine = ""
    for el in liste :
        chaine+=el+' '
    return chaine

global_df["predictions"] = global_df["predictions"].apply(liste_as_chaine)

final_df = global_df[["surveyId", "predictions"]]
final_df.to_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/merge_predictions2.csv", index=False)