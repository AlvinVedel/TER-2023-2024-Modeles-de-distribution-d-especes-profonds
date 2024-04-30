# %%
import pandas as pd
import numpy as np

best_bioclim = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/topX+25cnn_bioclim.csv")
best_landsat = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/topX+25cnn_landsat.csv")

def chaine_as_list(chaine) :
    splited = chaine.split()
    return splited

best_bioclim["liste"] = best_bioclim["predictions"].apply(chaine_as_list)
best_landsat["liste"] = best_landsat["predictions"].apply(chaine_as_list)


# %%

global_df = best_bioclim.merge(best_landsat, how='inner', on='surveyId')

global_df
# %%


big_liste = []
for i in range(len(global_df)):
    liste = []
    for specie in global_df["liste_y"][i] :
        if specie in global_df["liste_x"][i] :
            liste.append(specie)
    big_liste.append(liste)

#
global_df["predictions"] = big_liste


nb_esp = pd.read_csv("../../../our_data/data_ter_distribution/outputed_csv/nb_esp_regression.csv")


for i in range(len(global_df)) :
    landsat = True
    nb_a_ajt = int(nb_esp["nb_a_pred"][i]+19 - len(global_df["predictions"][i]))
    indices_landsat = [global_df["liste_y"][i].index(valeur) for valeur in global_df["liste_y"][i] if valeur not in global_df["predictions"][i]]
    indices_bioclim = [global_df["liste_x"][i].index(valeur) for valeur in global_df["liste_x"][i] if valeur not in global_df["predictions"][i]]
    added = 0
    while nb_a_ajt > added :
        if len(indices_landsat)> indices_landsat[0] < indices_bioclim[0]+3 :
            global_df["predictions"][i].append(global_df["liste_y"][i][indices_landsat[0]])
            indices_landsat.pop(0)
            added+=1
        else :
            global_df["predictions"][i].append(global_df["liste_x"][i][indices_bioclim[0]])
            indices_bioclim.pop(0)
            added+=1

        




def liste_as_chaine(liste) :
    chaine = ""
    for el in liste :
        chaine+=el+' '
    return chaine

global_df["predictions"] = global_df["predictions"].apply(liste_as_chaine)

final_df = global_df[["surveyId", "predictions"]]
final_df.to_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/result/merge_predictions3_2.csv", index=False)