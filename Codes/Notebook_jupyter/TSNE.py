# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# Données d'exemple (remplacez cela par vos propres données)
df_fb = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/coord_acp_landsat_train153va.csv")
df_r = pd.read_csv("/home/data/ter_meduse_log/our_data/data_ter_distribution/outputed_csv/all_infos.csv", index_col=0)
df_rt = df_r.iloc[:, 0]
df_rt
# %%

df_r = df_r.iloc[:, 51:]
df_r["surveyId"] = df_rt.values

#X = np.random.rand(100, 50)  # 100 échantillons avec 50 dimensions
X = df_fb.drop(columns='surveyId').values
df_r
# %%
df_fb = pd.merge(df_fb, df_r, on="surveyId", how="inner")

# Appliquer t-SNE
tsne = TSNE(n_components=2, perplexity=50, random_state=42)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, columns=['dim_1_tsne_bio', 'dim_2_tsne_bio'])
df["surveyId"] = df_rt

df.to_csv("coordonnees_tsne_landsat.csv", index=False)

# %%
"""
for i in range(len(df_r.columns)) :
    id_esp = df_r.columns[i]
    mask_0 = df_fb[id_esp] == 0
    mask_1 = df_fb[id_esp] == 1
    # Visualiser les données t-SNE
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[mask_0, 0], X_tsne[mask_0, 1], c='blue', label='0')
    plt.scatter(X_tsne[mask_1, 0], X_tsne[mask_1, 1], c='red', label='1')
    plt.title('t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig("tsne_bioclim50_espece "+str(id_esp)+".png")
    plt.show()
"""
