# %%

import pandas as pd
import numpy as np

# %%
df = pd.read_csv('./RF_PA_100RF_2000PO_5RF_prob02.csv')

# %%
nb = []
for i in range(len(df['predictions'])):
 nb.append(len(str(df['predictions'].loc[i]).split()))



# %%# %%
import statistics
statistics.mean(nb)