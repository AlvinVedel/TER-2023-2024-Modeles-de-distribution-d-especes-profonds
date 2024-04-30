# %%

import pandas as pd
import numpy as np
from PIL import Image

# %%

image_pil = Image.open('/home/data/ter_meduse_log/our_data/output/SatellitePatches/pa_train_patches_rgb/34/94/229434.jpeg')

image_np = np.array(image_pil)

# Afficher le type de données et la forme du tableau NumPy
print("Type de données:", image_np.dtype)
print("Forme:", image_np.shape)
