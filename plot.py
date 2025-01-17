import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

run = 14

out_path = Path("MC_charge_spread/SPC_outputs")
# read the json file
with open(out_path / f"{str(run).zfill(4)}_photons.json" , "r", encoding="utf-8") as in_file:
    data = json.load(in_file)

cluster_lists = list(data[0].keys())

all_photons = []

# generate histogram for all clusters
for frame in data:
    for cluster in cluster_lists:
        all_photons += frame[cluster]

# convert to numpy array
all_photons = np.array(all_photons)

print(all_photons.shape)

hist = plt.hist(all_photons, bins=500, range=(0, 500))
plt.xlabel("ADU")
plt.ylabel("CLuster counts")
plt.yscale('log')
plt.show()



# record the histogram as a numpy array
#
np.save(out_path / f"{str(run).zfill(4)}_hist.npy", np.array())
