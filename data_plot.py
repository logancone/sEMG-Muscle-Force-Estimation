import numpy as np
import matplotlib.pyplot as plt


# data = np.load("raw_data/raw_data_compiled/Subject_-101_Compiled.npz")
data = np.load("processed_data/Subject_1_Processed.npz")

print(data['windowed_semg'].shape)

# plt.plot(data["semgVals"])
# plt.show()