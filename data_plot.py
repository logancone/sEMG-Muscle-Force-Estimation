import numpy as np
import matplotlib.pyplot as plt


data = np.load("raw_data/raw_data_compiled/Subject_-101_Compiled.npz")

plt.plot(data["semgVals"])
plt.show()