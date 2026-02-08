import torch
from torch.utils.data import Dataset
import os
import numpy as np

import argparse
     
class SEMGDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        semg, force = self.data[idx]
        semg = torch.tensor(semg, dtype=torch.float32)
        force = torch.tensor(force, dtype=torch.float32)
        semg = semg.unsqueeze(0)
        
        return semg, force


parser = argparse.ArgumentParser(description="Data Collection Parameters")

parser.add_argument('--subj_id', type=int, default=-99)
args = parser.parse_args()

# File paths
oldFolder = "raw_data/raw_data_compiled/"
newFolder = "raw_data/windowed_data/"
filename = oldFolder + f"Subject_{args.subj_id}_Compiled.npz"

# Load data
raw_data = np.load(filename)

header = raw_data['header']

semgVals = raw_data['semgVals']
forceIdxs = raw_data['forceIdxs']
forceVals = raw_data['forceVals']

# ReLU the forceVals (set negatives to 0 since negative force just means force plate is lifting off of backplate meaning 0 force)
for i in range(forceVals.size):
    if forceVals[i] < 0:
        forceVals[i] = 0

# Remove any semg values after the final force timestamp (may be up to 90 values, useless since cannot interpolate without low and high point)
maxIdx = forceIdxs[forceIdxs.size - 1]
if semgVals.size > maxIdx + 1:
    semgVals = semgVals[:maxIdx+1]

# Min-max normalization function
def normalize_value(num, min_value, max_value):
    new_num = (num - min_value) / (max_value - min_value)
    return new_num

# Collect max values from header (check data_compile.py for indicies)
max_semg = header[1] 
max_force = header[2]

normalized_semg_temp = []
normalized_force_temp = []

# Normalize semg data with subjects max semg value:
for i in range(semgVals.size):
    normalized_semg_temp.append(normalize_value(semgVals[i], 0, max_semg))

# Normalize force data with subject max force value
for i in range(forceVals.size):
    normalized_force_temp.append(normalize_value(forceVals[i], 0, max_force))

# Set normalized vals to their own array
normalized_semg = np.array(normalized_semg_temp)
normalized_force = np.array(normalized_force_temp)

# Window the data
window_size = 200
stride = 50

# windowed_semg = np.empty(0, dtype=float)
# windowed_force = np.empty(0, dtype=float)

semg_array_list = []
force_list = []
i = 0
while i + window_size - 1 < normalized_semg.size:
    # Add windowed section to windowed_semg
    tempSemg = np.array(normalized_semg[i:i+window_size])
    # print(tempSemg.size)
    # windowed_semg = np.vstack((windowed_semg, tempSemg))
    semg_array_list.append(tempSemg)

    # Add final force value to windowed_force
    # Find the nearest 2 force timestamps to end value

    targetTimestamp = i + window_size - 1
    idx = 0
    # Find the first instance of a timestamp greater than the target value (this means the timestamp before this is less than the target value)
    for j in range(forceIdxs.size):
        
        if forceIdxs[j] >= targetTimestamp:
            idx = j
            break
    
    assert idx > 0 #Ensure idx was assigned

    # If linear interpolation is needed (99% of time)   
    if forceIdxs[idx] != targetTimestamp:
            smallForceVal = normalized_force[idx-1]
            largeForceVal = normalized_force[idx]

            smallTimestamp = forceIdxs[idx-1]
            largeTimestamp = forceIdxs[idx]

            timeBetweenVals = largeTimestamp - smallTimestamp

            slope = (largeForceVal - smallForceVal) / timeBetweenVals

            distanceFromTarget = targetTimestamp - smallTimestamp 

            interpolatedVal = smallForceVal + (slope * distanceFromTarget)

            # windowed_force = np.append(windowed_force, interpolatedVal)
            force_list.append(interpolatedVal)

    # If timestamp aligns perfectly
    else:
        # windowed_force = np.append(windowed_force, normalized_force[idx])
        force_list.append(normalized_force[idx])

    # Increase i by stride
    i += stride

windowed_semg = np.vstack(semg_array_list)
windowed_force = np.array(force_list)


# print(windowed_semg.shape)
# print(windowed_force.shape)

filename = f"processed_data/Subject_{args.subj_id}_Processed"
np.savez(filename, header=header, windowed_semg=windowed_semg, windowed_force=windowed_force)
# np.savetxt('windowSemg', windowed_semg, fmt='%.50f')
# np.savetxt('windowForce', windowed_force, fmt='%.10f')