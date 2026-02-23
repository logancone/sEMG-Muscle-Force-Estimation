import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Data Collection Parameters")

parser.add_argument('--subj_id', '-id', type=int, required=True)

parser.add_argument('--age', '-a', type=int, required=True)
parser.add_argument('--sex', '-s', type=int, required=True) # 0 for male, 1 for female
parser.add_argument('--height', '-hi', type=int, required=True) 
parser.add_argument('--weight', '-w', type=int, required=True)
parser.add_argument('--lifts_weights', '-lw', type=bool, required=True)
parser.add_argument('--activity_level', '-al', type=int, required=True)
args = parser.parse_args()

trial_id = 1
chunk_id = 0

# Collect MVC Values
max_semg = 0
max_force = 0

# Folder paths
mvcFolder = "raw_data/raw_mvc_trials/"
chunkFolder = "raw_data/raw_data_chunks/"
compileFolder = "raw_data/raw_data_compiled/"
windowFolder = "processed_data/"

# File path
mvcFilename = mvcFolder + f"Subject_{args.subj_id}_MVC_Trial_{trial_id}.npz"

# While target mvc file exists
while os.path.isfile(mvcFilename):
    # Print filename that's being compiled
    print(mvcFilename)

    # Load data
    data = np.load(mvcFilename)

    # Determine max semg value
    if max(data['npSemgVals']) > max_semg:
        max_semg = max(data['npSemgVals'])

    # Determine max force value
    if max(data['npForceVals']) > max_force:
        max_force = max(data['npForceVals'])

    # Increase trial
    trial_id += 1

    # Update filename
    mvcFilename = mvcFolder + f"Subject_{args.subj_id}_MVC_Trial_{trial_id}.npz"

# Print maximums
print(f"MVC: {max_force} | Max sEMG: {max_semg}")

# Create the header array for file with form [subj_id, max_semg, max_force, age, sex, height, weight, lifts_weights, activity_level]
header = np.array([args.subj_id, max_semg, max_force, args.age, args.sex, args.height, args.weight, args.lifts_weights, args.activity_level])

# Normal Data

# Make empty ndarrays
semgVals = np.empty(0, dtype=int)
forceIdxs = np.empty(0, dtype=int)
forceVals = np.empty(0, dtype=int)

# File path
filename = chunkFolder + f"Subject_{args.subj_id}_Chunk_{chunk_id}.npz"


# While target file exists
while os.path.isfile(filename):
    # Print filename that's being compiled
    print(filename)

    # Load data and add to ndarrays
    data = np.load(filename)
    semgVals = np.concatenate((semgVals, data['npSemgVals']))
    forceIdxs = np.concatenate((forceIdxs, data['npForceIdxs']))
    forceVals = np.concatenate((forceVals, data['npForceVals']))

    # Increase chunk_id
    chunk_id += 1

    # Update filename
    filename = chunkFolder + f"Subject_{args.subj_id}_Chunk_{chunk_id}.npz"

# Save as one big compiled file
newFilename = compileFolder + f"Subject_{args.subj_id}_Compiled"

np.savez(newFilename, header=header, semgVals=semgVals, forceIdxs=forceIdxs, forceVals=forceVals)

# Begin preprocessing the data ------------------------------------------------------------------------------------------------

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
    if max_value - min_value == 0:
        print("No range!!")
        new_num = 0
    else:
        new_num = (num - min_value) / (max_value - min_value)
    return new_num

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

# Store windowed vals in python lists
semg_array_list = []
force_list = []
i = 0
while i + window_size - 1 < normalized_semg.size:
    # Slice windowed section
    tempSemg = np.array(normalized_semg[i:i+window_size])

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

            force_list.append(interpolatedVal)

    # If timestamp aligns perfectly
    else:
        force_list.append(normalized_force[idx])

    # Increase i by stride
    i += stride

# Compile into ndarrays
windowed_semg = np.vstack(semg_array_list)
windowed_force = np.array(force_list)

# Save windowed data into one file
filename = windowFolder + f"Subject_{args.subj_id}_Processed"

np.savez(filename, header=header, windowed_semg=windowed_semg, windowed_force=windowed_force)

print(f"Full preprocess of Subject {args.subj_id} sucessful!")








