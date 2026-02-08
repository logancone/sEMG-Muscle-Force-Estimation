import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Data Collection Parameters")

parser.add_argument('--subj_id', type=int, required=True)

parser.add_argument('--age', type=int, required=True)
parser.add_argument('--sex', type=int, required=True) # 0 for male, 1 for female
parser.add_argument('--height', type=float, required=True) 
parser.add_argument('--weight', type=float, required=True)
parser.add_argument('--lifts_weights', type=bool, required=True)
parser.add_argument('--activity_level', type=int, required=True)
args = parser.parse_args()

trial_id = 1
chunk_id = 0

# Collect MVC Values
max_semg = 0
max_force = 0

# File paths
mvcFolder = "raw_data/raw_mvc_trials/"
mvcFilename = mvcFolder + f"Subject_{args.subj_id}_MVC_Trial_{trial_id}.npz"
# print(mvcFilename)
# While target file exists
while os.path.isfile(mvcFilename):
    print(mvcFilename)

    # Load data and add to ndarrays
    data = np.load(mvcFilename)

    # Determine max semg value
    if max(data['npSemgVals']) > max_semg:
        max_semg = max(data['npSemgVals'])

    # Determine max force value
    if max(data['npForceVals']) > max_force:
        max_force = max(data['npForceVals'])

    trial_id += 1

    # Update filename
    mvcFilename = mvcFolder + f"Subject_{args.subj_id}_MVC_Trial_{trial_id}.npz"

print(f"MVC: {max_force} | Max sEMG: {max_semg}")

# Create the header array for file with form [subj_id, max_semg, max_force, age, sex, height, weight, lifts_weights, activity_level]
header = np.array([args.subj_id, max_semg, max_force, args.age, args. sex, args.height, args.weight, args.lifts_weights, args.activity_level])

# Normal Data

# Reset chunk_id
chunk_id = 0

# Make empty ndarrays
semgVals = np.empty(0, dtype=int)
forceIdxs = np.empty(0, dtype=int)
forceVals = np.empty(0, dtype=int)

# File paths
oldFolder = "raw_data/raw_data_chunks/"
newFolder = "raw_data/raw_data_compiled/"
filename = oldFolder + f"Subject_{args.subj_id}_Chunk_{chunk_id}.npz"


# While target file exists
while os.path.isfile(filename):
    print(filename)

    # Load data and add to ndarrays
    data = np.load(filename)
    semgVals = np.concatenate((semgVals, data['npSemgVals']))
    forceIdxs = np.concatenate((forceIdxs, data['npForceIdxs']))
    forceVals = np.concatenate((forceVals, data['npForceVals']))

    chunk_id += 1

    # Update filename
    filename = oldFolder + f"Subject_{args.subj_id}_Chunk_{chunk_id}.npz"

# Save as one big file
newFilename = newFolder + f"Subject_{args.subj_id}_Compiled"
np.savez(newFilename, header=header, semgVals=semgVals, forceIdxs=forceIdxs, forceVals=forceVals)

# np.savetxt('header', header, fmt='%i')
# np.savetxt('semgVals', semgVals, fmt='%i')
# np.savetxt('forceIdxs', forceIdxs, fmt='%i')
# np.savetxt('forceVals', forceVals, fmt='%i')






