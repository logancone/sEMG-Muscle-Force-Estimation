import torch
from torch.utils.data import Dataset
import numpy as np



def load_data(data_path, validation_subject, left_out_subject, window_size, stride):
        raw_data = np.load(data_path) # Load raw data from file

        #Ensure left_out_subject and validation_subject is valid number (less than the total number of subjects) and that they arent the same number
        assert left_out_subject <= len(raw_data) and validation_subject <= len(raw_data) and left_out_subject != validation_subject
        train_data = []
        val_data = []
        test_data = []
        sub_num = 1
        for subj in raw_data:
            semg, force = raw_data[subj] # Extract sEmg and force data from raw data
            
            # Split data into windows and assign to train and test data depending on if it is the chosen subject or not
            if sub_num != left_out_subject and sub_num != validation_subject: 
                for i in range(int(len(semg)/stride)):
                    train_data.append((semg[i:i+window_size], force[i+window_size - 1]))
            elif sub_num != left_out_subject:
                for i in range(int(len(semg)/stride)):
                    val_data.append((semg[i:i+window_size], force[i+window_size - 1]))
            else:
                 for i in range(int(len(semg)/stride)):
                    test_data.append((semg[i:i+window_size], force[i+window_size - 1]))
        
        return train_data, val_data, test_data
     
     
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
         
        
