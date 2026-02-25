import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from models.cnn import CNN
from models.clstm import CLSTM
from models.tcn import TCN

import argparse
import logging

import numpy as np

class SEMGDataset(Dataset):
    def __init__(self, semg, force):
        self.semg = torch.tensor(semg, dtype=torch.float32)
        self.force = torch.tensor(force, dtype=torch.float32)
    
    def __len__(self):
        return len(self.force)
    
    def __getitem__(self, idx):
        x = self.semg[idx].unsqueeze(0)
        y = self.force[idx].unsqueeze(0)
        
        return x, y
    

def create_datasets(val_subj, test_subj):
    subject_count = 14 #Constant, update as needed

    #Ensure all params are valid
    assert val_subj != test_subj
    assert val_subj <= subject_count and val_subj >= 1
    assert test_subj <= subject_count and test_subj >= 1

    train_semg = []
    train_force = []
    val_semg = []
    val_force = []
    test_semg = []
    test_force = []

    for i in range(1, subject_count+1):
        # TEMPORARY!!!! REMOVE ONCE SUVJECT 5 DATA COLLECTED AND PROCESSED !!!!!!!!
        if i == 5:
            continue
        filename = f"processed_data/Subject_{i}_Processed.npz"
        data = np.load(filename)

        if i == val_subj:
            val_semg.append(data['windowed_semg'])
            val_force.append(data['windowed_force'])
        elif i == test_subj:
            test_semg.append(data['windowed_semg'])
            test_force.append(data['windowed_force'])
        else:
            train_semg.append(data['windowed_semg'])
            train_force.append(data['windowed_force'])
        
    train_semg = np.vstack(train_semg)
    train_force = np.concatenate(train_force)
    val_semg = np.vstack(val_semg)
    val_force = np.concatenate(val_force)
    test_semg = np.vstack(test_semg)
    test_force = np.concatenate(test_force)

    train_dataset = SEMGDataset(train_semg, train_force)
    val_dataset = SEMGDataset(val_semg, val_force)
    test_dataset = SEMGDataset(test_semg, test_force)

    return train_dataset, val_dataset, test_dataset
parser = argparse.ArgumentParser(description="Training Loop Parameters")

log = logging.Logger(__name__)

log.addHandler(logging.FileHandler('text.txt'))
log.addHandler(logging.StreamHandler())

parser.add_argument('--model', required=True)
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()
    
model = None
if args.model.lower() == 'cnn':
    model = CNN()
elif args.model.lower() == 'clstm' or args.model.lower() == 'c-lstm':
    model = CLSTM(1, 1, 2, 5)
elif args.model.lower() == 'tcn':
    model = TCN(1, 1, [30]*8, 7, 0.0)
assert model != None, "Invalid model name, please enter cnn, clstm, or tcn"

# Main training loop
def train_loop(
        model: nn.Module,
        device: str = 'cuda:0'
):
    if not torch.cuda.is_available():
        print("Fail to use GPU.")
        device = 'cpu'

    train_dataset, val_dataset, test_dataset = create_datasets(3, 6)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    loss_function = nn.MSELoss()
    
    # Epoch loop
    for epoch in range(args.epochs):
        train_running_loss = 0.0
        train_avg_loss = 0.0

        model.train()

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch: {epoch+1} | Loss: {0.000}")

        for batch_id, (semg, force) in progress_bar:
            semg = semg.to(device)
            force = force.to(device)

            optimizer.zero_grad()

            output = model(semg) 
            loss = loss_function(output, force)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_avg_loss = train_running_loss / (batch_id + 1)


            progress_bar.set_description(f"Epoch: {epoch+1} | Loss: {train_avg_loss:.4f}")
            progress_bar.set_postfix(loss=train_avg_loss)

        # Validation
        model.eval()

        val_running_loss = 0.0
        val_avg_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Evaluation | Loss: {0.000}")
            for batch_id, (semg, force) in progress_bar:
                semg = semg.to(device)
                force = force.to(device)

                output = model(semg)

                loss = loss_function(output, force)

                val_running_loss += loss.item()

                val_avg_loss = val_running_loss / (batch_id + 1)
                
                progress_bar.set_description(f"Evaluation | Loss: {val_avg_loss:.4f}")

            scheduler.step(val_avg_loss)


if __name__ == "__main__":

    train_loop(model)
    log.info("YAYYYY")

    


