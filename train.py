import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from models.cnn import CNN
from models.clstm import CLSTM
from models.tcn import TCN

import argparse
import logging

import numpy as np

from datetime import datetime

from pathlib import Path

import random

# Constants
subject_count = 17
train_epochs = 100
tl_epochs = 5
patience = 25
min_delta = 1e-5

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
    #Ensure all params are valid
    assert val_subj != test_subj
    assert val_subj <= subject_count and val_subj >= 1
    assert test_subj <= subject_count and test_subj >= 1

    train_semg = []
    train_force = []
    val_semg = []
    val_force = []
    tl_semg = []
    tl_force = []
    test_semg = []
    test_force = []

    for i in range(1, subject_count+1):
        filename = f"processed_data/Subject_{i}_Processed.npz"
        data = np.load(filename)

        if i == val_subj:
            val_semg.append(data['windowed_semg'])
            val_force.append(data['windowed_force'])
        elif i == test_subj:
            # First 0-296 windows (297 total) include data from first 15sec of data collection. Discard 3 for overlap, start eval data at window 300
            tl_semg.append(data['windowed_semg'][:297]) #Final index exclusive
            tl_force.append(data['windowed_force'][:297])

            test_semg.append(data['windowed_semg'][300:])
            test_force.append(data['windowed_force'][300:])

            # tl_semg.append(data['windowed_semg'][:1000]) #Final index exclusive
            # tl_force.append(data['windowed_force'][:1000])

            # test_semg.append(data['windowed_semg'][1004:])
            # test_force.append(data['windowed_force'][1004:])
        else:
            train_semg.append(data['windowed_semg'])
            train_force.append(data['windowed_force'])
        
    train_semg = np.vstack(train_semg)
    train_force = np.concatenate(train_force)
    val_semg = np.vstack(val_semg)
    val_force = np.concatenate(val_force)
    tl_semg = np.vstack(tl_semg)
    tl_force = np.concatenate(tl_force)
    test_semg = np.vstack(test_semg)
    test_force = np.concatenate(test_force)

    train_dataset = SEMGDataset(train_semg, train_force)
    val_dataset = SEMGDataset(val_semg, val_force)
    tl_dataset = SEMGDataset(tl_semg, tl_force)
    test_dataset = SEMGDataset(test_semg, test_force)

    return train_dataset, val_dataset, tl_dataset, test_dataset

# Main training loop
def train_loop(
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        logger: logging.Logger,
        writer: SummaryWriter,
        save_dir: Path,
        device: str = 'cuda:0'
):
    if not torch.cuda.is_available():
        print("Fail to use GPU.")
        device = 'cpu'

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience=5)

    loss_function = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    
    # Epoch loop
    for epoch in range(train_epochs):
        train_running_loss = 0.0
        train_avg_loss = 0.0

        model.train()

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch: {epoch+1}")

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
            
            progress_bar.set_description(f"Epoch: {epoch+1}")
            progress_bar.set_postfix(loss=train_avg_loss)

        logger.info(f"Epoch: {epoch+1} | Train Loss: {train_avg_loss:.5f}")

        writer.add_scalar("Loss/train", train_avg_loss, epoch)

        # Validation
        model.eval()

        val_running_loss = 0.0
        val_avg_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Validation")
            for batch_id, (semg, force) in progress_bar:
                semg = semg.to(device)
                force = force.to(device)

                output = model(semg)

                loss = loss_function(output, force)

                val_running_loss += loss.item()
                val_avg_loss = val_running_loss / (batch_id + 1)
                
                progress_bar.set_postfix(loss=val_avg_loss)

            logger.info(f"Validation | Loss: {val_avg_loss:.5f}")
            writer.add_scalar("Loss/val", val_avg_loss, epoch)

            

            scheduler.step(val_avg_loss)
        
        if val_avg_loss < best_val_loss - min_delta:
            best_val_loss = val_avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/model_pre_tl.pt")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered on epoch {epoch+1}")
            break



def fine_tune(
        model: nn.Module,
        tl_dataloader: DataLoader,
        logger: logging.Logger,
        writer: SummaryWriter,
        save_dir: Path,
        device: str = 'cuda:0',
):
   
    if not torch.cuda.is_available():
        print("Fail to use GPU.")
        device = 'cpu'

    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    loss_function = nn.MSELoss()

    best_tl_loss = float('inf')
    
    # Epoch loop
    for epoch in range(tl_epochs):
        tl_running_loss = 0.0
        tl_avg_loss = 0.0

        # model.train()

        progress_bar = tqdm(enumerate(tl_dataloader), total=len(tl_dataloader), desc=f"Epoch: {epoch+1}")

        for batch_id, (semg, force) in progress_bar:
            semg = semg.to(device)
            force = force.to(device)

            optimizer.zero_grad()

            output = model(semg) 
            loss = loss_function(output, force)

            loss.backward()
            optimizer.step()

            tl_running_loss += loss.item()
            tl_avg_loss = tl_running_loss / (batch_id + 1)
            
            progress_bar.set_description(f"Epoch: {epoch+1}")
            progress_bar.set_postfix(loss=tl_avg_loss)

        logger.info(f"Epoch: {epoch+1} | TL Loss: {tl_avg_loss:.5f}")

        writer.add_scalar("Loss/tl", tl_avg_loss, epoch)

        if tl_avg_loss < best_tl_loss:
            best_tl_loss = tl_avg_loss
            torch.save(model.state_dict(), f"{save_dir}/model_post_tl.pt")


def evaluate(
        model: nn.Module,
        test_dataloader: DataLoader,
        logger: logging.Logger,
        writer: SummaryWriter,
        device: str = 'cuda:0'
):
    if not torch.cuda.is_available():
        print("Fail to use GPU.")
        device = 'cpu'

    model = model.to(device)

    

    loss_function = nn.MSELoss()

    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Testing")
        test_running_loss = 0.0
        test_avg_loss = 0.0

        

        for batch_id, (semg, force) in progress_bar:
            semg = semg.to(device)
            force = force.to(device)

            output = model(semg)

            loss = loss_function(output, force)

            test_running_loss += loss.item()
            test_avg_loss = test_running_loss / (batch_id + 1)
            
            progress_bar.set_postfix(loss=test_avg_loss)

        logger.info(f"Test | Loss: {test_avg_loss:.5f}")
        writer.add_scalar("Loss/test", test_avg_loss)

def train_cnn(test_id, val_id):
    # Establish logger and writer
    folder_path = Path(f"logs/{datetime.strftime(datetime.now(), '%Y-%m-%d__%H-%M-%S')}_CNN")
    folder_path.mkdir()
    model_logger = logging.Logger(__name__)
    model_logger.addHandler(logging.FileHandler(f"{folder_path}/logger.log"))
    model_logger.addHandler(logging.StreamHandler())
    writer = SummaryWriter(f"{folder_path}/writer")

    # Create datasets and dataloaders
    train_dataset, val_dataset, tl_dataset, test_dataset = create_datasets(val_id, test_id)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    tl_dataloader = DataLoader(tl_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Log current model and subj info
    model_logger.info("CNN")
    model_logger.info(f"Test Subject Id: {test_id} | Val Subject Id: {val_id}")

    # Create the initial model, and begin training
    model_before_tl = CNN()
    train_loop(model_before_tl, train_dataloader, val_dataloader, model_logger, writer, folder_path)
    model_before_tl.load_state_dict(torch.load(f"{folder_path}/model_pre_tl.pt"))

    # Clone the trained model by creating a new model and loading the other model's params
    model_after_tl = CNN()
    model_after_tl.load_state_dict(model_before_tl.state_dict())

    # Freeze conv layers for tl
    for param in model_after_tl.conv_1.parameters():
        param.requires_grad = False
    for param in model_after_tl.conv_2.parameters():
        param.requires_grad = False

    # Set to eval and train to ensure proper params update (batchnorm has weird behavior)
    model_after_tl.conv_1.eval()
    model_after_tl.conv_2.eval()
    model_after_tl.fcs.train()

    # Fine tune cloned model on tl data
    fine_tune(model_after_tl, tl_dataloader, model_logger, writer, folder_path)
    model_after_tl.load_state_dict(torch.load(f"{folder_path}/model_post_tl.pt"))

    # Evaluate both models on same test data
    model_logger.info("Before TL:")
    evaluate(model_before_tl, test_dataloader, model_logger, writer)
    model_logger.info("After TL:")
    evaluate(model_after_tl, test_dataloader, model_logger, writer)

def train_clstm(test_id, val_id):
    # Establish logger and writer
    folder_path = Path(f"logs/{datetime.strftime(datetime.now(), '%Y-%m-%d__%H-%M-%S')}_CLSTM")
    folder_path.mkdir()
    model_logger = logging.Logger(__name__)
    model_logger.addHandler(logging.FileHandler(f"{folder_path}/logger.log"))
    model_logger.addHandler(logging.StreamHandler())
    writer = SummaryWriter(f"{folder_path}/writer")

    # Create datasets and dataloaders
    train_dataset, val_dataset, tl_dataset, test_dataset = create_datasets(val_id, test_id)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    tl_dataloader = DataLoader(tl_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Log current model and subj info
    model_logger.info("C-LSTM")
    model_logger.info(f"Test Subject Id: {test_id} | Val Subject Id: {val_id}")

    # Create the initial model, and begin training
    model_before_tl = CLSTM(2, 5)
    train_loop(model_before_tl, train_dataloader, val_dataloader, model_logger, writer, folder_path)
    model_before_tl.load_state_dict(torch.load(f"{folder_path}/model_pre_tl.pt"))

    # Clone the trained model by creating a new model and loading the other model's params
    model_after_tl = CLSTM(2, 5)
    model_after_tl.load_state_dict(model_before_tl.state_dict())

    # Freeze conv layers for tl
    for param in model_after_tl.conv_1.parameters():
        param.requires_grad = False
    for param in model_after_tl.conv_2.parameters():
        param.requires_grad = False
    for param in model_after_tl.conv_3.parameters():
        param.requires_grad = False

    # Set to eval and train to ensure proper params update (batchnorm has weird behavior)
    model_after_tl.conv_1.eval()
    model_after_tl.conv_2.eval()
    model_after_tl.conv_3.eval()
    model_after_tl.fcs.train()

    # Fine tune cloned model on tl data
    fine_tune(model_after_tl, tl_dataloader, model_logger, writer, folder_path)
    model_after_tl.load_state_dict(torch.load(f"{folder_path}/model_post_tl.pt"))

    # Evaluate both models on same test data
    model_logger.info("Before TL:")
    evaluate(model_before_tl, test_dataloader, model_logger, writer)
    model_logger.info("After TL:")
    evaluate(model_after_tl, test_dataloader, model_logger, writer)

def train_tcn(test_id, val_id):
    # Establish logger and writer
    folder_path = Path(f"logs/{datetime.strftime(datetime.now(), '%Y-%m-%d__%H-%M-%S')}_TCN")
    folder_path.mkdir()
    model_logger = logging.Logger(__name__)
    model_logger.addHandler(logging.FileHandler(f"{folder_path}/logger.log"))
    model_logger.addHandler(logging.StreamHandler())
    writer = SummaryWriter(f"{folder_path}/writer")

    # Create datasets and dataloaders
    train_dataset, val_dataset, tl_dataset, test_dataset = create_datasets(val_id, test_id)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    tl_dataloader = DataLoader(tl_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Log current model and subj info
    model_logger.info("TCN")
    model_logger.info(f"Test Subject Id: {test_id} | Val Subject Id: {val_id}")

    # Create the initial model, and begin training
    model_before_tl = TCN(1, 1, [30]*8, 7, 0.0)
    train_loop(model_before_tl, train_dataloader, val_dataloader, model_logger, writer, folder_path)
    model_before_tl.load_state_dict(torch.load(f"{folder_path}/model_pre_tl.pt"))

    # Clone the trained model by creating a new model and loading the other model's params
    model_after_tl = TCN(1, 1, [30]*8, 7, 0.0)
    model_after_tl.load_state_dict(model_before_tl.state_dict())

    # Freeze conv layers for tl
    for param in model_after_tl.tcn.parameters():
        param.requires_grad = False

    # Set to eval and train to ensure proper params update (batchnorm has weird behavior)
    model_after_tl.tcn.eval()
    model_after_tl.linear.train()

    # Fine tune cloned model on tl data
    fine_tune(model_after_tl, tl_dataloader, model_logger, writer, folder_path)
    model_after_tl.load_state_dict(torch.load(f"{folder_path}/model_post_tl.pt"))

    # Evaluate both models on same test data
    model_logger.info("Before TL:")
    evaluate(model_before_tl, test_dataloader, model_logger, writer)
    model_logger.info("After TL:")
    evaluate(model_after_tl, test_dataloader, model_logger, writer)

def full_train_loop():
    for i in range(subject_count, 0, -1): #TEMP REVERSE
        val_id = random.randint(1, subject_count)
        while val_id == i or val_id == 5:
            val_id = random.randint(1, subject_count)
        
        print(f"Test Subject Id: {i} | Val Subject Id: {val_id}")
        train_cnn(i, val_id)
        train_clstm(i, val_id)
        train_tcn(i, val_id)

def train_to_failure():
    temp = Path.read_text(Path('subj_combos.txt')).splitlines()
    avail_combos = []
    for t in temp:
        avail_combos.append(eval(t))
    
    while len(avail_combos) > 0:
        combo_id = random.randint(0, len(avail_combos)//2)
        tup = avail_combos.pop(combo_id)

        print(f"Test Subject Id: {tup[0]} | Val Subject Id: {tup[1]}")
        train_cnn(tup[0], tup[1])
        train_clstm(tup[0], tup[1])
        train_tcn(tup[0], tup[1])


if __name__ == "__main__":
    # full_train_loop()
    # train_cnn(17, 8)
    train_to_failure()
