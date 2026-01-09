import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_preprocess import load_data, SEMGDataset
    

# Main training loop
def train_loop(
        model: nn.Module,
        num_epochs: int = 100,
        device: str = 'cuda:0'
):
    if not torch.cuda.is_available():
        print("Fail to use GPU.")
        device = 'cpu'

    
    raw_train_data, raw_val_data, raw_test_data = load_data("test_data.npz", 2, 3, 20, 5)

    train_dataset = SEMGDataset(raw_train_data)
    val_dataset = SEMGDataset(raw_val_data)
    test_dataset = SEMGDataset(raw_test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    loss_function = nn.MSELoss()
    
    # Epoch loop
    for epoch in range(num_epochs):
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




    


