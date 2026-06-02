import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mimo_simulator import MIMOSimulator
from model import AutoMIMONet
import json
import os

# Best Parameters found by HPO (Example - these would usually be passed from train_hpo.py)
BEST_PARAMS = {
    'num_blocks': 3,
    'hidden_dim': 64,
    'dropout': 0.1,
    'lr': 0.001
}

NR, NT = 8, 32
SNR_DB = 15
TOTAL_SAMPLES = 5000
EPOCHS = 20

def train_and_save():
    print("🔥 Training the Final 'Auto-MIMO' Model...")
    
    # 1. Prepare Large Dataset
    sim = MIMOSimulator(nt=NT, nr=NR, snr_db=SNR_DB)
    X, Y = sim.get_data(batch_size=TOTAL_SAMPLES)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Initialize Model
    model = AutoMIMONet(NR, NT, BEST_PARAMS)
    optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS['lr'])
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(loader):.6f}")
            
    # 4. Save Model and Metadata
    torch.save(model.state_dict(), 'mimo_model.pth')
    with open('model_config.json', 'w') as f:
        json.dump(BEST_PARAMS, f)
        
    print("\n✅ Model saved as 'mimo_model.pth'")
    print("✅ Configuration saved as 'model_config.json'")

if __name__ == "__main__":
    train_and_save()
