import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import numpy as np
from mimo_simulator import MIMOSimulator
from model import AutoMIMONet
import matplotlib.pyplot as plt

# Configuration
NR, NT = 8, 32  # Smaller for faster demo, can be scaled to 16/64
SNR_DB = 15
BATCH_SIZE = 32
EPOCHS = 5
NUM_TRIALS = 10 # Number of HPO trials

def get_data_loaders():
    sim = MIMOSimulator(nt=NT, nr=NR, snr_db=SNR_DB)
    
    # Generate Training Data
    X_train, Y_train = sim.get_data(batch_size=2000)
    X_val, Y_val = sim.get_data(batch_size=500)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def objective(trial):
    # Hyperparameters to tune
    hpo_params = {
        'num_blocks': trial.suggest_int('num_blocks', 1, 5),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.4),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    }
    
    model = AutoMIMONet(NR, NT, hpo_params)
    param_count = count_parameters(model)
    
    optimizer = optim.Adam(model.parameters(), lr=hpo_params['lr'])
    criterion = nn.MSELoss()
    
    train_loader, val_loader = get_data_loaders()
    
    # Training Loop for Trial
    model.train()
    for epoch in range(EPOCHS):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    
    mse = val_loss / len(val_loader)
    
    # Return two objectives: [Accuracy (MSE), Complexity (Param Count)]
    return mse, param_count

def run_optimization():
    print("🚀 Starting Multi-Objective Auto-MIMO HPO (MSE vs Model Size)...")
    # We want to minimize both MSE and Parameter Count
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=NUM_TRIALS)
    
    print("\n🏆 Pareto Front (Best Trade-offs):")
    for i, trial in enumerate(study.best_trials):
        print(f"  Trial {i}: MSE={trial.values[0]:.6f}, Params={trial.values[1]}")
        print(f"    Params: {trial.params}")
        
    # Plot Pareto Front
    try:
        fig = optuna.visualization.plot_pareto_front(study, target_names=["MSE", "Param Count"])
        fig.write_image("pareto_front.png")
        print("📈 Saved pareto_front.png")
    except Exception as e:
        print(f"Could not save Pareto plot: {e} (Maybe kaleido is not installed)")
    
    return study.best_trials[0].params # Return the most accurate one for now

if __name__ == "__main__":
    best_params = run_optimization()
    print("\n✅ Multi-Objective Search Complete.")
