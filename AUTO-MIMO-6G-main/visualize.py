import torch
import numpy as np
import matplotlib.pyplot as plt
from mimo_simulator import MIMOSimulator
from model import AutoMIMONet

def visualize_estimation(model, nr, nt, snr_db):
    sim = MIMOSimulator(nt=nt, nr=nr, snr_db=snr_db)
    X, H_true = sim.get_data(batch_size=1)
    
    model.eval()
    with torch.no_grad():
        H_pred = model(torch.from_numpy(X)).numpy()
    
    # Extract Real part of the first sample
    h_true_real = H_true[0, 0, :, :]
    h_pred_real = H_pred[0, 0, :, :]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot True H
    im1 = axes[0].imshow(h_true_real, aspect='auto', cmap='viridis')
    axes[0].set_title("True Channel (Real)")
    fig.colorbar(im1, ax=axes[0])
    
    # Plot Predicted H
    im2 = axes[1].imshow(h_pred_real, aspect='auto', cmap='viridis')
    axes[1].set_title("AI Estimated Channel")
    fig.colorbar(im2, ax=axes[1])
    
    # Plot Error
    error = np.abs(h_true_real - h_pred_real)
    im3 = axes[2].imshow(error, aspect='auto', cmap='magma')
    axes[2].set_title("Estimation Error")
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('estimation_comparison.png')
    print("🖼️ Saved estimation_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Example usage (would typically load a saved model)
    NR, NT = 8, 32
    # Mock model for demonstration
    model = AutoMIMONet(NR, NT) 
    visualize_estimation(model, NR, NT, 20)
