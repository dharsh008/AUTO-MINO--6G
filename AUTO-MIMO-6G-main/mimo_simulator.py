import numpy as np
from scipy import special

class MIMOSimulator:
    def __init__(self, nt=64, nr=16, snr_db=20):
        """
        nt: Number of transmit antennas
        nr: Number of receive antennas
        snr_db: Signal-to-Noise Ratio in dB
        """
        self.nt = nt
        self.nr = nr
        self.snr_db = snr_db

    def generate_channel(self, batch_size=1):
        """
        Generates a Rayleigh Fading Channel Matrix H
        H has complex Gaussian entries with zero mean and unit variance.
        """
        # Complex Channel H ~ CN(0, I)
        h_real = np.random.normal(0, np.sqrt(0.5), (batch_size, self.nr, self.nt))
        h_imag = np.random.normal(0, np.sqrt(0.5), (batch_size, self.nr, self.nt))
        return h_real + 1j * h_imag

    def get_data(self, batch_size=1000, pilot_symbols=None):
        """
        Simulates the transmission of pilot symbols through the channel.
        Y = HX + N
        """
        if pilot_symbols is None:
            # Simple Identity matrix as pilot for simplicity in H estimation
            # In real systems, this would be a sequence.
            pilot_symbols = np.eye(self.nt)
        
        H = self.generate_channel(batch_size)
        
        # Expand pilot for batch
        X = np.tile(pilot_symbols, (batch_size, 1, 1))
        
        # Y = H @ X
        Y_clean = np.matmul(H, X)
        
        # Add AWGN
        snr_linear = 10**(self.snr_db / 10.0)
        power_h = np.mean(np.abs(H)**2)
        noise_std = np.sqrt(power_h / snr_linear)
        
        noise_real = np.random.normal(0, noise_std/np.sqrt(2), Y_clean.shape)
        noise_imag = np.random.normal(0, noise_std/np.sqrt(2), Y_clean.shape)
        noise = noise_real + 1j * noise_imag
        
        Y_noisy = Y_clean + noise
        
        # Format for Neural Network: [Batch, Channels(Real/Imag), Nr, Nt]
        # We'll flatten the inputs for a simple MLP or keep 2D for CNN
        
        # Return (Inputs: Noisy Y, Targets: True H)
        # Convert complex to 2-channel real (Real, Imag)
        y_input = np.stack([Y_noisy.real, Y_noisy.imag], axis=1) # [B, 2, Nr, Nt]
        h_target = np.stack([H.real, H.imag], axis=1)           # [B, 2, Nr, Nt]
        
        return y_input.astype(np.float32), h_target.astype(np.float32)

if __name__ == "__main__":
    sim = MIMOSimulator(nt=8, nr=4, snr_db=10)
    Y, H = sim.get_data(batch_size=5)
    print(f"Input Shape (Y): {Y.shape}")
    print(f"Target Shape (H): {H.shape}")
