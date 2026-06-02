# Auto-MIMO 6G: Self-Optimizing Deep Learning for Intelligent Channel Estimation

## 🌟 Overview
**Auto-MIMO 6G** is a next-generation wireless communication project that uses **Automated Machine Learning (AutoML)** to solve the problem of Channel Estimation in Massive MIMO systems. 

Unlike traditional estimators, this system uses **Multi-Objective Hyperparameter Optimization** to find the perfect balance between signal reconstruction accuracy and computational efficiency—a critical requirement for battery-powered 6G devices.

## 🚀 Key Features
- **Intelligent Channel Estimation**: Uses a Deep Residual Network (ResNet) to reconstruct high-fidelity channel matrices from noisy pilot signals.
- **Pareto-Optimal Search**: Automatically tunes neural architectures to minimize Bit Error Rate (BER) while simultaneously reducing model size (Parameter Count).
- **Physics-Informed Simulation**: Includes a custom Rayleigh Fading simulator to generate synthetic 6G-standard data.
- **Hardware-Aware Design**: Optimized for low-latency inference on edge devices.

## 📁 System Architecture
1. **`mimo_simulator.py`**: Synthetic radio environment generator.
2. **`model.py`**: The Deep-MIMO Residual Architecture.
3. **`train_hpo.py`**: The Multi-Objective Optimizer (using Optuna).
4. **`final_train.py`**: Production-ready trainer and model exporter.
5. **`visualize.py`**: Performance evaluation and constellation mapping.

## 🛠️ Getting Started

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Hyperparameter Optimization
Discover the best model architecture for your specific SNR environment:
```bash
python train_hpo.py
```

### 3. Production Training
Train the final model with discovered parameters for high precision:
```bash
python final_train.py
```

### 4. Visualize Results
Generate heatmaps of estimation errors:
```bash
python visualize.py
```

## 📊 Evaluation Metrics
- **NMSE**: Normalized Mean Square Error of the Channel Matrix.
- **Efficiency**: Model parameter count and inference time.
- **Robustness**: Performance across varying Signal-to-Noise Ratios (SNR).

---
*Created as a unique intersection of AI and Wireless Communications.*
