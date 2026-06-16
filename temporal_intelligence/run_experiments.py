import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from .sequence_loader import TemporalSequenceDataset
from .mamba_model import MambaClassifier
from .baselines import LSTMClassifier, TransformerClassifier

# Aesthetics setup
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

DEVICE = torch.device("cpu")

def train_model(model, train_loader, epochs=8, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def run_temporal_experiments(data_dir="../DataEngineering"):
    print("==============================================================================")
    print("        MAMBA EXPERIMENT RUNNER: RESILIENCE & INTERPRETABILITY ANALYSIS       ")
    print("==============================================================================")
    
    # 1. LOAD NORMAL DATASET & SPLIT
    dataset_normal = TemporalSequenceDataset(data_dir=data_dir, seed=42)
    val_size = int(0.2 * len(dataset_normal))
    train_size = len(dataset_normal) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset_normal, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader_normal = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Train the models on clean normal data
    d_model = 128
    n_layers = 2
    
    print("\n[Phase 1] Training core models on normal clean training data...")
    mamba_model = MambaClassifier(in_channels=17, embedding_dim=d_model, num_layers=n_layers, d_state=16, num_classes=4)
    lstm_model = LSTMClassifier(in_channels=17, embedding_dim=d_model, num_layers=n_layers, num_classes=4)
    trans_model = TransformerClassifier(in_channels=17, embedding_dim=d_model, num_layers=n_layers, nhead=8, num_classes=4)
    
    train_model(mamba_model, train_loader, epochs=8)
    train_model(lstm_model, train_loader, epochs=8)
    train_model(trans_model, train_loader, epochs=8)
    
    # 2. RUN RESILIENCE STUDIES (MISSING TIMESTAMPS & CLOUD CONTAMINATION)
    print("\n[Phase 2] Evaluating resilience under degraded observation conditions...")
    
    # Create evaluation loaders for degraded conditions
    # Missing timestamps tests
    dataset_missing_20 = TemporalSequenceDataset(data_dir=data_dir, missing_rate=0.2, seed=42)
    _, val_dataset_missing_20 = random_split(dataset_missing_20, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader_missing_20 = DataLoader(val_dataset_missing_20, batch_size=128, shuffle=False)
    
    dataset_missing_40 = TemporalSequenceDataset(data_dir=data_dir, missing_rate=0.4, seed=42)
    _, val_dataset_missing_40 = random_split(dataset_missing_40, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader_missing_40 = DataLoader(val_dataset_missing_40, batch_size=128, shuffle=False)
    
    # Cloud contamination tests
    dataset_cloudy_25 = TemporalSequenceDataset(data_dir=data_dir, cloud_contamination=True, cloud_rate=0.25, seed=42)
    _, val_dataset_cloudy_25 = random_split(dataset_cloudy_25, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader_cloudy_25 = DataLoader(val_dataset_cloudy_25, batch_size=128, shuffle=False)
    
    dataset_cloudy_50 = TemporalSequenceDataset(data_dir=data_dir, cloud_contamination=True, cloud_rate=0.50, seed=42)
    _, val_dataset_cloudy_50 = random_split(dataset_cloudy_50, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader_cloudy_50 = DataLoader(val_dataset_cloudy_50, batch_size=128, shuffle=False)
    
    scenarios = [
        ("Clean (Normal)", val_loader_normal),
        ("20% Missing", val_loader_missing_20),
        ("40% Missing", val_loader_missing_40),
        ("25% Clouds", val_loader_cloudy_25),
        ("50% Clouds", val_loader_cloudy_50),
    ]
    
    acc_results = {"Mamba": [], "LSTM": [], "Transformer": []}
    scenario_names = [s[0] for s in scenarios]
    
    for name, loader in scenarios:
        acc_mamba = evaluate_accuracy(mamba_model, loader)
        acc_lstm = evaluate_accuracy(lstm_model, loader)
        acc_trans = evaluate_accuracy(trans_model, loader)
        
        acc_results["Mamba"].append(acc_mamba * 100)
        acc_results["LSTM"].append(acc_lstm * 100)
        acc_results["Transformer"].append(acc_trans * 100)
        
        print(f"Scenario: {name:<15} | Mamba: {acc_mamba*100:.2f}% | LSTM: {acc_lstm*100:.2f}% | Transformer: {acc_trans*100:.2f}%")

    # 3. VISUALIZE SELECTIVE RECURRENCE FOCUS (MAMBA GATING Delta & State Activations)
    print("\n[Phase 3] Extracting Mamba internal gating delta and state activations...")
    # Get a sample Kharif Stressed (Class 1) sample with cloud contamination at month 1
    # We'll feed a batch through the model to retrieve internal activations
    mamba_model.eval()
    sample_dataset = TemporalSequenceDataset(data_dir=data_dir, cloud_contamination=True, cloud_rate=1.0, seed=42)
    sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
    
    target_idx = -1
    for idx, (x, y) in enumerate(sample_loader):
        # Find a Stressed Paddy (Class 1)
        if y.item() == 1:
            target_idx = idx
            sample_x = x.to(DEVICE)
            sample_y = y
            break
            
    # Forward pass to trigger hooks/activations
    with torch.no_grad():
        _ = mamba_model(sample_x)
        
    # Retrieve tracked values from the first Mamba block
    first_block = mamba_model.encoder.layers[0]
    # last_deltas shape: (1, L, E) -> Average over embedding dimensions
    deltas = first_block.last_deltas[0].mean(dim=-1).numpy()
    # last_state_activations shape: (1, L)
    state_acts = first_block.last_state_activations[0].numpy()
    
    # 4. DIMENSIONALITY REDUCTION ON MAMBA SEASON EMBEDDINGS
    print("\n[Phase 4] Generating PCA and t-SNE of Mamba Season Embeddings...")
    mamba_model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader_normal:
            x = x.to(DEVICE)
            season_emb, _ = mamba_model.encoder(x)
            all_embeddings.append(season_emb.cpu().numpy())
            all_labels.extend(y.numpy())
            
    X_emb = np.vstack(all_embeddings)
    y_labels = np.array(all_labels)
    
    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_emb)
    var_exp = pca.explained_variance_ratio_ * 100
    
    # t-SNE projection
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_emb)
    
    # Mapping label numbers to descriptive names
    label_mapping = {
        0: "Kharif Normal (Paddy)",
        1: "Kharif Stressed (Paddy)",
        2: "Rabi Normal (Wheat)",
        3: "Rabi Stressed (Wheat)"
    }
    y_descriptive = [label_mapping[l] for l in y_labels]
    
    # 5. COMPILE PLOT PANELS
    print("\nCompiling experimental dashboard layout...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # PANEL A: Resilience Comparison
    x_scenarios = np.arange(len(scenario_names))
    width = 0.25
    axes[0, 0].bar(x_scenarios - width, acc_results["LSTM"], width, label='LSTM', color='#3498db')
    axes[0, 0].bar(x_scenarios, acc_results["Transformer"], width, label='Transformer', color='#e74c3c')
    axes[0, 0].bar(x_scenarios + width, acc_results["Mamba"], width, label='Mamba', color='#9b59b6')
    axes[0, 0].set_title("Validation Accuracy under Degraded Observations", fontweight='bold')
    axes[0, 0].set_xticks(x_scenarios)
    axes[0, 0].set_xticklabels(scenario_names)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(50, 102)
    axes[0, 0].legend()
    
    # PANEL B: Mamba Selective Recurrence Analysis
    time_steps = ["June\n(Sowing)", "July\n(Veg)", "Aug\n(Veg/Flo)", "Sept\n(Flo)", "Oct\n(Grain)", "Nov\n(Harvest)"]
    ax2 = axes[0, 1]
    # Normalize deltas for plotting
    norm_deltas = deltas / deltas.max()
    norm_states = state_acts / state_acts.max()
    
    ax2.plot(time_steps, norm_deltas, label='Normalized Gating Step $\Delta_t$ (Update Rate)', color='#e67e22', marker='o', linewidth=2.5)
    ax2.plot(time_steps, norm_states, label='Normalized State Norm $||h_t||$ (Memory Size)', color='#9b59b6', marker='s', linewidth=2.5, linestyle='--')
    
    # Draw cloud event (cloud at t=1, July)
    ax2.axvspan(1.0, 1.2, color='#95a5a6', alpha=0.3, label='Cloud Interference (July)')
    ax2.set_title("Mamba Selective Recurrence Focus over Growing Season", fontweight='bold')
    ax2.set_ylabel("Normalized Activation Value")
    ax2.set_xlabel("Phenological Stages (Kharif Horizon)")
    ax2.legend()
    
    # Color palette for manifolds
    palette = {
        "Kharif Normal (Paddy)": "#2ecc71",
        "Kharif Stressed (Paddy)": "#27ae60",
        "Rabi Normal (Wheat)": "#f1c40f",
        "Rabi Stressed (Wheat)": "#d35400"
    }
    
    # PANEL C: PCA of Mamba Season Embeddings
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_descriptive, palette=palette, alpha=0.7, s=20, ax=axes[1, 0])
    axes[1, 0].set_title(f"PCA Projections of Mamba Season Embeddings\n(PC1 Var: {var_exp[0]:.1f}% | PC2 Var: {var_exp[1]:.1f}%)", fontweight='bold')
    axes[1, 0].set_xlabel("Principal Component 1")
    axes[1, 0].set_ylabel("Principal Component 2")
    axes[1, 0].legend(title="Phenology Classes", fontsize=8, title_fontsize=9)
    
    # PANEL D: t-SNE of Mamba Season Embeddings
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_descriptive, palette=palette, alpha=0.7, s=20, ax=axes[1, 1])
    axes[1, 1].set_title("t-SNE Manifold of Mamba Season Embeddings\n(Crop & Stress State Clusters)", fontweight='bold')
    axes[1, 1].set_xlabel("t-SNE Axis 1")
    axes[1, 1].set_ylabel("t-SNE Axis 2")
    axes[1, 1].legend().remove() # Clear legend duplication
    
    plt.suptitle("Mamba SSM Research: Temporal Crop-Season Modeling and Observational Noise Resilience", fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    output_png = os.path.join(os.path.dirname(__file__), "mamba_experiment_results.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Experimental analysis completed. Saved charts to: '{output_png}'")

if __name__ == "__main__":
    run_temporal_experiments(data_dir="./DataEngineering")
