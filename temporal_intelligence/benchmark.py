import os
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
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

def get_memory_usage_mb():
    """
    Returns the current Resident Set Size (RSS) memory of the process in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def train_and_evaluate(model_name, model, train_loader, val_loader, epochs=10, lr=1e-3):
    print(f"\nTraining {model_name} on {DEVICE}...")
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics trackers
    epoch_times = []
    peak_memory = 0.0
    initial_memory = get_memory_usage_mb()
    
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        
        # Track memory usage during epoch
        epoch_memory = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Periodically sample memory
            if batch_idx % 10 == 0:
                epoch_memory.append(get_memory_usage_mb())
                
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Track global peak memory
        if len(epoch_memory) > 0:
            peak_memory = max(peak_memory, max(epoch_memory))
            
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Time: {epoch_time:.2f}s")
        
    # Calculate memory footprint attributable to training this model
    memory_footprint = max(0.0, peak_memory - initial_memory)
    avg_epoch_time = np.mean(epoch_times)
    total_train_time = np.sum(epoch_times)
    
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    
    # Class-wise F1 scores
    class_f1s = f1_score(all_targets, all_preds, average=None)
    
    print(f"{model_name} Training Complete!")
    print(f"  • Avg Epoch Time:    {avg_epoch_time:.3f}s")
    print(f"  • Memory Footprint:  {memory_footprint:.2f} MB")
    print(f"  • Accuracy:          {accuracy*100:.2f}%")
    print(f"  • F1 Score (w):      {f1*100:.2f}%")
    
    return {
        "model_name": model_name,
        "avg_epoch_time": avg_epoch_time,
        "total_train_time": total_train_time,
        "memory_footprint_mb": memory_footprint,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "class_f1s": class_f1s
    }

def run_benchmarks(data_dir="../DataEngineering", batch_size=128, epochs=10):
    print("==============================================================================")
    print("            BENCHMARK SUITE: LSTM vs. TRANSFORMER vs. MAMBA                   ")
    print("==============================================================================")
    
    # Load dataset
    dataset = TemporalSequenceDataset(data_dir=data_dir, seed=42)
    
    # Split: 80% Train, 20% Val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset summary:")
    print(f"  • Total samples:      {len(dataset)}")
    print(f"  • Train size:         {train_size}")
    print(f"  • Validation size:    {val_size}")
    print(f"  • Time steps (L):     {dataset[0][0].shape[0]}")
    print(f"  • Ingest channels:    {dataset[0][0].shape[1]}")
    print("------------------------------------------------------------------------------")
    
    # Instantiate models
    # Parameters are kept identical to make it a fair scientific test
    d_model = 128
    n_layers = 2
    d_state = 16  # Specific to Mamba state matrix
    num_classes = 4
    
    lstm_model = LSTMClassifier(
        in_channels=17,
        embedding_dim=d_model,
        num_layers=n_layers,
        num_classes=num_classes
    )
    
    transformer_model = TransformerClassifier(
        in_channels=17,
        embedding_dim=d_model,
        num_layers=n_layers,
        nhead=8,
        num_classes=num_classes
    )
    
    mamba_model = MambaClassifier(
        in_channels=17,
        embedding_dim=d_model,
        num_layers=n_layers,
        d_state=d_state,
        num_classes=num_classes
    )
    
    # Run evaluations
    lstm_results = train_and_evaluate("LSTM", lstm_model, train_loader, val_loader, epochs=epochs)
    transformer_results = train_and_evaluate("Transformer", transformer_model, train_loader, val_loader, epochs=epochs)
    mamba_results = train_and_evaluate("Mamba", mamba_model, train_loader, val_loader, epochs=epochs)
    
    results = [lstm_results, transformer_results, mamba_results]
    
    # Print comparison Markdown Table
    print("\n" + "=" * 80)
    print("                      BENCHMARK PERFORMANCE SUMMARY                           ")
    print("=" * 80)
    print(f"| {'Model Architecture':<20} | {'Val Accuracy':<12} | {'Val F1-Score':<12} | {'Epoch Time (s)':<14} | {'Memory (MB)':<11} |")
    print("| " + "-"*20 + " | " + "-"*12 + " | " + "-"*12 + " | " + "-"*14 + " | " + "-"*11 + " |")
    for r in results:
        print(f"| {r['model_name']:<20} | {r['accuracy']*100:<11.2f}% | {r['f1']*100:<11.2f}% | {r['avg_epoch_time']:<14.3f} | {r['memory_footprint_mb']:<11.1f} |")
    print("=" * 80 + "\n")
    
    # Save chart dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = [r["model_name"] for r in results]
    colors = ["#3498db", "#e74c3c", "#9b59b6"]
    
    # 1. Accuracy Panel
    axes[0, 0].bar(model_names, [r["accuracy"] * 100 for r in results], color=colors, width=0.5)
    axes[0, 0].set_title("Validation Accuracy (%)", fontweight='bold')
    axes[0, 0].set_ylabel("Percentage (%)")
    axes[0, 0].set_ylim(80, 102)
    for idx, r in enumerate(results):
        axes[0, 0].text(idx, r["accuracy"]*100 + 0.5, f"{r['accuracy']*100:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    # 2. Time per Epoch Panel
    axes[0, 1].bar(model_names, [r["avg_epoch_time"] for r in results], color=colors, width=0.5)
    axes[0, 1].set_title("Average Training Time per Epoch (seconds)", fontweight='bold')
    axes[0, 1].set_ylabel("Seconds (s)")
    for idx, r in enumerate(results):
        axes[0, 1].text(idx, r["avg_epoch_time"] + 0.02, f"{r['avg_epoch_time']:.3f}s", ha='center', va='bottom', fontweight='bold')
        
    # 3. Peak Memory Footprint
    axes[1, 0].bar(model_names, [r["memory_footprint_mb"] for r in results], color=colors, width=0.5)
    axes[1, 0].set_title("Peak Training Memory Footprint (MB)", fontweight='bold')
    axes[1, 0].set_ylabel("RAM Memory (MB)")
    for idx, r in enumerate(results):
        axes[1, 0].text(idx, r["memory_footprint_mb"] + 1.0, f"{r['memory_footprint_mb']:.1f} MB", ha='center', va='bottom', fontweight='bold')
        
    # 4. Class-wise F1 Comparison
    x = np.arange(4)
    width = 0.25
    class_labels = ["Kharif Normal", "Kharif Stress", "Rabi Normal", "Rabi Stress"]
    
    axes[1, 1].bar(x - width, results[0]["class_f1s"] * 100, width, label='LSTM', color='#3498db')
    axes[1, 1].bar(x, results[1]["class_f1s"] * 100, width, label='Transformer', color='#e74c3c')
    axes[1, 1].bar(x + width, results[2]["class_f1s"] * 100, width, label='Mamba', color='#9b59b6')
    
    axes[1, 1].set_title("F1-Score Class Breakdown (%)", fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_labels)
    axes[1, 1].set_ylabel("F1 Score (%)")
    axes[1, 1].set_ylim(80, 102)
    axes[1, 1].legend()
    
    plt.suptitle("Agricultural Phenology Sequence Modeling: LSTM vs. Transformer vs. Mamba Benchmarking", fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    output_png = os.path.join(os.path.dirname(__file__), "benchmark_analysis_report.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Benchmark comparative charts dashboard compiled and saved to disk: '{output_png}'\n")

if __name__ == "__main__":
    run_benchmarks(data_dir="./DataEngineering", epochs=10)
