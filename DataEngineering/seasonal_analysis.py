import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import dimensional models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Import our new API
import embedding_api

# Set clean academic styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16
})

def run_indian_season_experiments(lat=16.5062, lon=80.6480):
    print("==============================================================================")
    print("      EXPERIMENTAL SUITE: INDIAN CROP SEASON SEPARABILITY (KEERTHANA)         ")
    print("==============================================================================")
    
    # --------------------------------------------------------------------------
    # STEP 1: EXTRACTION & ENCODING FOR KHARIF (JUN - NOV)
    # --------------------------------------------------------------------------
    print("\n[Phase 1] Processing Kharif (Monsoon) Time-Series...")
    # Using the API which handles GEE extraction + Presto encoding
    # Start Month: June -> 5
    kharif_embeddings = embedding_api.get_farm_embeddings(
        lat=lat,
        lon=lon,
        start_date="2024-06-01",
        end_date="2024-11-30",
        start_month=5
    )
    print(f"-> Kharif Matrix Shape: {kharif_embeddings.shape}")

    # --------------------------------------------------------------------------
    # STEP 2: EXTRACTION & ENCODING FOR RABI (DEC - MAY)
    # --------------------------------------------------------------------------
    print("\n[Phase 2] Processing Rabi (Winter) Time-Series...")
    # Start Month: December -> 11
    rabi_embeddings = embedding_api.get_farm_embeddings(
        lat=lat,
        lon=lon,
        start_date="2024-12-01",
        end_date="2025-05-31",
        start_month=11
    )
    print(f"-> Rabi Matrix Shape: {rabi_embeddings.shape}")

    # --------------------------------------------------------------------------
    # STEP 3: CONSERVE, MERGE & SANITIZE NUMERICAL STACKS
    # --------------------------------------------------------------------------
    X = np.vstack([kharif_embeddings, rabi_embeddings])
    num_pixels = kharif_embeddings.shape[0]
    labels = np.array(["Kharif (Monsoon)"] * num_pixels + ["Rabi (Winter)"] * num_pixels)
    
    # Clean up NaNs
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"\n[Data Cleaning] Sanitizing {nan_count} NaN outputs inside the matrix.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nCombined Dataset Array Ready for Dimensional Models: {X.shape}")

    # --------------------------------------------------------------------------
    # STEP 4: DIMENSIONALITY REDUCTION EXPERIMENTS
    # --------------------------------------------------------------------------
    print("Running Experiment A: Principal Component Analysis (PCA)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100
    
    print("Running Experiment B: t-Distributed Stochastic Neighbor Embedding (t-SNE)...")
    tsne = TSNE(n_components=2, perplexity=min(40, X.shape[0] - 1), max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    print("Running Experiment C: Uniform Manifold Approximation (UMAP)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)

    # --------------------------------------------------------------------------
    # STEP 5: VISUALIZATION REPORT COMPILATION
    # --------------------------------------------------------------------------
    print("\nRendering comparative analytical chart panels...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    palette_colors = {"Kharif (Monsoon)": "#27ae60", "Rabi (Winter)": "#f39c12"}
    
    # Panel 1: PCA Plotting
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=palette_colors, alpha=0.5, s=12, ax=axes[0])
    axes[0].set_title(f"Linear PCA Projections\n(PC1 Var: {var_exp[0]:.1f}% | PC2 Var: {var_exp[1]:.1f}%)")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].legend(title="Cropping Horizon")
    
    # Panel 2: t-SNE Plotting
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette=palette_colors, alpha=0.5, s=12, ax=axes[1])
    axes[1].set_title("Non-Linear t-SNE Topology\n(Local Neighborhood Distance)")
    axes[1].set_xlabel("t-SNE Axis 1")
    axes[1].set_ylabel("t-SNE Axis 2")
    axes[1].legend().remove()
    
    # Panel 3: UMAP Plotting
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, palette=palette_colors, alpha=0.5, s=12, ax=axes[2])
    axes[2].set_title("Global-Local UMAP Manifold\n(Continuity Preservation)")
    axes[2].set_xlabel("UMAP Axis 1")
    axes[2].set_ylabel("UMAP Axis 2")
    axes[2].legend().remove()
    
    plt.suptitle("Empirical Separability Analysis of Indian Crop Seasons inside Presto Feature Space", fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_png = "indian_season_separability_report.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Analysis completed. Clean chart saved to disk: '{output_png}'")

if __name__ == "__main__":
    # Test on a specific coordinate in Andhra Pradesh
    run_indian_season_experiments(lat=16.5062, lon=80.6480)
