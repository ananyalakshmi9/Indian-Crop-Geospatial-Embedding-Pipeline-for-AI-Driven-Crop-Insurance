import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingSequencePreparer(torch.nn.Module):
    """
    Learned neural network block that projects raw 17-channel satellite and 
    weather vectors into a dense, high-dimensional Presto-like feature space (128-d).
    Allows end-to-end backprop from down-stream temporal models to the raw inputs.
    """
    def __init__(self, in_channels=17, embedding_dim=128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_channels, embedding_dim),
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        # Input shape: (Batch, Time, Channels)
        # Output shape: (Batch, Time, EmbeddingDim)
        return self.network(x)


class TemporalSequenceDataset(Dataset):
    """
    PyTorch Dataset mapping multi-temporal farm patches (Kharif & Rabi) 
    into pixel-level sequence classification tasks.
    
    Classes:
      0: Healthy Kharif (Monsoon Paddy)
      1: Stressed Kharif (Monsoon Paddy + Drought Stress at Flowering)
      2: Healthy Rabi (Winter Wheat)
      3: Stressed Rabi (Winter Wheat + Heat Stress at Maturity)
    """
    def __init__(
        self,
        data_dir="../DataEngineering",
        missing_rate=0.0,
        cloud_contamination=False,
        cloud_rate=0.2,
        seed=42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.missing_rate = missing_rate
        self.cloud_contamination = cloud_contamination
        self.cloud_rate = cloud_rate
        
        np.random.seed(seed)
        
        # Resolve paths
        kharif_file = os.path.join(data_dir, "farm_timeseries_kharif.npy")
        rabi_file = os.path.join(data_dir, "farm_timeseries_rabi.npy")
        
        # Fallback if paths are wrong (when executing from child folders)
        if not os.path.exists(kharif_file):
            data_dir = "./DataEngineering"
            kharif_file = os.path.join(data_dir, "farm_timeseries_kharif.npy")
            rabi_file = os.path.join(data_dir, "farm_timeseries_rabi.npy")
            
        if not os.path.exists(kharif_file):
            raise FileNotFoundError(f"Geospatial crop tensors not found. Searched {kharif_file}.")

        # Load tensors: shape (6, 64, 64, 17)
        self.kharif_raw = np.load(kharif_file)
        self.rabi_raw = np.load(rabi_file)
        
        # Clean up continuous no-data placeholder values (like -1.79e+308)
        self.kharif_raw[self.kharif_raw < -999.0] = 0.0
        self.rabi_raw[self.rabi_raw < -999.0] = 0.0
        self.kharif_raw = np.nan_to_num(self.kharif_raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.rabi_raw = np.nan_to_num(self.rabi_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        T, H, W, C = self.kharif_raw.shape
        num_pixels = H * W  # 4096
        
        # Reshape to pixel-level timeseries (Batch, Time, Channels)
        kharif_pixels = self.kharif_raw.reshape(T, num_pixels, C).transpose(1, 0, 2)
        rabi_pixels = self.rabi_raw.reshape(T, num_pixels, C).transpose(1, 0, 2)
        
        # Construct the dataset
        half_pixels = num_pixels // 2
        
        # 1. Healthy Kharif
        c0_data = kharif_pixels[:half_pixels].copy()
        c0_labels = np.zeros(half_pixels)
        
        # 2. Stressed Kharif (Monsoon Paddy with severe Drought at Flowering stage: months 3 & 4 - Sept/Oct)
        c1_data = kharif_pixels[half_pixels:].copy()
        # Reduce NDVI (ch 16) by 35% during Sept/Oct
        c1_data[:, 3:5, 16] *= 0.65
        # Increase temperature (ch 12) by 5K during drought
        c1_data[:, 3:5, 12] += 5.0
        # Drop precipitation (ch 13)
        c1_data[:, 3:5, 13] *= 0.1
        c1_labels = np.ones(half_pixels)
        
        # 3. Healthy Rabi
        c2_data = rabi_pixels[:half_pixels].copy()
        c2_labels = np.ones(half_pixels) * 2
        
        # 4. Stressed Rabi (Winter Wheat with Heat/Water Stress at Sowing/Maturity stage: months 4 & 5 - Mar/Apr)
        c3_data = rabi_pixels[half_pixels:].copy()
        # Reduce NDVI (ch 16) by 30% during Mar/Apr
        c3_data[:, 4:6, 16] *= 0.70
        # Increase temperature (ch 12) by 6K
        c3_data[:, 4:6, 12] += 6.0
        # Drop precipitation (ch 13)
        c3_data[:, 4:6, 13] *= 0.05
        c3_labels = np.ones(half_pixels) * 3
        
        # Combine
        self.data = np.vstack([c0_data, c1_data, c2_data, c3_data]).astype(np.float32)
        self.labels = np.concatenate([c0_labels, c1_labels, c2_labels, c3_labels]).astype(np.int64)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(self.data))
        self.data = self.data[shuffle_idx]
        self.labels = self.labels[shuffle_idx]
        
        # Apply simulations
        if self.missing_rate > 0:
            self._apply_missing_timestamps()
            
        if self.cloud_contamination:
            self._apply_cloud_contamination()

    def _apply_missing_timestamps(self):
        """
        Simulates missing timestamps due to sensor failure, scheduling gaps, or GEE server dropouts.
        Reconstructs the series via linear interpolation to keep structural shape.
        """
        B, T, C = self.data.shape
        for i in range(B):
            missing_mask = np.random.rand(T) < self.missing_rate
            # Ensure at least 2 timestamps remain to allow interpolation
            if missing_mask.sum() >= T - 1:
                missing_mask[np.random.randint(T)] = False
                missing_mask[np.random.randint(T)] = False
            
            for c in range(C):
                y = self.data[i, :, c]
                x = np.arange(T)
                
                # Known values
                known_idx = x[~missing_mask]
                known_vals = y[~missing_mask]
                
                # Interpolate missing values
                interpolated = np.interp(x, known_idx, known_vals)
                self.data[i, :, c] = interpolated

    def _apply_cloud_contamination(self):
        """
        Simulates monsoon cloud contamination in optical bands (June - August for Kharif, Dec - Feb for Rabi).
        Cloud contamination ruins Sentinel-2 optical bands (0-9) and NDVI (16), 
        but leaves Sentinel-1 radar bands (10, 11) and temperature/precip (12, 13) intact.
        
        We corrupt the optical channels with high values (reflectance of clouds) and drop NDVI.
        """
        B, T, C = self.data.shape
        
        # Channels to corrupt: B2-B12 (indices 0-9) and NDVI (index 16)
        optical_indices = list(range(10)) + [16]
        
        for i in range(B):
            # Target timestamps 0, 1, 2 (the peak monsoon months for Kharif, or winter mist/clouds for Rabi)
            cloudy_months = [0, 1, 2] if self.labels[i] in [0, 1] else [1, 2]
            
            for t in cloudy_months:
                if np.random.rand() < self.cloud_rate:
                    # Optical bands saturate with bright white cloud reflectance
                    self.data[i, t, :10] = self.data[i, t, :10] * 1.5 + np.random.normal(500, 100, 10)
                    # NDVI drops significantly due to cloud obstruction
                    self.data[i, t, 16] = 0.15 + np.random.normal(0, 0.05)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

if __name__ == "__main__":
    # Test the loader
    dataset = TemporalSequenceDataset(data_dir="./DataEngineering")
    print(f"Dataset Size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample X shape: {x.shape}, label: {y.item()}")
