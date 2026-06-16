import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. LOAD DATASET TENSOR
# ==============================================================================
file_path = "farm_timeseries.npy"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing data tracking asset: {file_path}. Please execute gee_timeseries_pipeline.py first.")

# Input layout expected: (Timestamps, Height, Width, Channels)
tensor = np.load(file_path)

CHANNELS = ["B4 (Red)", "B3 (Green)", "B2 (Blue)", "B8 (NIR)", "VV (Radar)", "VH (Radar)", "Elevation", "NDVI"]
num_timestamps, height, width, num_channels = tensor.shape

print("==============================================================================")
print("             GEOSPATIAL TENSOR DATA INTEGRITY VERIFICATION                    ")
print("==============================================================================")
print(f"Tensor Dimensional Matrix Structure: {tensor.shape}")
print(f"  • Timestamps Available: {num_timestamps}")
print(f"  • Patch Resolution:     {height} x {width} pixels")
print(f"  • Channels Ingested:    {num_channels}")
print("------------------------------------------------------------------------------")

# ==============================================================================
# 2. STATISTICAL VALIDATION (CHANNEL-BY-CHANNEL ANALYSIS)
# ==============================================================================
print(f"{'Channel Name':<15} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Status':<12}")
print("-" * 65)

all_zeros_detected = False

for idx, channel_name in enumerate(CHANNELS):
    channel_data = tensor[:, :, :, idx]
    c_min = channel_data.min()
    c_max = channel_data.max()
    c_mean = channel_data.mean()
    
    # Run sanity validation thresholds
    if c_min == 0.0 and c_max == 0.0:
        status = "❌ ALL ZEROS"
        all_zeros_detected = True
    elif np.isnan(c_mean) or np.isinf(c_mean):
        status = "❌ BAD VALUES"
    else:
        status = "✅ VALID"
        
    print(f"{channel_name:<15} | {c_min:<10.2f} | {c_max:<10.2f} | {c_mean:<10.2f} | {status}")

print("-" * 65)
if all_zeros_detected:
    print("⚠️ WARNING: Empty channel masks detected. Verify cloud-mask filters or server timeouts.")
else:
    print("🚀 SUCCESS: Matrix numbers are bounded and structurally valid for embedding learning.")
print("==============================================================================\n")

# ==============================================================================
# 3. SPATIAL VISUALIZATION (TIMESTAMP: 0)
# ==============================================================================
print("Generating analytical inline layout visualization for Timestamp Index 0...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Panel A: True Color Composite (RGB) ---
# Extract Red (B4), Green (B3), Blue (B2) channels
rgb_raw = tensor[0, :, :, 0:3]

# Normalize 12-bit Sentinel-2 surface reflectance (0-3000 mapping scale) to standard float range [0, 1]
rgb_normalized = np.clip(rgb_raw / 3000.0, 0.0, 1.0)

axes[0].imshow(rgb_normalized)
axes[0].set_title("True Color Composite (RGB)\n[Bands: B4, B3, B2]", fontsize=12, fontweight='bold')
axes[0].axis('off')

# --- Panel B: NDVI Matrix (Vegetation Index) ---
# Pull calculated NDVI sequence index from channel 7
ndvi_matrix = tensor[0, :, :, 7]

im_ndvi = axes[1].imshow(ndvi_matrix, cmap='YlGn', vmin=-0.1, vmax=0.9)
axes[1].set_title("Vegetation Index Matrix\n[Calculated NDVI]", fontsize=12, fontweight='bold')
axes[1].axis('off')
fig.colorbar(im_ndvi, ax=axes[1], fraction=0.046, pad=0.04, label="NDVI Scaling Vector")

# --- Panel C: Active Radar Backscatter Structure (VV Channel) ---
# Pull Sentinel-1 Ground Range Detected backscatter from channel 4
radar_matrix = tensor[0, :, :, 4]

im_radar = axes[2].imshow(radar_matrix, cmap='bone')
axes[2].set_title("Synthetic Aperture Radar Backscatter\n[Band: Sentinel-1 VV]", fontsize=12, fontweight='bold')
axes[2].axis('off')
fig.colorbar(im_radar, ax=axes[2], fraction=0.046, pad=0.04, label="Decibels (dB)")

plt.tight_layout()

# Save diagnostic visualization output file to project folder
output_image_name = "data_verification_report.png"
plt.savefig(output_image_name, dpi=150, bbox_inches='tight')
print(f"Pipeline verification file compiled and saved to disk: '{output_image_name}'")

# Render map layout window if running in desktop interactive console
plt.show()
