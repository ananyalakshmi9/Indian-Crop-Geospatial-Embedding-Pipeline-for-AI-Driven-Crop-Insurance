import numpy as np
import torch
from presto import Presto

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_presto_embeddings(tensor, latlons, month):
    """
    Generates Presto embeddings from a given Sentinel/ERA5 time-series tensor.
    
    Args:
        tensor (np.ndarray): The 4D tensor (T, H, W, C) from GEE pipeline.
        latlons (np.ndarray or list): The [[lat, lon], ...] coordinates for each batch.
        month (int or np.ndarray): The starting month(s) (1-12) or (0-11) based on what Presto expects.
                                   We assume 0-indexed month is passed if using `eval_task=True` with internal presto, 
                                   but typical Presto takes 1-12.
    
    Returns:
        np.ndarray: The resulting Presto embeddings.
    """
    print("=" * 70)
    print("GENERATING PRESTO EMBEDDINGS")
    print("=" * 70)

    print("Input Shape:", tensor.shape)
    T, H, W, C = tensor.shape

    # --------------------------------------------------
    # Convert spatial cube -> pixel timeseries
    # --------------------------------------------------
    pixel_series = tensor.reshape(T, H * W, C)
    pixel_series = np.transpose(pixel_series, (1, 0, 2))
    batch_size = pixel_series.shape[0]

    print("Pixel Batch:", pixel_series.shape)

    # --------------------------------------------------
    # PRESTO INPUTS
    # --------------------------------------------------
    x = torch.tensor(pixel_series, dtype=torch.float32)
    mask = torch.zeros_like(x)
    dynamic_world = torch.zeros((batch_size, T), dtype=torch.long)
    
    # Expand latlons to match batch_size (number of pixels)
    if isinstance(latlons, list):
        latlons = np.array(latlons)
        
    if latlons.ndim == 1:
        latlons = np.array([latlons])
        
    if latlons.shape[0] == 1:
        # Repeat the single lat/lon for all pixels
        latlons = np.repeat(latlons, batch_size, axis=0)
        
    latlons_tensor = torch.tensor(latlons, dtype=torch.float32)

    # Handle month scalar vs array
    if isinstance(month, int):
        month_tensor = torch.full((batch_size,), month, dtype=torch.long)
    else:
        # Assuming month is an array-like matching batch size
        month_tensor = torch.tensor(month, dtype=torch.long)

    print("\nLoading NASA Harvest Presto")
    model = Presto.load_pretrained()
    model.eval()
    model.to(DEVICE)

    with torch.no_grad():
        embeddings = model.encoder(
            x=x.to(DEVICE),
            dynamic_world=dynamic_world.to(DEVICE),
            latlons=latlons_tensor.to(DEVICE),
            mask=mask.to(DEVICE),
            month=month_tensor.to(DEVICE),
            eval_task=True
        )

    embeddings_np = embeddings.cpu().numpy()

    print("\nEmbedding Shape:", embeddings_np.shape)

    return embeddings_np

if __name__ == "__main__":
    # Example usage for testing
    print("Please import and use generate_presto_embeddings(tensor, latlons, month)")