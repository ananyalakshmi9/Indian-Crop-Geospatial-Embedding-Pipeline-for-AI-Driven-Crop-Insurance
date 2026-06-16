import numpy as np
import gee_timeseries_pipeline
import presto_encoder

def get_farm_embeddings(lat, lon, start_date, end_date, start_month):
    """
    Complete pipeline API to generate Presto embeddings for a given farm coordinate and time period.
    
    Args:
        lat (float): Latitude of the farm.
        lon (float): Longitude of the farm.
        start_date (str): The start date for the GEE extraction in YYYY-MM-DD.
        end_date (str): The end date for the GEE extraction in YYYY-MM-DD.
        start_month (int): The 0-indexed starting month value expected by Presto 
                           (e.g., June = 5, December = 11).
                           
    Returns:
        np.ndarray: The generated Presto embeddings of shape (batch_size, embedding_dim).
    """
    print(f"\n--- API Request: Fetching Embeddings for Farm ({lat}, {lon}) ---")
    print(f"Period: {start_date} to {end_date}")
    
    # 1. Extract Time-Series Tensor from GEE
    print("\n[API] Step 1: Extracting GEE Data...")
    gee_result = gee_timeseries_pipeline.generate_timeseries_tensor(
        lat=lat, 
        lon=lon, 
        start_date=start_date, 
        end_date=end_date
    )
    tensor = gee_result["tensor"]
    
    # 2. Encode with Presto
    print("\n[API] Step 2: Generating Presto Embeddings...")
    embeddings = presto_encoder.generate_presto_embeddings(
        tensor=tensor,
        latlons=[lat, lon],
        month=start_month
    )
    
    print("\n[API] Processing Complete!")
    return embeddings

if __name__ == "__main__":
    # Quick test of the API
    embeddings = get_farm_embeddings(
        lat=16.5062, 
        lon=80.6480, 
        start_date="2024-06-01", 
        end_date="2024-11-30", 
        start_month=5
    )
    print("API returned embeddings shape:", embeddings.shape)