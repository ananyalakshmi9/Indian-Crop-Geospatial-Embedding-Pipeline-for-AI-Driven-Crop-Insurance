import ee
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import os
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# CONFIG
# ==================================================

PROJECT_ID = os.getenv("EE_PROJECT_ID")

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)


# ==================================================
# DATE GENERATION
# ==================================================

def monthly_ranges(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    ranges = []
    current = start

    while current < end:
        next_month = current + relativedelta(months=1)

        ranges.append(
            (
                current.strftime("%Y-%m-%d"),
                next_month.strftime("%Y-%m-%d")
            )
        )

        current = next_month

    return ranges


# ==================================================
# SINGLE MONTH EXTRACTION
# ==================================================

def get_patch_for_period(
    lat,
    lon,
    start_date,
    end_date,
    patch_size=64
):

    point = ee.Geometry.Point([lon, lat])

    # 64 pixels × 10m resolution
    buffer_distance = (patch_size * 10) / 2

    roi = point.buffer(buffer_distance).bounds()

    # --------------------------------------
    # Sentinel-1
    # --------------------------------------

    s1_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH"
            )
        )
    )

    s1_count = s1_collection.size().getInfo()

    if s1_count == 0:
        print(f"No Sentinel-1 images found for {start_date}")
        return None

    s1 = (
        s1_collection
        .mean()
        .clip(roi)
    )

    # --------------------------------------
    # Sentinel-2
    # --------------------------------------

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                80
            )
        )
    )

    s2_count = s2_collection.size().getInfo()

    print(
        f"{start_date} -> {end_date} | "
        f"S1: {s1_count} images | "
        f"S2: {s2_count} images"
    )

    if s2_count == 0:
        print("Skipping month due to missing Sentinel-2 imagery.")
        return None

    s2 = (
        s2_collection
        .median()
        .clip(roi)
    )

    # --------------------------------------
    # DEM
    # --------------------------------------

    dem = (
        ee.Image("NASA/NASADEM_HGT/001")
        .select("elevation")
        .clip(roi)
    )
    terrain = ee.Terrain.products(dem)

    slope = terrain.select("slope")
    # --------------------------------------
    # NDVI
    # --------------------------------------

    ndvi = (
        s2.normalizedDifference(
            ["B8", "B4"]
        )
        .rename("NDVI")
    )

    # --------------------------------------
    # STACK CHANNELS
    # --------------------------------------
    era5 = (
        ee.ImageCollection(
            "ECMWF/ERA5_LAND/MONTHLY_AGGR"
        )
        .filterDate(
            start_date,
            end_date
        )
        .mean()
        .clip(roi)
    )
    temperature = era5.select(
        "temperature_2m"
    )
    precipitation = era5.select(
        "total_precipitation_sum"
    )
    stack = ee.Image.cat([

        s2.select("B2"),
        s2.select("B3"),
        s2.select("B4"),

        s2.select("B5"),
        s2.select("B6"),
        s2.select("B7"),

        s2.select("B8"),
        s2.select("B8A"),

        s2.select("B11"),
        s2.select("B12"),

        s1.select("VV"),
        s1.select("VH"),

        temperature,
        precipitation,

        dem,
        slope,
        ndvi

    ]).double()
    
    band_names = [

        "B2",
        "B3",
        "B4",

        "B5",
        "B6",
        "B7",

        "B8",
        "B8A",

        "B11",
        "B12",

        "VV",
        "VH",

        "temperature_2m",
        "total_precipitation_sum",

        "elevation",
        "slope",
        "NDVI"
    ]

    prepared = stack.clipToBoundsAndScale(
        geometry=roi,
        scale=10
    )

    pixels = ee.data.computePixels({
        "expression": prepared,
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": band_names
    })

    tensor = np.stack(
        [pixels[b] for b in band_names],
        axis=-1
    )

    tensor = np.nan_to_num(
        tensor,
        nan=0.0
    )

    # --------------------------------------
    # FORCE EXACT PATCH SIZE
    # --------------------------------------

    h, w, c = tensor.shape

    start_h = (h - patch_size) // 2
    start_w = (w - patch_size) // 2

    tensor = tensor[
        start_h:start_h + patch_size,
        start_w:start_w + patch_size,
        :
    ]

    return tensor


# ==================================================
# COMPLETE TIME SERIES
# ==================================================

def generate_timeseries_tensor(
    lat,
    lon,
    start_date,
    end_date,
    patch_size=64
):

    periods = monthly_ranges(
        start_date,
        end_date
    )

    tensors = []
    timestamps = []

    for start, end in periods:

        print(f"\nProcessing {start} -> {end}")

        patch = get_patch_for_period(
            lat=lat,
            lon=lon,
            start_date=start,
            end_date=end,
            patch_size=patch_size
        )

        if patch is None:
            continue

        tensors.append(patch)
        timestamps.append(start)

    if len(tensors) == 0:
        raise ValueError(
            "No valid satellite observations found."
        )

    tensor_sequence = np.stack(
        tensors,
        axis=0
    )

    return {
        "tensor": tensor_sequence,
        "timestamps": timestamps,
        "channels": [
        "B2",
        "B3",
        "B4",

        "B5",
        "B6",
        "B7",

        "B8",
        "B8A",

        "B11",
        "B12",

        "VV",
        "VH",

        "temperature_2m",
        "total_precipitation_sum",

        "elevation",
        "slope",
        "NDVI"
        ]
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    result = generate_timeseries_tensor(
        lat=16.5062,
        lon=80.6480,
        start_date="2024-06-01",
        end_date="2024-11-30"
    )

    tensor = result["tensor"]

    print("\n=================================")
    print("FINAL OUTPUT")
    print("=================================")

    print("Tensor Shape:", tensor.shape)

    print("\nTimestamps:")
    print(result["timestamps"])

    print("\nChannels:")
    print(result["channels"])

    np.save(
        "farm_timeseries.npy",
        tensor
    )
    

    print("\nSaved file:")
    print("farm_timeseries.npy")