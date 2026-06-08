import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from scipy.interpolate import make_interp_spline
from .crop_knowledge_db import CropKnowledgeDB

class GrowthStageEngine:
    """
    Growth Stage Engine.
    Analyzes temporal vegetation index patterns, aligns them with crop-specific calendars,
    and classifies satellite observation windows into precise biological growth stages.
    """
    
    def __init__(self):
        self.db = CropKnowledgeDB()

    def interpolate_to_daily(self, timestamps: List[str], values: np.ndarray, num_days: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates sparse observations (e.g. monthly GEE data) to smooth daily values.
        
        Args:
            timestamps (List[str]): List of date strings in YYYY-MM-DD format.
            values (np.ndarray): 1D array of observed values (e.g. NDVI).
            num_days (int): Target length of daily interpolation.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (days_array, daily_values_array)
        """
        # Parse dates and convert to relative days from start
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in timestamps]
        start_date = dates[0]
        days_from_start = np.array([(d - start_date).days for d in dates])
        
        # Ensure days are sorted
        sort_idx = np.argsort(days_from_start)
        days_from_start = days_from_start[sort_idx]
        values = values[sort_idx]
        
        # Target daily axis
        daily_days = np.arange(num_days)
        
        # Handle small array edge cases
        if len(days_from_start) < 3:
            # Linear interpolation for very few points
            daily_values = np.interp(daily_days, days_from_start, values)
        else:
            try:
                # Smooth cubic spline interpolation
                spline = make_interp_spline(days_from_start, values, k=2)
                daily_values = spline(daily_days)
            except Exception:
                daily_values = np.interp(daily_days, days_from_start, values)
                
        # Clip NDVI to realistic biological bounds
        daily_values = np.clip(daily_values, 0.05, 0.95)
        
        return daily_days, daily_values

    def detect_sowing_date(self, daily_days: np.ndarray, daily_ndvi: np.ndarray) -> int:
        """
        Heuristically estimates when the crop was actually sown in the observation window.
        Looks for the start of the sustained vegetative green-up phase (local minimum before a sharp rise).
        """
        # Calculate moving differences (NDVI velocity)
        window = 10
        velocity = np.zeros_like(daily_ndvi)
        for i in range(len(daily_ndvi) - window):
            velocity[i] = daily_ndvi[i + window] - daily_ndvi[i]
            
        # Sowing is typically when NDVI is low (near bare soil 0.15 - 0.25) and velocity becomes positive
        sowing_candidates = []
        for idx, val in enumerate(daily_ndvi):
            if val < 0.28 and velocity[idx] > 0.01:
                sowing_candidates.append(idx)
                
        if sowing_candidates:
            return sowing_candidates[0]
        
        # Fallback to day 0 if no clear green-up is found
        return 0

    def analyze_phenology(self, crop_name: str, timestamps: List[str], observed_ndvi: np.ndarray) -> Dict[str, Any]:
        """
        Maps a sequence of satellite observations to biological stages.
        
        Args:
            crop_name (str): Paddy, Wheat, Cotton, or Maize.
            timestamps (List[str]): List of GEE timestamps.
            observed_ndvi (np.ndarray): Spatially averaged NDVI for each timestamp.
            
        Returns:
            Dict[str, Any]: Analysis results including daily timeline, stage mapping, and alignment scores.
        """
        crop_info = self.db.get_crop_details(crop_name)
        total_duration_days = int(crop_info["duration_months"] * 30)

        
        # 1. Smooth the observed data to daily steps
        daily_days, daily_ndvi = self.interpolate_to_daily(timestamps, observed_ndvi, num_days=total_duration_days)
        
        # 2. Generate the expected normal profile for this crop
        expected_ndvi = self.db.generate_expected_ndvi_curve(crop_name, num_days=total_duration_days)
        
        # 3. Detect actual sowing day relative to the start of observation
        actual_sowing_day = self.detect_sowing_date(daily_days, daily_ndvi)
        
        # 4. Map daily states to biological stages
        stage_by_day = []
        stage_sequence = []
        
        for day in daily_days:
            # Shift day relative to actual sowing to align stages
            aligned_day = max(0, day - actual_sowing_day)
            stage = self.db.get_stage_for_day(crop_name, aligned_day)
            stage_by_day.append(stage)
            
        # 5. Extract stage index mappings for the original GEE timestamps
        # We find which day of the daily timeline corresponds to the GEE dates
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in timestamps]
        start_date = dates[0]
        
        observation_stages = []
        for d in dates:
            days_diff = (d - start_date).days
            # Clamp to timeline
            days_diff = min(max(0, days_diff), total_duration_days - 1)
            stage = stage_by_day[days_diff]
            observation_stages.append({
                "date": d.strftime("%Y-%m-%d"),
                "relative_day": days_diff,
                "stage": stage
            })

        # 6. Calculate fit alignment score (Correlation & RMSE)
        # Shift the observed curve to align sowing dates for matching with the template
        aligned_observed = np.zeros_like(expected_ndvi)
        aligned_len = len(daily_ndvi) - actual_sowing_day
        aligned_observed[:aligned_len] = daily_ndvi[actual_sowing_day:]
        # Fill remaining with bare soil/senescence values
        if actual_sowing_day > 0:
            aligned_observed[aligned_len:] = daily_ndvi[-1]
            
        rmse = np.sqrt(np.mean((aligned_observed - expected_ndvi) ** 2))
        
        # Pearson Correlation
        corr = np.corrcoef(aligned_observed, expected_ndvi)[0, 1]
        if np.isnan(corr):
            corr = 0.0
            
        # Profile match classification (helps detect crop misreporting fraud)
        is_profile_match = corr > 0.65 and rmse < 0.22
        
        return {
            "crop_name": crop_name,
            "actual_sowing_day": actual_sowing_day,
            "daily_timeline": {
                "days": daily_days.tolist(),
                "observed_ndvi": daily_ndvi.tolist(),
                "expected_ndvi": expected_ndvi.tolist(),
                "stages": stage_by_day
            },
            "observation_stages": observation_stages,
            "alignment_metrics": {
                "rmse": float(rmse),
                "correlation": float(corr),
                "is_profile_match": bool(is_profile_match)
            }
        }

if __name__ == "__main__":
    # Test run
    engine = GrowthStageEngine()
    test_dates = ["2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01", "2024-11-01"]
    test_ndvi = np.array([0.18, 0.22, 0.45, 0.78, 0.72, 0.35])
    
    result = engine.analyze_phenology("Paddy", test_dates, test_ndvi)
    print("Alignment Metrics:", result["alignment_metrics"])
    print("Observation Stages:")
    for obs in result["observation_stages"]:
        print(f"  {obs['date']}: {obs['stage']}")
