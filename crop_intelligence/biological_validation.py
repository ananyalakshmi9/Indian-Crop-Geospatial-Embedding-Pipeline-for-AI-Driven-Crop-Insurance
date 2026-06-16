import numpy as np
from typing import Dict, Any, List, Tuple
from .crop_knowledge_db import CropKnowledgeDB
from .growth_stage_engine import GrowthStageEngine


class BiologicalValidator:
    """
    Biological Validation Module.
    Combines satellite-derived vegetation indices, SAR backscatter features, 
    and meteorological inputs to perform stage-aware stress profiling 
    and yield loss risk analysis.
    """
    
    def __init__(self):
        self.db = CropKnowledgeDB()

    def calculate_vci(self, observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
        """
        Calculates a localized Vegetation Condition Index (VCI) over time.
        VCI = Observed / Expected. Values below 0.85 indicate stress.
        """
        # Prevent division by zero
        safe_expected = np.where(expected == 0.0, 0.001, expected)
        vci = observed / safe_expected
        return np.clip(vci, 0.0, 2.0)

    def assess_meteorological_stress(
        self,
        daily_stages: List[str],
        daily_temp: np.ndarray,
        daily_precip: np.ndarray,
        crop_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates thermal and moisture stress index curves on a daily basis.
        
        - Temperature Stress: Occurs if temperature is outside the optimal range.
        - Moisture Stress: Occurs if precipitation is below crop stage water demand.
        """
        crop_info = self.db.get_crop_details(crop_name)
        stages_info = crop_info["stages"]
        
        temp_stress = np.zeros_like(daily_temp)
        moisture_stress = np.zeros_like(daily_precip)
        
        # We need to map daily precipitation (which is typically given as monthly values or accumulated daily values)
        # ERA5 precipitation is usually in meters of water equivalent. 
        # Let's assume we converted it to mm (e.g. multiplied by 1000).
        # We will set a threshold for daily water requirement per stage.
        # e.g., Paddy Vegetative: 5mm/day, Flowering: 8mm/day, Maturity: 3mm/day.
        # Wheat Vegetative: 2mm/day, Flowering: 4mm/day, Maturity: 1.5mm/day.
        stage_water_demands = {
            "Sowing/Establishment": 3.0,
            "Vegetative": 5.0,
            "Flowering/Reproductive": 8.0,
            "Maturity/Grain Filling": 3.0
        }
        if crop_name == "Wheat":
            stage_water_demands = {
                "Sowing/Establishment": 1.5,
                "Vegetative": 3.0,
                "Flowering/Reproductive": 5.0,
                "Maturity/Grain Filling": 2.0
            }
            
        for i, stage in enumerate(daily_stages):
            if stage not in stages_info:
                continue
                
            info = stages_info[stage]
            min_opt, max_opt = info["temp_range_c"]
            
            # --- Temperature Stress Calculation ---
            t = daily_temp[i]
            if t < min_opt:
                # Cold stress
                temp_stress[i] = np.clip((min_opt - t) / 10.0, 0.0, 1.0)
            elif t > max_opt:
                # Heat stress (critical during flowering)
                temp_stress[i] = np.clip((t - max_opt) / 8.0, 0.0, 1.0)
            else:
                temp_stress[i] = 0.0
                
            # --- Moisture Stress Calculation ---
            # Compare daily rainfall against stage water demand
            p = daily_precip[i]
            demand = stage_water_demands.get(stage, 4.0)
            
            if p < demand:
                moisture_stress[i] = np.clip((demand - p) / demand, 0.0, 1.0)
            else:
                moisture_stress[i] = 0.0
                
        return temp_stress, moisture_stress

    def validate_biological_stress(
        self,
        phenology_results: Dict[str, Any],
        raw_temperatures: np.ndarray,
        raw_precipitations: np.ndarray,
        raw_sar_vh_vv: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Runs the full physiological stress assessment, weighting anomalies
        by stage-specific crop sensitivity factors.
        
        Args:
            phenology_results (Dict): Output from GrowthStageEngine.analyze_phenology.
            raw_temperatures (np.ndarray): 1D array of observed temperature (Kelvin or Celsius) at GEE timestamps.
            raw_precipitations (np.ndarray): 1D array of observed precipitation (m/day or mm/day) at GEE timestamps.
            raw_sar_vh_vv (np.ndarray, optional): 1D array of VH/VV ratio from Sentinel-1.
            
        Returns:
            Dict[str, Any]: Detailed biological stress verification report.
        """
        crop_name = phenology_results["crop_name"]
        crop_info = self.db.get_crop_details(crop_name)
        stages_info = crop_info["stages"]
        
        daily_timeline = phenology_results["daily_timeline"]
        days = np.array(daily_timeline["days"])
        observed_ndvi = np.array(daily_timeline["observed_ndvi"])
        expected_ndvi = np.array(daily_timeline["expected_ndvi"])
        daily_stages = daily_timeline["stages"]
        
        # 1. Convert temperature from Kelvin to Celsius if needed
        if raw_temperatures.mean() > 200:
            # ERA5 is in Kelvin, convert to Celsius
            temperatures_c = raw_temperatures - 273.15
        else:
            temperatures_c = raw_temperatures
            
        # 2. Convert precipitation from meters to mm if needed
        if raw_precipitations.mean() < 0.1:
            # ERA5 monthly precip is often in meters, convert to average daily mm
            # ERA5 total_precipitation_sum is monthly sum in meters. Let's convert: meters * 1000 / 30
            precip_mm_day = (raw_precipitations * 1000.0) / 30.0
        else:
            precip_mm_day = raw_precipitations

        # 3. Interpolate weather variables to daily to match NDVI timeline
        # Use growth stage engine's daily timeline to align weather
        dummy_dates = [f"2024-06-{i+1:02d}" for i in range(len(raw_temperatures))] # dummy dates for interpolation helper
        timestamps = [obs["date"] for obs in phenology_results["observation_stages"]]
        
        _, daily_temp = GrowthStageEngine().interpolate_to_daily(timestamps, temperatures_c, num_days=len(days))
        _, daily_precip = GrowthStageEngine().interpolate_to_daily(timestamps, precip_mm_day, num_days=len(days))
        
        # 4. Assessment of Vegetation Condition Index (VCI)
        daily_vci = self.calculate_vci(observed_ndvi, expected_ndvi)
        
        # 5. Assess temperature and moisture stress curves
        daily_temp_stress, daily_moisture_stress = self.assess_meteorological_stress(
            daily_stages, daily_temp, daily_precip, crop_name
        )
        
        # 6. Calculate Biologically Weighted Stage Stress
        # Combining VCI anomaly, thermal stress, and moisture stress, weighted by stage sensitivity
        daily_total_stress = np.zeros_like(days, dtype=float)
        
        stage_wise_stress = {stage: [] for stage in stages_info}
        
        for i, stage in enumerate(daily_stages):
            if stage not in stages_info:
                continue
                
            info = stages_info[stage]
            ky = info["water_sensitivity_ky"]
            kt = info["temp_sensitivity_kt"]
            
            # Stress component definitions
            ndvi_stress_val = max(0.0, 1.0 - daily_vci[i])
            w_stress_val = daily_moisture_stress[i] * ky
            t_stress_val = daily_temp_stress[i] * kt
            
            # Combine stress indicators (capped at 1.0)
            combined_stress = np.clip(0.4 * ndvi_stress_val + 0.3 * w_stress_val + 0.3 * t_stress_val, 0.0, 1.0)
            daily_total_stress[i] = combined_stress
            
            stage_wise_stress[stage].append(combined_stress)

        # 7. Compute Stage-averaged Stress Deficits
        avg_stage_stress = {}
        for stage, stresses in stage_wise_stress.items():
            avg_stage_stress[stage] = float(np.mean(stresses)) if stresses else 0.0
            
        # 8. Biological Yield Loss Prediction (Jensen Multiplicative Model)
        # Yield Reduction = 1 - Prod_over_stages ( 1 - Ky_stage * Avg_Stress_stage )
        yield_retention_ratio = 1.0
        
        for stage, info in stages_info.items():
            ky_stage = info["water_sensitivity_ky"]
            stage_stress = avg_stage_stress.get(stage, 0.0)
            
            # Reduce retention ratio by stage sensitivity
            stage_retention = 1.0 - (ky_stage * stage_stress * 0.5) # scaled factor for realistic loss
            stage_retention = np.clip(stage_retention, 0.1, 1.0)
            
            yield_retention_ratio *= stage_retention
            
        yield_loss_percentage = (1.0 - yield_retention_ratio) * 100.0
        
        # If SAR data is available, compute canopy water stress proxy (VH/VV ratio)
        sar_canopy_water_index = None
        if raw_sar_vh_vv is not None:
            # S1 VV and VH are backscatter coefficients in dB (usually -20 to -5)
            # In dB, VH - VV is equivalent to VH/VV ratio in linear scale.
            # For healthy dense crops, VH - VV increases (towards -5 to -10 dB difference).
            # If vegetation is dry or water-stressed, VH/VV ratio drops significantly.
            _, daily_sar = GrowthStageEngine().interpolate_to_daily(timestamps, raw_sar_vh_vv, num_days=len(days))
            sar_canopy_water_index = daily_sar.tolist()

        return {
            "crop_name": crop_name,
            "daily_indicators": {
                "vci": daily_vci.tolist(),
                "temperature_stress": daily_temp_stress.tolist(),
                "moisture_stress": daily_moisture_stress.tolist(),
                "total_stress": daily_total_stress.tolist(),
                "canopy_water_index": sar_canopy_water_index
            },
            "stage_stresses": avg_stage_stress,
            "estimated_yield_loss_pct": round(yield_loss_percentage, 2),
            "biophysically_valid": yield_loss_percentage > 10.0
        }

if __name__ == "__main__":
    # Test run
    from .growth_stage_engine import GrowthStageEngine
    
    engine = GrowthStageEngine()
    test_dates = ["2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01", "2024-11-01"]
    test_ndvi = np.array([0.18, 0.22, 0.45, 0.78, 0.72, 0.35])
    
    # 300K is ~26.8C, 303K is ~29.8C, etc.
    test_temps = np.array([301, 303, 305, 308, 304, 298]) # Kelvin
    test_precip = np.array([0.12, 0.15, 0.08, 0.02, 0.01, 0.05]) # meters
    
    phen_res = engine.analyze_phenology("Paddy", test_dates, test_ndvi)
    validator = BiologicalValidator()
    
    stress_res = validator.validate_biological_stress(
        phen_res, test_temps, test_precip
    )
    print("Stage Wise Stresses:", stress_res["stage_stresses"])
    print("Estimated Yield Loss:", stress_res["estimated_yield_loss_pct"], "%")
