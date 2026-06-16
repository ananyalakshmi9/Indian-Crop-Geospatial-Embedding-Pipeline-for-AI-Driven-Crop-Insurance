import numpy as np
import json
from typing import Dict, Any, List

class CropKnowledgeDB:
    """
    Crop Knowledge Layer & Growth Curve Database.
    Contains physiological details, calendar schedules, stress sensitivities, 
    and normal NDVI progression definitions for key Indian crops.
    """
    
    # ----------------------------------------------------
    # Database Definitions (Indian Crops Focus)
    # ----------------------------------------------------
    CROPS = {
        "Paddy": {
            "season": "Kharif",
            "sowing_month": 6,       # June (0-indexed month is 5, calendar month is 6)
            "duration_months": 6,    # June to November
            "stages": {
                "Sowing/Establishment": {
                    "start_rel_month": 0,
                    "end_rel_month": 1,
                    "expected_ndvi_range": (0.15, 0.25),
                    "water_sensitivity_ky": 0.2,   # Low: wet soils but flooding is mostly controlled
                    "temp_range_c": (20.0, 35.0),
                    "temp_sensitivity_kt": 0.1,
                    "description": "Seedbed preparation, transplanting, and early root establishment."
                },
                "Vegetative": {
                    "start_rel_month": 1,
                    "end_rel_month": 3,
                    "expected_ndvi_range": (0.25, 0.70),
                    "water_sensitivity_ky": 0.5,   # Moderate: tillering requires consistent water
                    "temp_range_c": (22.0, 32.0),
                    "temp_sensitivity_kt": 0.3,
                    "description": "Tillering, stem elongation, and rapid canopy expansion."
                },
                "Flowering/Reproductive": {
                    "start_rel_month": 3,
                    "end_rel_month": 4.5,
                    "expected_ndvi_range": (0.70, 0.85),
                    "water_sensitivity_ky": 1.2,   # Very High: drought during anthesis leads to high sterility
                    "temp_range_c": (25.0, 33.0),
                    "temp_sensitivity_kt": 0.8,    # Very High: heat waves during flowering cause spikelet sterility
                    "description": "Panicle initiation, heading, flowering, and pollination. Most critical stage."
                },
                "Maturity/Grain Filling": {
                    "start_rel_month": 4.5,
                    "end_rel_month": 6,
                    "expected_ndvi_range": (0.40, 0.75),
                    "water_sensitivity_ky": 0.6,   # Moderate: grain ripening and dry down
                    "temp_range_c": (18.0, 28.0),
                    "temp_sensitivity_kt": 0.4,
                    "description": "Milking, dough stage, ripening, and senescence."
                }
            },
            "yield_potential_kg_ha": 4500.0,
            "description": "Rice is the primary monsoon crop, requiring abundant water and warm temperatures."
        },
        "Wheat": {
            "season": "Rabi",
            "sowing_month": 11,      # November
            "duration_months": 5,    # November to March/April
            "stages": {
                "Sowing/Establishment": {
                    "start_rel_month": 0,
                    "end_rel_month": 1,
                    "expected_ndvi_range": (0.15, 0.20),
                    "water_sensitivity_ky": 0.1,
                    "temp_range_c": (10.0, 20.0),
                    "temp_sensitivity_kt": 0.2,
                    "description": "Germination and crown root initiation."
                },
                "Vegetative": {
                    "start_rel_month": 1,
                    "end_rel_month": 2.5,
                    "expected_ndvi_range": (0.20, 0.60),
                    "water_sensitivity_ky": 0.6,
                    "temp_range_c": (12.0, 22.0),
                    "temp_sensitivity_kt": 0.3,
                    "description": "Tillering and jointing stage."
                },
                "Flowering/Reproductive": {
                    "start_rel_month": 2.5,
                    "end_rel_month": 4,
                    "expected_ndvi_range": (0.60, 0.80),
                    "water_sensitivity_ky": 1.1,   # High: booting and flowering
                    "temp_range_c": (15.0, 25.0),
                    "temp_sensitivity_kt": 0.9,    # Extremely High: Terminal heat stress causes grain shriveling
                    "description": "Booting, heading, flowering, and pollination."
                },
                "Maturity/Grain Filling": {
                    "start_rel_month": 4,
                    "end_rel_month": 5,
                    "expected_ndvi_range": (0.25, 0.60),
                    "water_sensitivity_ky": 0.4,
                    "temp_range_c": (15.0, 30.0),
                    "temp_sensitivity_kt": 0.5,
                    "description": "Milk to soft dough, physiological maturity, and harvest dry down."
                }
            },
            "yield_potential_kg_ha": 3500.0,
            "description": "Wheat is the dominant winter crop in India, highly susceptible to late-season heat waves."
        },
        "Cotton": {
            "season": "Kharif",
            "sowing_month": 6,       # June
            "duration_months": 6.5,  # June to December
            "stages": {
                "Sowing/Establishment": {
                    "start_rel_month": 0,
                    "end_rel_month": 1,
                    "expected_ndvi_range": (0.15, 0.22),
                    "water_sensitivity_ky": 0.2,
                    "temp_range_c": (22.0, 38.0),
                    "temp_sensitivity_kt": 0.1,
                    "description": "Emergence and early leaf formation."
                },
                "Vegetative": {
                    "start_rel_month": 1,
                    "end_rel_month": 3,
                    "expected_ndvi_range": (0.22, 0.65),
                    "water_sensitivity_ky": 0.4,
                    "temp_range_c": (25.0, 35.0),
                    "temp_sensitivity_kt": 0.2,
                    "description": "Squaring (first flower buds) and branches extension."
                },
                "Flowering/Reproductive": {
                    "start_rel_month": 3,
                    "end_rel_month": 5,
                    "expected_ndvi_range": (0.65, 0.82),
                    "water_sensitivity_ky": 0.9,   # High: peak bloom and boll formation
                    "temp_range_c": (25.0, 32.0),
                    "temp_sensitivity_kt": 0.6,
                    "description": "Flowering, boll development, and fiber elongation."
                },
                "Maturity/Grain Filling": {
                    "start_rel_month": 5,
                    "end_rel_month": 6.5,
                    "expected_ndvi_range": (0.35, 0.65),
                    "water_sensitivity_ky": 0.3,   # Needs dry weather for bolls to open without rotting
                    "temp_range_c": (20.0, 30.0),
                    "temp_sensitivity_kt": 0.3,
                    "description": "Boll opening, leaf defoliation, and bursting."
                }
            },
            "yield_potential_kg_ha": 2000.0,
            "description": "Cotton is a cash crop preferring warm climates, needing moderate rainfall during bolls growth."
        },
        "Maize": {
            "season": "Kharif",
            "sowing_month": 6,       # June
            "duration_months": 4.5,  # June to October
            "stages": {
                "Sowing/Establishment": {
                    "start_rel_month": 0,
                    "end_rel_month": 1,
                    "expected_ndvi_range": (0.15, 0.25),
                    "water_sensitivity_ky": 0.2,
                    "temp_range_c": (18.0, 32.0),
                    "temp_sensitivity_kt": 0.1,
                    "description": "Germination and early leaf stage."
                },
                "Vegetative": {
                    "start_rel_month": 1,
                    "end_rel_month": 2.5,
                    "expected_ndvi_range": (0.25, 0.75),
                    "water_sensitivity_ky": 0.4,
                    "temp_range_c": (20.0, 32.0),
                    "temp_sensitivity_kt": 0.2,
                    "description": "Rapid stem elongation (knee-high to tasseling)."
                },
                "Flowering/Reproductive": {
                    "start_rel_month": 2.5,
                    "end_rel_month": 3.5,
                    "expected_ndvi_range": (0.75, 0.88),
                    "water_sensitivity_ky": 1.3,   # Extremely High: drought during silking causes poor kernel set
                    "temp_range_c": (22.0, 30.0),
                    "temp_sensitivity_kt": 0.7,
                    "description": "Tasseling, silking, and pollination."
                },
                "Maturity/Grain Filling": {
                    "start_rel_month": 3.5,
                    "end_rel_month": 4.5,
                    "expected_ndvi_range": (0.35, 0.75),
                    "water_sensitivity_ky": 0.5,
                    "temp_range_c": (16.0, 26.0),
                    "temp_sensitivity_kt": 0.3,
                    "description": "Milking, starch accumulation, black layer formation, and senescence."
                }
            },
            "yield_potential_kg_ha": 5000.0,
            "description": "Maize is highly sensitive to soil moisture deficits, especially during pollination."
        }
    }

    @classmethod
    def get_crop_details(cls, crop_name: str) -> Dict[str, Any]:
        """Fetch morphological configurations for the crop."""
        if crop_name not in cls.CROPS:
            raise KeyError(f"Crop '{crop_name}' is not registered in CropKnowledgeDB. Available: {list(cls.CROPS.keys())}")
        return cls.CROPS[crop_name]

    @classmethod
    def generate_expected_ndvi_curve(cls, crop_name: str, num_days: int = 180) -> np.ndarray:
        """
        Generates the mathematically expected NDVI growth curve (Double Logistic model)
        for a crop based on its registered phenology profiles.
        
        Equation: NDVI(t) = min + (max - min) * ( 1 / (1 + exp(-m1 * (t - t1))) - 1 / (1 + exp(-m2 * (t - t2))) )
        """
        crop = cls.get_crop_details(crop_name)
        stages = crop["stages"]
        
        # Extract relative stages to map parameters
        min_ndvi = 0.15
        max_ndvi = 0.85
        
        # Sowing stage NDVI
        if "Sowing/Establishment" in stages:
            min_ndvi = stages["Sowing/Establishment"]["expected_ndvi_range"][0]
        # Peak NDVI from Flowering stage
        if "Flowering/Reproductive" in stages:
            max_ndvi = stages["Flowering/Reproductive"]["expected_ndvi_range"][1]
            
        duration = crop["duration_months"] * 30  # duration in days
        
        # Calculate parameters based on crop growth calendar
        # t1: vegetative green-up midpoint (typically around 25% of cycle)
        # t2: maturity senescence midpoint (typically around 80% of cycle)
        t1 = duration * 0.28
        t2 = duration * 0.78
        
        # Slopes for growth and decay
        m1 = 0.10  # Green-up rate
        m2 = 0.08  # Senescence rate
        
        t = np.arange(num_days)
        
        # Scale curves to crop duration
        # If simulation is longer than duration, NDVI drops to bare soil baseline
        scale_factor = num_days / duration
        t_scaled = t / scale_factor
        
        # Double Logistic calculation
        term1 = 1.0 / (1.0 + np.exp(-m1 * (t_scaled - t1)))
        term2 = 1.0 / (1.0 + np.exp(-m2 * (t_scaled - t2)))
        
        expected_ndvi = min_ndvi + (max_ndvi - min_ndvi) * (term1 - term2)
        expected_ndvi = np.clip(expected_ndvi, min_ndvi, max_ndvi)
        
        return expected_ndvi

    @classmethod
    def get_stage_for_day(cls, crop_name: str, day: int) -> str:
        """Determines the biological growth stage based on day of crop season."""
        crop = cls.get_crop_details(crop_name)
        total_duration_days = crop["duration_months"] * 30
        rel_month = (day / total_duration_days) * crop["duration_months"]
        
        for stage_name, stage_info in crop["stages"].items():
            if stage_info["start_rel_month"] <= rel_month < stage_info["end_rel_month"]:
                return stage_name
                
        # Default back to maturity or sowing if out of bounds
        if rel_month < 0:
            return list(crop["stages"].keys())[0]
        return list(crop["stages"].keys())[-1]

    @classmethod
    def export_db_json(cls, filepath: str):
        """Saves crop metadata configurations to JSON for system integration."""
        with open(filepath, 'w') as f:
            json.dump(cls.CROPS, f, indent=4)
        print(f"Exported crop knowledge database to: {filepath}")

if __name__ == "__main__":
    # Test generation
    db = CropKnowledgeDB()
    paddy_curve = db.generate_expected_ndvi_curve("Paddy")
    print("Paddy expected NDVI curve shape:", paddy_curve.shape)
    print("Paddy stage at day 90:", db.get_stage_for_day("Paddy", 90))
