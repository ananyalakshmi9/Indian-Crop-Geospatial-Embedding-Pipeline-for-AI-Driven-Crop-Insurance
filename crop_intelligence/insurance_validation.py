import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from .growth_stage_engine import GrowthStageEngine
from .biological_validation import BiologicalValidator
from .crop_knowledge_db import CropKnowledgeDB

class InsuranceClaimValidator:
    """
    Stage-Aware Insurance Claim Validator (The Wow Factor).
    Combines phenology stage classification and biophysical stress assessments
    to dynamically approve, audit, or reject crop insurance claims.
    """
    
    def __init__(self):
        self.stage_engine = GrowthStageEngine()
        self.validator = BiologicalValidator()
        self.db = CropKnowledgeDB()

    def process_claim(
        self,
        claim_id: str,
        reported_crop: str,
        reported_incident_date: str,
        reported_cause: str,
        timestamps: List[str],
        tensor_filepath: str,
        spatial_buffer: int = 16
    ) -> Dict[str, Any]:
        """
        Validates an insurance claim using the actual timeseries satellite tensor.
        
        Args:
            claim_id (str): Unique identifier for the insurance claim.
            reported_crop (str): The crop name declared by the farmer.
            reported_incident_date (str): Date of damage (YYYY-MM-DD).
            reported_cause (str): Reported cause (e.g. Drought, Extreme Heat, Excess Rainfall).
            timestamps (List[str]): Timestamps corresponding to the tensor file.
            tensor_filepath (str): Absolute path to the .npy file from the GEE pipeline.
            spatial_buffer (int): Window size around the center of the patch to average.
            
        Returns:
            Dict[str, Any]: Detailed insurance validation report with decision and confidence scores.
        """
        print(f"\n[Insurance Validation] Processing Claim ID: {claim_id}")
        
        # 1. Load the GEE tensor
        try:
            tensor = np.load(tensor_filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Satellite tensor not found at {tensor_filepath}. Please run GEE pipeline first.")
            
        # Shape: (T, H, W, C)
        T, H, W, C = tensor.shape
        
        # 2. Focus on the central farm patch to avoid boundary mixing (roads, trees, neighboring farms)
        half_h, half_w = H // 2, W // 2
        sh, eh = half_h - spatial_buffer, half_h + spatial_buffer
        sw, ew = half_w - spatial_buffer, half_w + spatial_buffer
        
        farm_patch = tensor[:, sh:eh, sw:ew, :]
        
        # 3. Extract channels (spatial averages over the farm patch)
        # Based on gee_timeseries_pipeline.py:
        # Channel 16 is NDVI
        # Channel 12 is Temperature (Kelvin)
        # Channel 13 is Precipitation (meters)
        # Channel 10 is VV (radar)
        # Channel 11 is VH (radar)
        
        observed_ndvi = np.mean(farm_patch[:, :, :, 16], axis=(1, 2))
        observed_temp = np.mean(farm_patch[:, :, :, 12], axis=(1, 2))
        observed_precip = np.mean(farm_patch[:, :, :, 13], axis=(1, 2))
        
        # SAR VV/VH ratio (computed in dB as VH - VV)
        observed_vv = np.mean(farm_patch[:, :, :, 10], axis=(1, 2))
        observed_vh = np.mean(farm_patch[:, :, :, 11], axis=(1, 2))
        sar_ratio = observed_vh - observed_vv  # SAR canopy structure indicator
        
        # 4. Phase 1: Phenology Alignment & Crop Misreporting Audit
        # Check alignment against reported crop
        phen_res = self.stage_engine.analyze_phenology(reported_crop, timestamps, observed_ndvi)
        alignment = phen_res["alignment_metrics"]
        
        # Season check: does the start date of the observations match the crop season?
        crop_info = self.db.get_crop_details(reported_crop)
        start_date_obj = datetime.strptime(timestamps[0], "%Y-%m-%d")
        observed_start_month = start_date_obj.month
        expected_sowing_month = crop_info["sowing_month"]
        
        # Calculate month difference wrapping around 12
        month_diff = min((observed_start_month - expected_sowing_month) % 12, (expected_sowing_month - observed_start_month) % 12)
        
        # Fraud Check: Crop Misreporting
        # If the observed curve matches another crop template significantly better than the reported crop,
        # or if there is a clear season mismatch.
        fraud_alert = False
        suggested_actual_crop = reported_crop
        best_corr = alignment["correlation"]
        
        if month_diff > 2:
            fraud_alert = True
            # Find a crop candidate that matches the season better
            for crop_candidate, cand_info in self.db.CROPS.items():
                cand_sowing = cand_info["sowing_month"]
                cand_diff = min((observed_start_month - cand_sowing) % 12, (cand_sowing - observed_start_month) % 12)
                if cand_diff <= 2:
                    suggested_actual_crop = crop_candidate
                    break
        else:
            for crop_candidate in self.db.CROPS.keys():
                if crop_candidate == reported_crop:
                    continue
                cand_res = self.stage_engine.analyze_phenology(crop_candidate, timestamps, observed_ndvi)
                cand_corr = cand_res["alignment_metrics"]["correlation"]
                
                if cand_corr > best_corr + 0.25 and cand_corr > 0.70:
                    fraud_alert = True
                    suggested_actual_crop = crop_candidate
                    best_corr = cand_corr

                
        # 5. Phase 2: Biological Stress Profiling
        stress_res = self.validator.validate_biological_stress(
            phen_res, observed_temp, observed_precip, sar_ratio
        )
        
        # 6. Claim Validation Logic
        # Parse incident date and locate which stage it occurred in
        incident_dt = datetime.strptime(reported_incident_date, "%Y-%m-%d")
        start_dt = datetime.strptime(timestamps[0], "%Y-%m-%d")
        incident_day = (incident_dt - start_dt).days
        
        stages = phen_res["daily_timeline"]["stages"]
        total_days = len(stages)
        
        if 0 <= incident_day < total_days:
            incident_stage = stages[incident_day]
        else:
            incident_stage = "Unknown / Out-of-season"
            
        # Check stress levels during the reported incident stage
        incident_stage_stress = stress_res["stage_stresses"].get(incident_stage, 0.0)
        yield_loss_est = stress_res["estimated_yield_loss_pct"]
        
        # Decision matrix based on stage-aware risk and metrics
        decision = "UNDER_REVIEW"
        biological_narrative = ""
        confidence_score = 0.0
        
        if fraud_alert:
            decision = "REJECTED_FRAUD"
            biological_narrative = (
                f"Crop misreporting detected. The satellite vegetation profile does not align "
                f"with {reported_crop} (Correlation: {alignment['correlation']:.2f}). "
                f"Instead, it strongly aligns with {suggested_actual_crop} (Correlation: {best_corr:.2f})."
            )
            confidence_score = 0.95
        elif incident_stage == "Unknown / Out-of-season":
            decision = "REJECTED"
            biological_narrative = f"Reported incident date ({reported_incident_date}) falls outside the active cropping season."
            confidence_score = 0.90
        else:
            # Check reported cause matches satellite indicators
            cause_validated = False
            
            if reported_cause.lower() in ["drought", "moisture stress", "water deficit"]:
                # Check moisture stress in that stage
                stage_moisture_stress = np.mean([
                    stress_res["daily_indicators"]["moisture_stress"][i]
                    for i, s in enumerate(stages) if s == incident_stage
                ])
                if stage_moisture_stress > 0.4:
                    cause_validated = True
                    biological_narrative = f"Satellite-retrieved soil moisture and rainfall indices confirm severe deficit during the {incident_stage} stage."
                else:
                    biological_narrative = f"No significant soil moisture or rainfall deficit detected during the {incident_stage} stage, contrary to the reported cause '{reported_cause}'."
                    
            elif reported_cause.lower() in ["extreme heat", "heat wave", "thermal stress"]:
                stage_temp_stress = np.mean([
                    stress_res["daily_indicators"]["temperature_stress"][i]
                    for i, s in enumerate(stages) if s == incident_stage
                ])
                if stage_temp_stress > 0.4:
                    cause_validated = True
                    biological_narrative = f"Met-reanalysis confirms temperature exceedance above crop-tolerance thresholds during the sensitive {incident_stage} stage."
                else:
                    biological_narrative = f"Thermal signatures were within optimal physiological ranges for {reported_crop} during the {incident_stage} stage."
            else:
                # Default validation on overall vegetation index drop (VCI)
                stage_vci = np.mean([
                    stress_res["daily_indicators"]["vci"][i]
                    for i, s in enumerate(stages) if s == incident_stage
                ])
                if stage_vci < 0.82:
                    cause_validated = True
                    biological_narrative = f"Vegetation Condition Index (VCI) shows significant anomalous biomass drop during the {incident_stage} stage."
                else:
                    biological_narrative = f"Vegetation development was normal or healthy (VCI: {stage_vci:.2f}) during the {incident_stage} stage."

            # Calculate decision based on biological yield loss estimation and stage-sensitivity
            stage_details = self.db.CROPS[reported_crop]["stages"].get(incident_stage, {})
            stage_sensitivity = stage_details.get("water_sensitivity_ky", 0.5)
            
            if cause_validated and yield_loss_est > 25.0:
                decision = "APPROVED"
                biological_narrative += (
                    f" The damage occurred during the critical '{incident_stage}' stage (Sensitivity Coefficient Ky: {stage_sensitivity}), "
                    f"which has an irreversible impact on yield. Estimated crop yield reduction is {yield_loss_est}%."
                )
                confidence_score = 0.88
            elif cause_validated and yield_loss_est > 10.0:
                decision = "PARTIALLY_APPROVED"
                biological_narrative += (
                    f" Moderate stress validated during the '{incident_stage}' stage (Sensitivity: {stage_sensitivity}). "
                    f"Estimated crop yield reduction is {yield_loss_est}%. Yield impact is mild as crop retains recovery potential."
                )
                confidence_score = 0.82
            else:
                decision = "REJECTED"
                biological_narrative += f" No significant yield-impacting biophysical anomalies validated. Overall yield loss risk is negligible ({yield_loss_est}%)."
                confidence_score = 0.85

        return {
            "claim_id": claim_id,
            "validation_decision": decision,
            "confidence_score": round(confidence_score, 2),
            "reported_crop": reported_crop,
            "detected_crop_match": suggested_actual_crop if fraud_alert else reported_crop,
            "crop_misreporting_detected": fraud_alert,
            "reported_cause": reported_cause,
            "incident_stage": incident_stage,
            "stage_sensitivity": stage_details.get("water_sensitivity_ky", 0.0) if 'stage_details' in locals() else 0.0,
            "estimated_yield_loss_pct": yield_loss_est,
            "biological_evidence": biological_narrative,
            "observed_ndvi_profile": observed_ndvi.tolist()
        }

if __name__ == "__main__":
    # Test validation
    validator = InsuranceClaimValidator()
    # Mock data to test structure
    print("Ready to process crop insurance claims.")
