import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import torch
from .growth_stage_engine import GrowthStageEngine
from .biological_validation import BiologicalValidator
from .crop_knowledge_db import CropKnowledgeDB
from temporal_intelligence.mamba_model import MambaClassifier
from torch.utils.data import DataLoader

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
        
        # Initialize Mamba Classifier
        self.mamba_model = MambaClassifier(in_channels=17, embedding_dim=128, num_layers=2, d_state=16, num_classes=4)
        self.mamba_model.eval()
        
        # Load weights
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(dir_path, "temporal_intelligence", "mamba_paddy_pilot.pt")
        if os.path.exists(weights_path):
            try:
                self.mamba_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
                print(f"[Mamba Integration] Loaded pre-trained model weights from: {weights_path}")
            except Exception as e:
                print(f"[Mamba Integration] Failed to load model weights: {e}. Running fallback training.")
                self._fallback_train(weights_path)
        else:
            print(f"[Mamba Integration] Checkpoint not found at {weights_path}. Running fallback training.")
            self._fallback_train(weights_path)

    def _fallback_train(self, weights_path: str):
        print("[Mamba Integration] Running fallback training on dataset...")
        try:
            from temporal_intelligence.sequence_loader import TemporalSequenceDataset
            import torch.optim as optim
            
            dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(dir_path, "DataEngineering")
            dataset = TemporalSequenceDataset(data_dir=data_dir)
            train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
            
            self.mamba_model.train()
            optimizer = optim.AdamW(self.mamba_model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            for epoch in range(3):
                for x, y in train_loader:
                    optimizer.zero_grad()
                    logits = self.mamba_model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    
            self.mamba_model.eval()
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(self.mamba_model.state_dict(), weights_path)
            print(f"[Mamba Integration] Fallback training complete. Saved weights to: {weights_path}")
        except Exception as e:
            print(f"[Mamba Integration] Fallback training failed: {e}. Model will use random initialization.")

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
        
        # Run Mamba Pixel Inference
        H_patch, W_patch = farm_patch.shape[1], farm_patch.shape[2]
        flat_patch = farm_patch.transpose(1, 2, 0, 3).reshape(H_patch * W_patch, T, C)
        
        # Clean up placeholders
        flat_patch[flat_patch < -999.0] = 0.0
        flat_patch = np.nan_to_num(flat_patch, nan=0.0, posinf=0.0, neginf=0.0)
        
        x_tensor = torch.tensor(flat_patch, dtype=torch.float32)
        with torch.no_grad():
            mamba_logits = self.mamba_model(x_tensor)
            mamba_probs = torch.softmax(mamba_logits, dim=-1)
            mean_probs = torch.mean(mamba_probs, dim=0).cpu().numpy()
            predicted_class = int(np.argmax(mean_probs))
            mamba_confidence = float(mean_probs[predicted_class])
            
        class_names = {
            0: "Healthy Kharif (Paddy)",
            1: "Stressed Kharif (Paddy)",
            2: "Healthy Rabi (Wheat)",
            3: "Stressed Rabi (Wheat)"
        }
        
        mamba_class_name = class_names.get(predicted_class, "Unknown")
        print(f"[Mamba Inference] Predicted Class: {mamba_class_name} (Conf: {mamba_confidence:.2f})")
        
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
        ai_recommendation = "RECOMMEND_AUDIT_LOW_PRIORITY"
        biological_narrative = ""
        confidence_score = 0.0
        human_auditor_action_items = []
        
        # Check Mamba crop classification match
        mamba_reported_crop_match = True
        suggested_mamba_crop = reported_crop
        if reported_crop.lower() == "paddy" and predicted_class not in [0, 1]:
            mamba_reported_crop_match = False
            suggested_mamba_crop = "Wheat"
        elif reported_crop.lower() == "wheat" and predicted_class not in [2, 3]:
            mamba_reported_crop_match = False
            suggested_mamba_crop = "Paddy"
            
        mamba_fraud_alert = not mamba_reported_crop_match
        crop_misreport = fraud_alert or mamba_fraud_alert
        detected_crop = suggested_actual_crop if fraud_alert else (suggested_mamba_crop if mamba_fraud_alert else reported_crop)
        
        if crop_misreport:
            ai_recommendation = "RECOMMEND_AUDIT_HIGH_PRIORITY"
            biological_narrative = (
                f"Crop misreporting suspected. The rule-based phenology check (correlation: {alignment['correlation']:.2f}) "
                f"and Mamba temporal classifier (prediction: {mamba_class_name}, confidence: {mamba_confidence:.2f}) "
                f"suggest a crop or season mismatch. Reported: {reported_crop}, Detected Match: {detected_crop}."
            )
            confidence_score = max(0.95, mamba_confidence)
            human_auditor_action_items.extend([
                "Verify sowing records and crop variety registry with local cooperative logs.",
                "Dispatch field adjuster for rapid crop type audit and photo-documentation.",
                "Cross-examine spatial boundary coordinates to verify crop classification history from previous seasons."
            ])
            
        elif incident_stage == "Unknown / Out-of-season":
            ai_recommendation = "RECOMMEND_AUDIT_HIGH_PRIORITY"
            biological_narrative = (
                f"Reported incident date ({reported_incident_date}) falls outside the active cropping season "
                f"for {reported_crop}."
            )
            confidence_score = 0.90
            human_auditor_action_items.extend([
                "Verify sowing date and actual crop establishment records.",
                "Confirm reported incident date with local meteorological anomaly events.",
                "Validate claim date against region-specific harvesting schedules."
            ])
            
        else:
            # Check reported cause matches satellite indicators
            cause_validated = False
            
            if reported_cause.lower() in ["drought", "moisture stress", "water deficit"]:
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
            
            # Mamba stress flags
            mamba_predicts_stressed = (reported_crop.lower() == "paddy" and predicted_class == 1) or \
                                      (reported_crop.lower() == "wheat" and predicted_class == 3)
            
            # Hybrid validation logic
            if cause_validated and yield_loss_est > 25.0 and mamba_predicts_stressed:
                ai_recommendation = "RECOMMEND_APPROVE"
                biological_narrative += (
                    f" The damage occurred during the critical '{incident_stage}' stage (Sensitivity Coefficient Ky: {stage_sensitivity}), "
                    f"which has an irreversible impact on yield. Estimated crop yield reduction is {yield_loss_est}% (biophysically validated by weather and Mamba models)."
                )
                confidence_score = 0.8 * mamba_confidence + 0.2 * alignment["correlation"]
                human_auditor_action_items.extend([
                    f"Generate payout estimate based on the validated yield loss of {yield_loss_est}%.",
                    "Verify bank account details and land ownership records.",
                    "Cross-check yield loss estimation with regional agricultural officer reports."
                ])
            elif cause_validated and yield_loss_est > 10.0 and mamba_predicts_stressed:
                ai_recommendation = "RECOMMEND_APPROVE_PARTIAL"
                biological_narrative += (
                    f" Moderate stress validated during the '{incident_stage}' stage (Sensitivity: {stage_sensitivity}). "
                    f"Estimated crop yield reduction is {yield_loss_est}%. Yield impact is mild as crop retains recovery potential (validated by weather and Mamba models)."
                )
                confidence_score = 0.8 * mamba_confidence + 0.2 * alignment["correlation"]
                human_auditor_action_items.extend([
                    f"Prepare partial payout schedule based on moderate yield loss of {yield_loss_est}%.",
                    "Review regional reports for localized recovery indicators."
                ])
            else:
                # Decide if it's high audit or low audit priority based on conflicts
                if not cause_validated and (yield_loss_est > 25.0 or mamba_predicts_stressed):
                    # Conflict: yield loss or Mamba says stressed, but weather cause is not validated
                    ai_recommendation = "RECOMMEND_AUDIT_HIGH_PRIORITY"
                    biological_narrative += (
                        f" Stress detected (Yield loss est: {yield_loss_est}%, Mamba stressed: {mamba_predicts_stressed}), "
                        f"but reported cause '{reported_cause}' is not validated by meteorological data. Potential incorrect cause declaration."
                    )
                    confidence_score = 0.85
                    human_auditor_action_items.extend([
                        "Check for alternative causes of crop damage (e.g. localized flooding, pest/disease, ground frost).",
                        "Verify localized weather anomalies with village-level cooperative weather station data."
                    ])
                elif cause_validated and not mamba_predicts_stressed:
                    # Conflict: weather validation passed, but Mamba shows healthy crop (no stress)
                    ai_recommendation = "RECOMMEND_AUDIT_HIGH_PRIORITY"
                    biological_narrative += (
                        f" Discrepancy detected: Weather model shows soil moisture/thermal deficit matching '{reported_cause}' "
                        f"(Jensen model yield loss estimation: {yield_loss_est}%), but Mamba temporal classifier "
                        f"predicts the crop is healthy (prediction: {mamba_class_name}, confidence: {mamba_confidence:.2f})."
                    )
                    confidence_score = 0.88
                    human_auditor_action_items.extend([
                        "Verify ground-truth crop condition with field photos and local cooperative reports.",
                        "Inspect for micro-irrigation or groundwater buffer capacity that might mitigate meteorological drought.",
                        "Verify if the crop recovered in subsequent months post-incident date."
                    ])
                else:
                    # Default: No significant yield-impacting biophysical anomalies validated
                    ai_recommendation = "RECOMMEND_AUDIT_LOW_PRIORITY"
                    biological_narrative += f" No significant yield-impacting biophysical anomalies validated. Overall yield loss risk is negligible ({yield_loss_est}%)."
                    confidence_score = 0.85
                    human_auditor_action_items.extend([
                        "Conduct routine audit checklist verification.",
                        "Inspect field photos for localized disease/pests which are not captured by weather models."
                    ])

        return {
            "claim_id": claim_id,
            "validation_decision": ai_recommendation,
            "confidence_score": round(confidence_score, 2),
            "reported_crop": reported_crop,
            "detected_crop_match": detected_crop,
            "crop_misreporting_detected": crop_misreport,
            "reported_cause": reported_cause,
            "incident_stage": incident_stage,
            "stage_sensitivity": stage_details.get("water_sensitivity_ky", 0.0) if 'stage_details' in locals() and isinstance(stage_details, dict) else 0.0,
            "estimated_yield_loss_pct": yield_loss_est,
            "biological_evidence": biological_narrative,
            "human_auditor_action_items": human_auditor_action_items,
            "observed_ndvi_profile": observed_ndvi.tolist()
        }

if __name__ == "__main__":
    # Test validation
    validator = InsuranceClaimValidator()
    # Mock data to test structure
    print("Ready to process crop insurance claims.")
