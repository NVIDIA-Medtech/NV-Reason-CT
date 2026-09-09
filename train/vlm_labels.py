# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared NV-Reason-CT chest and abdomen label definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CTLabels:
    """Region-specific labels used by NV-Reason-CT."""

    # CT-RATE original Chest 18 labels
    Medical_material: str = "Medical material"
    Arterial_wall_calcification: str = "Arterial wall calcification"
    Cardiomegaly: str = "Cardiomegaly"
    Pericardial_effusion: str = "Pericardial effusion"
    Coronary_artery_wall_calcification: str = "Coronary artery wall calcification"
    Hiatal_hernia: str = "Hiatal hernia"
    Lymphadenopathy: str = "Lymphadenopathy"
    Emphysema: str = "Emphysema"
    Atelectasis: str = "Atelectasis"
    Lung_nodule: str = "Lung nodule"
    Lung_opacity: str = "Lung opacity"
    Pulmonary_fibrotic_sequela: str = "Pulmonary fibrotic sequela"
    Pleural_effusion: str = "Pleural effusion"
    Mosaic_attenuation_pattern: str = "Mosaic attenuation pattern"
    Peribronchial_thickening: str = "Peribronchial thickening"
    Consolidation: str = "Consolidation"
    Bronchiectasis: str = "Bronchiectasis"
    Interlobular_septal_thickening: str = "Interlobular septal thickening"

    # Chest extended
    Lung_mass: str = "Lung mass"
    Pleural_thickening_or_mass: str = "Pleural thickening or mass"
    Mediastinal_mass: str = "Mediastinal mass"
    Lymphoma: str = "Lymphoma"
    Pneumothorax: str = "Pneumothorax"
    Thoracic_bone_lesion: str = "Thoracic bone lesion"
    Thoracic_fracture: str = "Thoracic fracture"
    Chest_wall_or_breast_mass: str = "Chest wall or breast mass"
    Pulmonary_embolism: str = "Pulmonary embolism"
    Thoracic_vascular_abnormality: str = "Thoracic vascular abnormality"
    Esophageal_abnormality: str = "Esophageal abnormality"
    Postoperative_or_treatment_related_change: str = "Postoperative or treatment-related change"
    No_Chest_Finding: str = "No Chest Finding"

    # Abdominal labels
    Liver_cyst: str = "Liver cyst"
    Liver_mass_or_non_cystic_lesion: str = "Liver mass or non-cystic lesion"
    Hepatic_steatosis: str = "Hepatic steatosis"
    Biliary_dilation: str = "Biliary dilation"
    Gallstone_or_gallbladder_abnormality: str = "Gallstone or gallbladder abnormality"
    Splenomegaly_or_splenic_lesion: str = "Splenomegaly or splenic lesion"
    Pancreatic_lesion_or_duct_abnormality: str = "Pancreatic lesion or duct abnormality"
    Pancreatitis: str = "Pancreatitis"
    Renal_cyst: str = "Renal cyst"
    Renal_mass: str = "Renal mass"
    Urinary_calculus: str = "Urinary calculus"
    Urinary_obstruction: str = "Urinary obstruction"
    Adrenal_nodule_or_mass: str = "Adrenal nodule or mass"
    Bowel_wall_thickening_or_mass: str = "Bowel wall thickening or mass"
    Bowel_obstruction_or_dilation: str = "Bowel obstruction or dilation"
    Diverticular_disease: str = "Diverticular disease"
    Appendiceal_abnormality: str = "Appendiceal abnormality"
    Ascites: str = "Ascites"
    Peritoneal_omental_disease: str = "Peritoneal/omental disease"
    Pneumoperitoneum: str = "Pneumoperitoneum"
    Abdominal_vascular_abnormality: str = "Abdominal vascular abnormality"
    Abdominal_vascular_calcification: str = "Abdominal vascular calcification"
    Abdominal_lymphadenopathy: str = "Abdominal lymphadenopathy"
    Abdominal_wall_hernia: str = "Abdominal wall hernia"
    Abdominal_soft_tissue_mass: str = "Abdominal soft tissue mass"
    Abdominopelvic_bone_lesion: str = "Abdominopelvic bone lesion"
    Abdominopelvic_fracture: str = "Abdominopelvic fracture"
    Abdominal_postoperative_change: str = "Abdominal postoperative change"
    Abdominal_medical_material: str = "Abdominal medical material"
    No_Abdominal_Finding: str = "No Abdominal Finding"

    @staticmethod
    def get_chest_list():
        return [
            CTLabels.Medical_material,
            CTLabels.Arterial_wall_calcification,
            CTLabels.Cardiomegaly,
            CTLabels.Pericardial_effusion,
            CTLabels.Coronary_artery_wall_calcification,
            CTLabels.Hiatal_hernia,
            CTLabels.Lymphadenopathy,
            CTLabels.Emphysema,
            CTLabels.Atelectasis,
            CTLabels.Lung_nodule,
            CTLabels.Lung_opacity,
            CTLabels.Pulmonary_fibrotic_sequela,
            CTLabels.Pleural_effusion,
            CTLabels.Mosaic_attenuation_pattern,
            CTLabels.Peribronchial_thickening,
            CTLabels.Consolidation,
            CTLabels.Bronchiectasis,
            CTLabels.Interlobular_septal_thickening,
            CTLabels.Lung_mass,
            CTLabels.Pleural_thickening_or_mass,
            CTLabels.Mediastinal_mass,
            CTLabels.Lymphoma,
            CTLabels.Pneumothorax,
            CTLabels.Thoracic_bone_lesion,
            CTLabels.Thoracic_fracture,
            CTLabels.Chest_wall_or_breast_mass,
            CTLabels.Pulmonary_embolism,
            CTLabels.Thoracic_vascular_abnormality,
            CTLabels.Esophageal_abnormality,
            CTLabels.Postoperative_or_treatment_related_change,
            CTLabels.No_Chest_Finding,
        ]

    @staticmethod
    def get_abdomen_list():
        return [
            CTLabels.Liver_cyst,
            CTLabels.Liver_mass_or_non_cystic_lesion,
            CTLabels.Hepatic_steatosis,
            CTLabels.Biliary_dilation,
            CTLabels.Gallstone_or_gallbladder_abnormality,
            CTLabels.Splenomegaly_or_splenic_lesion,
            CTLabels.Pancreatic_lesion_or_duct_abnormality,
            CTLabels.Pancreatitis,
            CTLabels.Renal_cyst,
            CTLabels.Renal_mass,
            CTLabels.Urinary_calculus,
            CTLabels.Urinary_obstruction,
            CTLabels.Adrenal_nodule_or_mass,
            CTLabels.Bowel_wall_thickening_or_mass,
            CTLabels.Bowel_obstruction_or_dilation,
            CTLabels.Diverticular_disease,
            CTLabels.Appendiceal_abnormality,
            CTLabels.Ascites,
            CTLabels.Peritoneal_omental_disease,
            CTLabels.Pneumoperitoneum,
            CTLabels.Abdominal_vascular_abnormality,
            CTLabels.Abdominal_vascular_calcification,
            CTLabels.Abdominal_lymphadenopathy,
            CTLabels.Abdominal_wall_hernia,
            CTLabels.Abdominal_soft_tissue_mass,
            CTLabels.Abdominopelvic_bone_lesion,
            CTLabels.Abdominopelvic_fracture,
            CTLabels.Abdominal_postoperative_change,
            CTLabels.Abdominal_medical_material,
            CTLabels.No_Abdominal_Finding,
        ]
