import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_PATH = PROJECT_ROOT / "datasets" / "raw"
SAVE_PATH = PROJECT_ROOT / "datasets" / "processed"
DASHBOARD_PATH = PROJECT_ROOT / "src" / "2_dashboard"
UTU_PATH = PROJECT_ROOT / "src" / "1_unified_task_universe"

TASK_EXTRACTION_CONFIG = {
    "file_paths": {
        "ssg_file": os.path.join(BASE_PATH, "Skills-Framework-Dataset-2024.xlsx"),
        "umap_crosswalk_file": os.path.join(BASE_PATH, "umap_ready_crosswalk.csv"),
        "semantic_crosswalk_file": os.path.join(BASE_PATH, "semantic_esco_crosswalk.csv"),
        "esco_occupations_file": os.path.join(BASE_PATH, "occupations_en.csv"),
        "onet_tasks_file": os.path.join(BASE_PATH, "Task_Statements.xlsx"),
        "onet_esco_crosswalk": os.path.join(BASE_PATH, "final_complete_esco_isco_crosswalk.xlsx"),
        "final_crosswalk": os.path.join(SAVE_PATH, "complete_crosswalk.csv"),
        "crosswalk_skills_tasks": os.path.join(SAVE_PATH, "crosswalks_skills_tasks.xlsx"),
        "llm_df_path": os.path.join(SAVE_PATH, "llm_meta_tasks.xlsx"),
        "master_jobs_with_umap": os.path.join(DASHBOARD_PATH, "jobs_and_meta_tasks.xlsx"),
        "umap_jobs": os.path.join(BASE_PATH, "umap_job_coordinates.csv"),
        "comparison_llms": os.path.join(SAVE_PATH, "llm_model_comparison.xlsx"),
        "onet_descriptions": os.path.join(BASE_PATH, "Occupation Data.xlsx"),
        "talent_landscape": os.path.join(BASE_PATH, "isco_codes_417_Including_PRESETS.xlsx"),
        "ref_df": os.path.join(BASE_PATH, "Reference_13065_AgeGenderEthnicityEduLevel.xlsx"),
        "umap_training_data": os.path.join(BASE_PATH, "final_cognitive_labeled.csv"),
        "archetype": os.path.join(BASE_PATH, "Archetypes_Spider.xlsx"),
        "archetype_with_recs": os.path.join(SAVE_PATH, "archetypes_with_model_recommendations.xlsx"),
        "ids": os.path.join(BASE_PATH, "Feedback collection participant IDs and codes.csv"),
        "cognitive_data1": os.path.join(BASE_PATH, "dashboard_cognitive_data.xlsx"),
        "description_nl": os.path.join(BASE_PATH, "occupations_nl.csv"),
        "dashboard_results": os.path.join(BASE_PATH, "Dashboard Study Results.xlsx"),  
    }
}
