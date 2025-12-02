import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_PATH = PROJECT_ROOT / "datasets" / "raw"

CONFIG = {
    "file_paths": {
        "my_scores": os.path.join(BASE_PATH, "saya.xlsx"),
        "my_task_preferences": os.path.join(BASE_PATH, "my_task_preferences.xlsx"),
        "ilja_scores": os.path.join(BASE_PATH, "ilja_scores.xlsx"),
        "ilja_preferences": os.path.join(BASE_PATH, "ilja_prefereces.xlsx"),
        "ref_df": os.path.join(BASE_PATH, "HCM2023_with_region_origin.xlsx"),
        "labeled_data": os.path.join(BASE_PATH, "final_cognitive_labeled.csv"),
    }
}

# Plot configurations
PLOT_LAYOUT = {
    'template': 'plotly_white',
    'width': 700,
    'height': 700
}

JOBS_MARKER_STYLE_2D = {
    'color': 'blue',
    'size': 8,
    'opacity': 0.6
}

TASKS_MARKER_STYLE_2D = {
    'color': 'red',
    'size': 10,
    'symbol': 'star'
}

JOBS_MARKER_STYLE_3D = {
    'size': 3,
    'color': 'blue',
    'symbol': 'circle',
    'opacity': 0.4
}

USER_MARKER_STYLE_3D = {
    'size': 8,
    'color': 'black',
    'symbol': 'diamond'
}