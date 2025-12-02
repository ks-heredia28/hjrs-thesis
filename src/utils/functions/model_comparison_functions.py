import pandas as pd
import numpy as np
from scipy.stats import zscore

def predict_coordinate_pc(model_coeffs, scores):
    """
    Predicts a coordinate based on a linear model.
    'scores' should be a dictionary of a user's cognitive scores.
    """
    coordinate = model_coeffs.get('Intercept', 0)
    for feature in model_coeffs:
        if feature != 'Intercept':
            score = scores.get(feature, 0)
            coefficient = model_coeffs[feature]
            coordinate += coefficient * score
    return coordinate

def calculate_raw_task_score(job_row, prefs, bonus, penalty):
    """Calculates a raw preference score for a single job."""
    total_score = 0
    job_task_set = set(job_row['meta_task_ids']) 
    
    for task_id, score in prefs.items():
        if task_id in job_task_set:
            total_score += score
        else:
            if score == 2: total_score -= penalty
            elif score == 1: total_score -= (penalty / 2)
            elif score == -2: total_score += bonus
            elif score == -1: total_score += (bonus / 2)
    return total_score

def normalize_series(series, new_min=0, new_max=100):
    """Min-max normalization for a pandas Series."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([new_max] * len(series), index=series.index)
    return (series - min_val) * (new_max - new_min) / (max_val - min_val) + new_min

# --- New Main Calculation Function ---

def calculate_all_scores_per_participant(task_scores_df, cognitive_scores_df, jobs_df, 
                                       pc2_coeffs, pc3_coeffs, bonus, penalty):
    """
    Calculates task, cognitive, and hybrid scores for all jobs for all participants.
    
    Args:
        task_scores_df (pd.DataFrame): Participant task ratings ['Participant ID', 'meta_task_ids', 'score']
        cognitive_scores_df (pd.DataFrame): Participant cognitive scores, with 'Participant ID'
        jobs_df (pd.DataFrame): All jobs, must include ['job_title', 'meta_task_ids', 'PC2', 'PC3', 'unique_id']
        pc2_coeffs (dict): Coefficients for PC2 model
        pc3_coeffs (dict): Coefficients for PC3 model
        bonus (float): Bonus for avoided negative tasks
        penalty (float): Penalty for missed positive tasks
        
    Returns:
        pd.DataFrame: A long-format DataFrame with all scores for all participants and all jobs.
    """
    
    all_scores = []
    
    # Prepare cognitive scores for easy lookup
    # Set index to 'Participant ID' and drop 'Intercept' if it's a column
    cog_scores_lookup = cognitive_scores_df.set_index('Participant ID')
    if 'Intercept' in cog_scores_lookup.columns:
        cog_scores_lookup = cog_scores_lookup.drop(columns=['Intercept'])
        
    participant_ids = task_scores_df['Participant ID'].unique()
    grouped_task_prefs = task_scores_df.groupby('Participant ID')
    
    for participant_id in participant_ids:
        
        # --- 1. Get Participant's Data ---
        try:
            # Get task preferences
            participant_prefs_df = grouped_task_prefs.get_group(participant_id)
            preferences_dict = dict(zip(participant_prefs_df['meta_task_ids'], 
                                        participant_prefs_df['score']))
            
            # Get cognitive scores
            user_cognitive_scores = cog_scores_lookup.loc[participant_id].to_dict()
        except KeyError:
            print(f"Skipping Participant ID {participant_id}: Missing cognitive data or task data.")
            continue

        # Make a fresh copy of jobs_df for this participant
        participant_jobs_df = jobs_df.copy()

        # --- 2. Calculate Task Score (0-100) ---
        participant_jobs_df['raw_task_score'] = participant_jobs_df.apply(
            lambda row: calculate_raw_task_score(row, preferences_dict, bonus, penalty), 
            axis=1
        )
        participant_jobs_df['task_score'] = normalize_series(participant_jobs_df['raw_task_score'])

        # --- 3. Calculate Cognitive Score (0-100) ---
        user_pc2 = predict_coordinate_pc(pc2_coeffs, user_cognitive_scores)
        user_pc3 = predict_coordinate_pc(pc3_coeffs, user_cognitive_scores)
        
        distances = np.sqrt((participant_jobs_df['PC2'] - user_pc2)**2 + 
                            (participant_jobs_df['PC3'] - user_pc3)**2)
        
        participant_jobs_df['pc_distance'] = distances
        participant_jobs_df['cognitive_similarity'] = 1 / (1 + participant_jobs_df['pc_distance'])
        participant_jobs_df['cognitive_score'] = normalize_series(participant_jobs_df['cognitive_similarity'])

        # --- 4. Calculate Hybrid Score (Z-scores) ---
        # Z-score calculation is done *per participant* across all jobs
        participant_jobs_df['task_zscore'] = zscore(participant_jobs_df['task_score'])
        participant_jobs_df['cognitive_zscore'] = zscore(participant_jobs_df['cognitive_score'])
        
        participant_jobs_df['hybrid_score'] = (0.5 * participant_jobs_df['task_zscore']) + \
                                              (0.5 * participant_jobs_df['cognitive_zscore'])
        
        # --- 5. Store Results ---
        participant_jobs_df['Participant ID'] = participant_id
        all_scores.append(participant_jobs_df)

    if not all_scores:
        return pd.DataFrame()
        
    final_df = pd.concat(all_scores).reset_index(drop=True)
    return final_df