import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
import ast
import datetime
import joblib
import numpy as np
from utils.config.dashboard_configuration import PRESAVED_LINEAR, DASHBOARD_CONFIG, TRANSLATIONS
from utils.functions.dashboard_functions import normalize_series, find_closest_jobs, predict_umap_coordinates, predict_coordinate_pc
from utils.constants import PREDICTORS_COMPLETE, PC2, PC3
from scipy.stats import zscore
import random

# --- HELPER FUNCTIONS ---
def t(key):
    """Fetches a translated string based on the selected language."""
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS[lang].get(key, f"_{key}_") # Fallback for missing keys

def get_score_map():
    """Returns the score map in the selected language."""
    lang = st.session_state.get('language', 'en')
    if lang == 'nl':
        return {
            "Vind ik absoluut niet leuk om te doen": -2,
            "Vind ik niet leuk om te doen": -1,
            "Neutraal": 0,
            "Vind ik leuk om te doen": 1,
            "Vind ik absoluut geweldig om te doen": 2
        }
    else: # Default to English
        return {
            "Strongly would not like doing this": -2,
            "Would not like doing this": -1,
            "Neutral": 0,
            "Would like doing this": 1,
            "Strongly would like doing this": 2
        }

FILE_PATHS = DASHBOARD_CONFIG['file_paths']

def connect_to_gsheet():
    """Connects to Google Sheets using Streamlit's secrets."""
    try:
        scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return None

def append_to_gsheet(client, sheet_name, tab_name, df_to_append):
    """Appends a pandas DataFrame to a specific tab in a Google Sheet."""
    if client is None:
        return
    try:
        spreadsheet = client.open(sheet_name)
        worksheet = spreadsheet.worksheet(tab_name)
        existing_df = pd.DataFrame(worksheet.get_all_records())
        updated_df = pd.concat([existing_df, df_to_append], ignore_index=True)
        worksheet.clear()
        set_with_dataframe(worksheet, updated_df)
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Tab '{tab_name}' not found in '{sheet_name}'. Please create it.")
    except Exception as e:
        st.error(f"Failed to write to Google Sheets: {e}")

def submit_ratings_data(client):
    """Compiles and submits the task and recommendation ratings to Google Sheets."""
    if client is None:
        st.warning("Could not connect to the database. Your ratings could not be saved.")
        return

    try:
        relevance_data = []
        for job_id, sources in st.session_state.recommendation_source_map.items():
            job_row = jobs_df.loc[jobs_df['unique_id'] == job_id]
            if not job_row.empty:
                job_title = job_row['job_title'].iloc[0]
                relevance_score = st.session_state.relevance_scores.get(job_id, 5) # Default to neutral
                for source in sources:
                    relevance_data.append({
                        "Participant ID": st.session_state.participant_id,
                        "Job Title": job_title,
                        "Recommendation Type": source,
                        "Relevance Score": relevance_score
                    })
        relevance_df = pd.DataFrame(relevance_data)
        if not relevance_df.empty:
            relevance_df.insert(1, "Timestamp", datetime.datetime.now())
            append_to_gsheet(client, "Dashboard Study Results", "Relevance Feedback", relevance_df)

        scores_df = pd.DataFrame.from_dict(st.session_state.scores, orient='index', columns=['score']).reset_index().rename(columns={'index': 'meta_task_id'})
        if not scores_df.empty:
            scores_df.insert(0, "Participant ID", st.session_state.participant_id)
            scores_df.insert(1, "Timestamp", datetime.datetime.now())
            append_to_gsheet(client, "Dashboard Study Results", "Task Scores", scores_df)

    except Exception as e:
        st.error(f"An error occurred while saving your data: {e}")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """
    Loads all required data. Task descriptions are loaded for both languages
    and returned as a dictionary.
    """
    try:
        FILE_PATHS = DASHBOARD_CONFIG['file_paths']
        tasks_df_en = pd.read_excel(FILE_PATHS['llm_meta_tasks'])
        tasks_df_nl = pd.read_excel(FILE_PATHS['llm_meta_tasks_nl'])
        all_tasks_df = {'en': tasks_df_en, 'nl': tasks_df_nl}

        jobs_df = pd.read_excel(FILE_PATHS["jobs_meta_tasks"])
        jobs_df['unique_id'] = jobs_df.index
        jobs_df['meta_task_ids'] = jobs_df['meta_task_ids'].apply(ast.literal_eval)
        
        cognitive_df = pd.read_excel(FILE_PATHS["cognitive_data1"])
        
        return all_tasks_df, jobs_df, cognitive_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure all required Excel files are present.")
        return None, None, None

all_tasks_df, jobs_df, cognitive_df = load_data()

# --- INITIALIZE SESSION STATE (MODIFIED) ---
if 'current_task_index' not in st.session_state:
    st.session_state.language = None
    st.session_state.current_task_index = 0
    st.session_state.scores = {}
    st.session_state.assessment_complete = False
    # Removed st.session_state.open_questions_complete
    st.session_state.start_assessment = False
    st.session_state.task_instructions_seen = False
    st.session_state.recommendation_instructions_seen = False
    st.session_state.feedback_submitted = False
    st.session_state.results_generated = False
    st.session_state.recommendation_rating_complete = False
    st.session_state.current_recommendation_index = 0
    st.session_state.relevance_scores = {}
    st.session_state.unique_recommendations_df = None
    st.session_state.recommendation_source_map = {}
    st.session_state.celebration_shown = False
    st.session_state.participant_id_submitted = False
    st.session_state.participant_id = None
    st.session_state.cognitive_scores = None

# --- LANGUAGE SELECTION SCREEN  ---
if st.session_state.language is None:
    st.title("Select Language / Selecteer Taal")
    lang_choice = st.radio(
        "Please select your language. / Selecteer alstublieft uw taal.",
        ('English', 'Nederlands'),
        key='lang_select'
    )
    if st.button("Confirm / Bevestigen", type="primary", use_container_width=True):
        st.session_state.language = 'nl' if lang_choice == 'Nederlands' else 'en'
        
        if all_tasks_df:
            tasks_df = all_tasks_df[st.session_state.language]
            st.session_state.randomized_tasks_df = tasks_df.sample(frac=1).reset_index(drop=True)
        
        st.rerun()

# --- MAIN APP LOGIC ---
elif all_tasks_df is not None and jobs_df is not None and cognitive_df is not None:

    st.set_page_config(
        page_title=t("page_title"),
        page_icon="📊",
        layout="centered"
    )
    
    SCORE_MAP = get_score_map()
    st.image(FILE_PATHS['brainsfirstlogo'], width=300)

    if not st.session_state.participant_id_submitted:
        st.title(t("welcome_id_title"))
        st.markdown("---")
        participant_id_input = st.text_input(t("id_prompt"), placeholder=t("id_placeholder"))

        if st.button(t("submit_id_button"), type="primary", use_container_width=True):
            if participant_id_input:
                p_id = participant_id_input.strip()
                participant_data = cognitive_df[cognitive_df['participant_id'].astype(str) == p_id]
                
                if not participant_data.empty:
                    scores_row = participant_data.iloc[0].drop('participant_id')
                    st.session_state.cognitive_scores = scores_row.to_dict()
                    st.session_state.participant_id = p_id
                    st.session_state.participant_id_submitted = True
                    st.rerun()
                else:
                    st.error(t("id_not_found_error"))
            else:
                st.warning(t("id_warning"))
    else:
        if not st.session_state.start_assessment:
            st.title(t("welcome_title"))
            st.markdown(f"#### Participant ID: `{st.session_state.participant_id}`")
            st.markdown(t("welcome_markdown"))
            st.markdown("---")
            if st.button(t("start_button"), type="primary", use_container_width=True):
                st.session_state.start_assessment = True
                st.rerun()
        else:
            if not st.session_state.task_instructions_seen:
                st.title(t("task_instructions_title"))
                st.markdown("---")
                st.markdown(t("task_instructions_markdown"))
                st.markdown("---")
                if st.button(t("continue_button"), type="primary", use_container_width=True):
                    st.session_state.task_instructions_seen = True
                    st.rerun()
            else: 
                if st.session_state.assessment_complete:
                    if not st.session_state.results_generated:
                        with st.spinner(t("spinner_text")):
                            AVOIDED_NEGATIVE_BONUS = 0.5 
                            MISSED_POSITIVE_PENALTY = 0.5
                            task_scores_data = []
                            for _, job_row in jobs_df.iterrows():
                                total_score = 0
                                job_task_set = set(job_row['meta_task_ids'])
                                for task_id, user_score in st.session_state.scores.items():
                                    if task_id in job_task_set: total_score += user_score
                                    else:
                                        if user_score == 2: total_score -= MISSED_POSITIVE_PENALTY
                                        elif user_score == 1: total_score -= (MISSED_POSITIVE_PENALTY / 2)
                                        elif user_score == -2: total_score += AVOIDED_NEGATIVE_BONUS
                                        elif user_score == -1: total_score += (AVOIDED_NEGATIVE_BONUS / 2)
                                task_scores_data.append({"unique_id": job_row['unique_id'], "raw_task_score": total_score})
                                
                            task_scores_df = pd.DataFrame(task_scores_data)
                            min_score, max_score = task_scores_df['raw_task_score'].min(), task_scores_df['raw_task_score'].max()
                            task_scores_df['task_score'] = 100.0 if max_score == min_score else ((task_scores_df['raw_task_score'] - min_score) / (max_score - min_score)) * 100

                            user_cognitive_scores = st.session_state.cognitive_scores
                            user_pc2 = predict_coordinate_pc(PC2, user_cognitive_scores)
                            user_pc3 = predict_coordinate_pc(PC3, user_cognitive_scores)
                            jobs_with_dist_df = jobs_df.copy()
                            distances = np.sqrt((jobs_with_dist_df['PC2'] - user_pc2)**2 + (jobs_with_dist_df['PC3'] - user_pc3)**2)
                            jobs_with_dist_df['pc_distance'] = distances
                            jobs_with_dist_df['cognitive_similarity'] = 1 / (1 + jobs_with_dist_df['pc_distance'])
                            jobs_with_dist_df['cognitive_score'] = normalize_series(jobs_with_dist_df['cognitive_similarity']) 
                            cognitive_scores_df = jobs_with_dist_df[['unique_id', 'cognitive_score']]                           

                            combined_scores_df = pd.merge(task_scores_df, cognitive_scores_df, on="unique_id")
                            combined_scores_df['task_zscore'] = zscore(combined_scores_df['task_score'])
                            combined_scores_df['cognitive_zscore'] = zscore(combined_scores_df['cognitive_score'])
                            combined_scores_df['total_match_score'] = (0.5 * combined_scores_df['task_zscore']) + (0.5 * combined_scores_df['cognitive_zscore'])
                            final_results_df = pd.merge(jobs_df, combined_scores_df, on="unique_id")

                            recommendation_sets = {
                                "Preference": final_results_df.sort_values(by="task_score", ascending=False).head(5),
                                "Cognitive": final_results_df.sort_values(by="cognitive_score", ascending=False).head(5),
                                "Combined": final_results_df.sort_values(by="total_match_score", ascending=False).head(5)
                            }
                                
                            recommended_ids = set(id for df in recommendation_sets.values() for id in df['unique_id'])
                            job_pool = jobs_df[~jobs_df['unique_id'].isin(recommended_ids)]
                            num_random_jobs = min(7, len(job_pool))
                            if not job_pool.empty:
                                recommendation_sets["Random"] = job_pool.sample(n=num_random_jobs, random_state=42)
                            else:
                                recommendation_sets["Random"] = pd.DataFrame(columns=jobs_df.columns)

                            source_map = {}
                            unique_job_ids_ordered = []
                            for rec_type, df in recommendation_sets.items():
                                for _, row in df.iterrows():
                                    job_id = row['unique_id']
                                    if job_id not in source_map:
                                        source_map[job_id] = []
                                        unique_job_ids_ordered.append(job_id)
                                    source_map[job_id].append(rec_type)

                            st.session_state.recommendation_source_map = source_map
                            st.session_state.unique_recommendations_df = jobs_df[jobs_df['unique_id'].isin(unique_job_ids_ordered)].set_index('unique_id').loc[unique_job_ids_ordered].reset_index()
                            st.session_state.results_generated = True
                        
                    if not st.session_state.celebration_shown:
                        st.balloons()
                        st.session_state.celebration_shown = True

                    if not st.session_state.recommendation_instructions_seen:
                        st.title(t("recommendation_instructions_title"))
                        st.markdown("---")
                        st.markdown(t("recommendation_instructions_markdown"))
                        st.markdown("---")
                        if st.button(t("continue_button"), type="primary", use_container_width=True):
                            st.session_state.recommendation_instructions_seen = True
                            st.rerun()
                    else:
                        if not st.session_state.recommendation_rating_complete:
                            st.header(t("recommendations_header"))
                            st.markdown("---")
                            st.markdown(t("recommendations_rating_prompt"))

                            rec_df = st.session_state.unique_recommendations_df
                            num_recs = len(rec_df)
                            current_index = st.session_state.current_recommendation_index
                                
                            if current_index < num_recs:
                                current_job = rec_df.iloc[current_index]
                                job_id = current_job['unique_id']
                                progress_text = t("progress_recommendation_text").format(current=current_index + 1, total=num_recs)
                                st.progress((current_index) / num_recs, text=progress_text)
                                
                                if st.session_state.language == 'nl':
                                    job_title = current_job['job_title_nl']
                                    job_description = current_job.get('description_nl')
                                    fallback_description = "Geen beschrijving beschikbaar."
                                else:
                                    job_title = current_job['job_title']
                                    job_description = current_job.get('description')
                                    fallback_description = "No description available."
                                
                                if pd.isna(job_description) or not job_description:
                                    job_description = fallback_description
                                    
                                st.subheader(job_title)
                                st.caption(job_description)
                                st.markdown("---")
                                st.write(t("relevance_question"))
                                cols = st.columns(9)
                                for i in range(1, 10):
                                    if cols[i-1].button(str(i), key=f"likert_{i}_{job_id}", use_container_width=True):
                                        st.session_state.relevance_scores[job_id] = i
                                        st.session_state.current_recommendation_index += 1
                                        
                                        if st.session_state.current_recommendation_index >= num_recs:
                                            st.session_state.recommendation_rating_complete = True
                                            
                                            gsheet_client = connect_to_gsheet()
                                            submit_ratings_data(gsheet_client)
                                            st.session_state.feedback_submitted = True
                                        
                                        st.rerun()
                            else:
                                st.session_state.recommendation_rating_complete = True
                            
                        else: 
                            # Open Questions block removed here
                            st.header(t("thank_you_header"))
                            st.success(t("feedback_submitted_success"))
                            st.balloons()
                            
                            if st.button(t("restart_button")):
                                st.session_state.clear()
                                st.rerun()
                else: 
                    st.title(t("task_rating_title"))
                    randomized_tasks = st.session_state.randomized_tasks_df
                    num_tasks = len(randomized_tasks)
                    current_index = st.session_state.current_task_index
                    current_task = randomized_tasks.iloc[current_index]
                    task_id = current_task['meta_task_id']
                    progress_text = t("progress_task_text").format(current=current_index + 1, total=num_tasks)
                    st.progress((current_index) / num_tasks, text=progress_text)
                    st.subheader(current_task['title'])
                    st.write(current_task['description'])
                    st.markdown("---")
                    cols = st.columns(5)
                    button_labels = list(SCORE_MAP.keys())
                    for i, label in enumerate(button_labels):
                        if cols[i].button(label, use_container_width=True, key=f"rate_{label}_{task_id}"):
                            st.session_state.scores[task_id] = SCORE_MAP[label]
                            next_index = st.session_state.current_task_index + 1
                            if next_index >= num_tasks:
                                st.session_state.assessment_complete = True
                            else:
                                st.session_state.current_task_index = next_index
                            st.rerun()