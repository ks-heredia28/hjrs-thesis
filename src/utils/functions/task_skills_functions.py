import pandas as pd
import numpy as np
import ast
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.stats import zscore
from scipy.spatial import KDTree
import ipywidgets as widgets
from ipywidgets import HBox, VBox, interactive_output
from utils.constants import PC2, PC3, PREDICTORS_COMPLETE

def calculate_initial_task_coords(df, tasks):
    """Calculates task coordinates by averaging job coordinates."""
    df['meta_task_ids'] = df['meta_task_ids'].apply(ast.literal_eval)
    df_long = df.explode('meta_task_ids')
    task_coordinates = df_long.groupby('meta_task_ids')[['PRESET_PC2', 'PRESET_PC3']].mean().reset_index()
    final_task_df = pd.merge(task_coordinates, tasks, on='meta_task_ids')
    return final_task_df

def _parse_to_list(val):
    """Helper to safely parse string representations of lists."""
    if isinstance(val, (list, np.ndarray)):
        return val
    if pd.isna(val):
        return []
    try:
        evaluated = ast.literal_eval(str(val))
        return evaluated if isinstance(evaluated, list) else [evaluated]
    except (ValueError, SyntaxError):
        return [str(val)]

def flatten_job_tasks(master_jobs, task_columns):
    """Flattens job data from multiple task columns into a long format."""
    all_job_tasks = []
    for col in task_columns:
        if col in master_jobs.columns:
            temp_df = master_jobs[['ISCO_code', 'ESCO or ISCO Title', col]].copy()
            temp_df.dropna(subset=[col], inplace=True)
            temp_df[col] = temp_df[col].apply(_parse_to_list)
            temp_df = temp_df.explode(col)
            temp_df.rename(columns={col: 'task_text'}, inplace=True)
            all_job_tasks.append(temp_df)
    
    job_tasks_long = pd.concat(all_job_tasks, ignore_index=True)
    job_tasks_long.dropna(subset=['task_text'], inplace=True)
    job_tasks_long['task_text'] = job_tasks_long['task_text'].astype(str).str.strip()
    return job_tasks_long

def calculate_refined_task_coords(job_tasks_long, tasks_df_path, df, tasks, top_n=3):
    """
    Refines task coordinates based on top N contributing jobs and
    returns both the coordinates and the top contributor mapping.
    """
    try:
        tasks_df = pd.read_pickle(tasks_df_path)
    except FileNotFoundError:
        print(f"Error: Tasks file not found at {tasks_df_path}")
        return pd.DataFrame(), pd.DataFrame()
        
    task_cluster_map = tasks_df[tasks_df['cluster_id'] != -1][['task_text', 'cluster_id']]
    job_meta_tasks = pd.merge(job_tasks_long, task_cluster_map, on='task_text', how='inner')
    contribution_counts = job_meta_tasks.groupby(['ISCO_code', 'ESCO or ISCO Title', 'cluster_id']).size().reset_index(name='raw_task_count')
    top_contributors = contribution_counts.sort_values('raw_task_count', ascending=False).groupby('cluster_id').head(top_n)
    
    df_temp = df.copy()
    if 'job_code' in df_temp.columns:
         df_temp.rename(columns={'job_code': 'ISCO_code'}, inplace=True)
    top_jobs_with_coords = pd.merge(top_contributors, df_temp[['ISCO_code', 'job_title', 'PRESET_PC2', 'PRESET_PC3']], on='ISCO_code', how='left')
    if top_jobs_with_coords[['PRESET_PC2', 'PRESET_PC3']].isnull().any().any():
        print("Warning: Some top contributing jobs are missing coordinates.")
        top_jobs_with_coords.dropna(subset=['PRESET_PC2', 'PRESET_PC3'], inplace=True)

    refined_coords = top_jobs_with_coords.groupby('cluster_id')[['PRESET_PC2', 'PRESET_PC3']].mean().reset_index()
    refined_coords.rename(columns={'cluster_id': 'meta_task_ids'}, inplace=True)

    final_refined_df = pd.merge(refined_coords, tasks, on='meta_task_ids', how='left')
    return final_refined_df, top_contributors

def calculate_refined_task_coords_unscaled(job_tasks_long, tasks_df_path, df, tasks, top_n=3):
    """
    Refines task coordinates based on top N contributing jobs and
    returns both the coordinates and the top contributor mapping.
    """
    try:
        tasks_df = pd.read_pickle(tasks_df_path)
    except FileNotFoundError:
        print(f"Error: Tasks file not found at {tasks_df_path}")
        return pd.DataFrame(), pd.DataFrame()
        
    task_cluster_map = tasks_df[tasks_df['cluster_id'] != -1][['task_text', 'cluster_id']]
    job_meta_tasks = pd.merge(job_tasks_long, task_cluster_map, on='task_text', how='inner')
    contribution_counts = job_meta_tasks.groupby(['ISCO_code', 'ESCO or ISCO Title', 'cluster_id']).size().reset_index(name='raw_task_count')
    top_contributors = contribution_counts.sort_values('raw_task_count', ascending=False).groupby('cluster_id').head(top_n)
    
    df_temp = df.copy()
    if 'job_code' in df_temp.columns:
         df_temp.rename(columns={'job_code': 'ISCO_code'}, inplace=True)
    top_jobs_with_coords = pd.merge(top_contributors, df_temp[['ISCO_code', 'job_title', 'PC2', 'PC3']], on='ISCO_code', how='left')
    if top_jobs_with_coords[['PC2', 'PC3']].isnull().any().any():
        print("Warning: Some top contributing jobs are missing coordinates.")
        top_jobs_with_coords.dropna(subset=['PC2', 'PC3'], inplace=True)

    refined_coords = top_jobs_with_coords.groupby('cluster_id')[['PC2', 'PC3']].mean().reset_index()
    refined_coords.rename(columns={'cluster_id': 'meta_task_ids'}, inplace=True)

    final_refined_df = pd.merge(refined_coords, tasks, on='meta_task_ids', how='left')
    return final_refined_df, top_contributors

# --- User Scoring and Recommendation ---

def predict_coordinate_pc(model_coeffs, scores):
    """
    Predicts a coordinate based on a linear model.
    """
    coordinate = model_coeffs.get('Intercept', 0)
    for feature in model_coeffs:
        if feature != 'Intercept':
            score = scores.get(feature, 0)
            coefficient = model_coeffs[feature]
            coordinate += coefficient * score
            
    return coordinate

def find_closest_jobs_pc(my_coords, jobs_df, top_n=5):
    my_pc2, my_pc3 = my_coords
    distances = np.sqrt((jobs_df['PC2'] - my_pc2)**2 + (jobs_df['PC3'] - my_pc3)**2)
    jobs_df['distance'] = distances
    return jobs_df.sort_values(by='distance').head(top_n)

def get_pc_recommendations(archetype_row, jobs_df):
    """Calculates top 5 PC job recommendations for a single person/archetype."""
    my_pc2 = predict_coordinate_pc(PC2, archetype_row)
    my_pc3 = predict_coordinate_pc(PC3, archetype_row)
    closest_jobs = find_closest_jobs_pc((my_pc2, my_pc3), jobs_df.copy(), top_n=5)
    return closest_jobs['job_title'].tolist(), my_pc2, my_pc3

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

def calculate_normalized_task_scores(df, preferences_dict, bonus, penalty):
    """Calculates and normalizes task scores for all jobs."""
    df['raw_task_score'] = df.apply(
        lambda row: calculate_raw_task_score(row, preferences_dict, bonus, penalty), 
        axis=1
    )
    min_score, max_score = df['raw_task_score'].min(), df['raw_task_score'].max()
    if max_score == min_score:
        df['task_score'] = 100.0
    else:
        df['task_score'] = 100 * (df['raw_task_score'] - min_score) / (max_score - min_score)
    return df

# --- 3D Landscape Data Generation ---

def generate_3d_landscape_data(df, user_coords, preferences_dict, bonus, penalty):
    """Generates all necessary data components for the 3D surface plot."""
    user_pc2, user_pc3 = user_coords
    
    # 1. Calculate task scores
    df_scored = calculate_normalized_task_scores(df.copy(), preferences_dict, bonus, penalty)
    
    # 2. Create grid, expanding slightly for user coordinates
    pc2_min, pc2_max = min(df['PC2'].min(), user_pc2) - 1, max(df['PC2'].max(), user_pc2) + 1
    pc3_min, pc3_max = min(df['PC3'].min(), user_pc3) - 1, max(df['PC3'].max(), user_pc3) + 1
    grid_pc2, grid_pc3 = np.mgrid[pc2_min:pc2_max:100j, pc3_min:pc3_max:100j]
    
    # 3. Calculate cognitive similarity grid
    dist_grid = np.sqrt((grid_pc2 - user_pc2)**2 + (grid_pc3 - user_pc3)**2)
    cognitive_similarity_grid = 1 / (1 + dist_grid)
    
    # 4. Interpolate task scores onto the grid
    points = df_scored[['PC2', 'PC3']].values
    values = df_scored['task_score'].values
    task_score_grid = griddata(points, values, (grid_pc2, grid_pc3), method='linear')
    task_score_grid = np.nan_to_num(task_score_grid, nan=np.mean(values))
    
    # 5. Combine scores using z-score
    cognitive_z = zscore(cognitive_similarity_grid.flatten())
    task_z = zscore(task_score_grid.flatten())
    final_score_flat = (0.5 * cognitive_z) + (0.5 * task_z)
    z_surface = final_score_flat.reshape(grid_pc2.shape)
    
    # 6. Create hover text for grid points
    kdtree = KDTree(df[['PC2', 'PC3']].values)
    grid_points = np.vstack([grid_pc2.ravel(), grid_pc3.ravel()]).T
    _, indices = kdtree.query(grid_points, k=3)
    hover_texts = [
        f"<b>Nearby Jobs:</b><br>- {df['job_title'].iloc[idx[0]]}<br>- {df['job_title'].iloc[idx[1]]}<br>- {df['job_title'].iloc[idx[2]]}"
        for idx in indices
    ]
    hover_text_grid = np.array(hover_texts).reshape(grid_pc2.shape)
    
    return {
        "grid_pc2": grid_pc2, "grid_pc3": grid_pc3, "z_surface": z_surface,
        "hover_text_grid": hover_text_grid, "df_scored": df_scored
    }

# --- Plotting Functions ---

def plot_occupational_space_2d(df_jobs, df_tasks, plot_config, job_style, task_style, normalized=False):
    """Creates a 2D scatter plot of jobs and tasks."""
    fig = go.Figure()
    x_col, y_col = ('PRESET_PC2', 'PRESET_PC3')
    
    fig.add_trace(go.Scatter(
        x=df_jobs[x_col], y=df_jobs[y_col], mode='markers',
        marker=job_style, hovertext=df_jobs['job_title'], name='Jobs'
    ))
    fig.add_trace(go.Scatter(
        x=df_tasks[x_col], y=df_tasks[y_col], mode='markers',
        marker=task_style, hovertext=df_tasks['title'], name='Tasks'
    ))
    
    layout_update = {
        'title': 'Talent Landscape: Jobs and Tasks' if normalized else 'Occupational Space: Jobs and Tasks',
        'xaxis_title': 'PC2', 'yaxis_title': 'PC3',
        'legend_title_text': 'Data Points'
    }
    if normalized:
        layout_update['yaxis'] = dict(scaleanchor="x", scaleratio=1)
        
    fig.update_layout(**plot_config, **layout_update)
    fig.show()


def plot_3d_landscape(landscape_data, df, user_coords, layout_config, job_style, user_style):
    """Creates a 3D surface plot of the personalized talent landscape."""
    g = landscape_data # shorthand
    user_pc2, user_pc3 = user_coords
    
    fig = go.Figure(data=[go.Surface(
        z=g['z_surface'], x=g['grid_pc2'], y=g['grid_pc3'],
        colorscale='Viridis', opacity=0.8, name='Match Surface',
        text=g['hover_text_grid'], hoverinfo='text+z'
    )])
    
    # Interpolate Z values for jobs and user for accurate plotting on the surface
    grid_points_flat = (g['grid_pc2'].flatten(), g['grid_pc3'].flatten())
    z_surface_flat = g['z_surface'].flatten()
    job_z_values = griddata(grid_points_flat, z_surface_flat, (df['PC2'], df['PC3']), method='linear')
    user_z_value = griddata(grid_points_flat, z_surface_flat, (user_pc2, user_pc3), method='linear')

    fig.add_trace(go.Scatter3d(
        x=df['PC2'], y=df['PC3'], z=job_z_values, mode='markers',
        marker=job_style, hovertext=df['job_title'], name='Jobs'
    ))
    fig.add_trace(go.Scatter3d(
        x=[user_pc2], y=[user_pc3], z=user_z_value, mode='markers',
        marker=user_style, name='Your Position'
    ))
    
    fig.update_layout(
        title='Personalized 3D Talent Landscape',
        scene=dict(xaxis_title='PC2', yaxis_title='PC3', zaxis_title='Overall Match Score'),
        autosize=False,
        **layout_config
    )
    return fig

    # --- Interactive Dashboard ---

def create_interactive_dashboard(df, tasks, user_coords, bonus, penalty):
    """Creates and displays an interactive 3D landscape dashboard."""
    user_pc2, user_pc3 = user_coords
    pc2_min, pc2_max = df['PC2'].min(), df['PC2'].max()
    pc3_min, pc3_max = df['PC3'].min(), df['PC3'].max()
    grid_pc2, grid_pc3 = np.mgrid[pc2_min:pc2_max:100j, pc3_min:pc3_max:100j]

    fig = go.FigureWidget(layout=go.Layout(
        title='Interactive Talent Landscape',
        scene=dict(
            xaxis_title='PC2', yaxis_title='PC3', zaxis_title='Task Score',
            zaxis=dict(range=[0, 100])
        ),
        width=700, height=700
    ))

    # Add plot elements to the FigureWidget
    fig.add_surface(z=np.zeros(grid_pc2.shape), x=grid_pc2, y=grid_pc3, colorscale='Viridis', opacity=0.8, colorbar=dict(title='Task Score'))
    fig.add_scatter3d(x=df['PC2'], y=df['PC3'], z=np.zeros(len(df)), mode='markers', marker=dict(size=3, color='blue'), name='Jobs', hovertext=df['job_title'], hoverinfo='text')
    fig.add_scatter3d(x=[user_pc2], y=[user_pc3], z=[0], mode='markers', marker=dict(size=8, color='black', symbol='diamond'), name='Your Position')

    task_name_to_id = dict(zip(tasks.title, tasks.meta_task_ids))
    sliders = {
        row['title']: widgets.IntSlider(min=-2, max=2, value=0, description=row['title'], continuous_update=False)
        for _, row in tasks.iterrows()
    }

    def update_landscape(**kwargs):
        preferences = {task_name_to_id[name]: value for name, value in kwargs.items()}
        df_scored = calculate_normalized_task_scores(df.copy(), preferences, bonus, penalty)
        
        points = df_scored[['PC2', 'PC3']].values
        new_z_surface = griddata(points, df_scored['task_score'], (grid_pc2, grid_pc3), method='linear')
        new_z_surface = np.nan_to_num(new_z_surface, nan=np.mean(df_scored['task_score']))
        
        flat_grid = (grid_pc2.flatten(), grid_pc3.flatten())
        flat_surface = new_z_surface.flatten()
        job_z = griddata(flat_grid, flat_surface, (df['PC2'], df['PC3']), method='linear')
        user_z = griddata(flat_grid, flat_surface, user_coords, method='linear')

        with fig.batch_update():
            fig.data[0].z = new_z_surface
            fig.data[1].z = job_z
            fig.data[2].z = user_z

    out = interactive_output(update_landscape, sliders)
    slider_box = VBox(list(sliders.values()), layout={'width': '350px'})
    dashboard = HBox([slider_box, fig])
    
    # Trigger initial plot state
    update_landscape(**{name: slider.value for name, slider in sliders.items()})
    
    return dashboard