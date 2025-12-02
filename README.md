# Hybrid Job Recommendation System Thesis

This is the codebase for the BDS Master's Thesis titled "A Hybrid Job Recommendation System: Integrating User's Task Preferences and Cognitive Profile for Better Career Guidance". This project aims to develop a system that provides users a way to express their work activity prefereces, combine them with their cognitive scores, and recommend suitable jobs.

## Dependencies
To ensure everything runs smoothly, make sure the required dependencies are installed. Use the following command in your terminal:
```{bash}
uv sync
```

## Data Files
The data folder called `datasets` contains all the relevant files to run this smoothly. This folder has two subdirectories:
* `raw`: Contains unaltered original data files.
* `processed`: Contains intermediate and final files produced by the code.
All output files are already present in `processed`, so you can run the notebooks in different orders. 

Inside `src/2_dashboard`, there are also some files needed for the interactive dashboard to run. However, to project participants' privacy, the real cognitive information inside "cognitive_data_1.xlsx" has been deleted, and only a thesis example remains. 

## Project Structure
In the `src` folder (source code), you'll find three main subfolders representing three key parts of this project:
1. `1_unified_task_universe`: Creating the list of jobs and meta-tasks.
2. `2_dashboard`: The necessary files and scripts for the interactive dashboard.
3. `3_results_analysis`: Analyzing the collected data to see which model is better.

The first and third folder contain a Jupyter notebook each. If you just want to recreate this project, you can simply press *"Run All"* at the top of each notebook.

### `task_extraction.ipynb`
Everything is ready to just be run again without the need of API keys for OpenRouter and Gemini. However, if changes want to be made with the LLM outputs, then the API keys need to be saved as environment variables.
At the end of the notebook is the 3D representation of the Unified Task Universe for observance. 

### `interactive_dashboard.py`
To see the dashboard in function paste and run the following code in your terminal:
```{bash}
cd src/2_dashboard
python3 -m streamlit run interactive_dashboard.py
```
A local host page will open up in you computer. 
Select the language of your preference, then enter the access code **thesis_example**. You should be able to carry out the entire dashboard with no problems. The results from this session will not be saved anywhere. The deployed version of the dashboard (https://hybridjobrecommendation-brainsfirst.streamlit.app/), in which results are saved, is dormant from inactivity.

### `model_comparison.ipynb`
This notebook can be run as is for the plots and results found in the *Results* section of the thesis.

## Utils
Utils are divided into three main components:
* Configurations: contains `.py` files that contain centralized configurations and dictionaries to manage file paths and parameters. For example, indicating the name and location of files.
* Functions: Contains `.py` files that contain all the helper functions to execute the project. These are modular and reused across notebooks.
* Constants: it is a `.py` file that contains fixed values, like set coefficients, or names, to avoid repetition throughout the code.
These three elements work together behind the scenes to support the main notebooks. 

## Author
**Karen Saya Heredia** Behavioral Data Science MSc