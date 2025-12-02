import pandas as pd
import numpy as np
import msoffcrypto
from io import BytesIO
import os
import re
import logging
from typing import List, Dict, Optional, Tuple
import ast

def load_decrypted_excel(filepath: str, password: str) -> Optional[pd.DataFrame]:
    """Loads and decrypts a password-protected Excel file."""
    try:
        with open(filepath, 'rb') as f:
            encrypted = msoffcrypto.OfficeFile(f)
            encrypted.load_key(password=password)
            decrypted = BytesIO()
            encrypted.decrypt(decrypted)
            df = pd.read_excel(decrypted)
            logging.info(f"Successfully loaded and decrypted: {filepath}")
            return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Failed to load/decrypt {filepath}: {e}")
        return None
    
def load_file(filepath: str, encrypted: bool = False, password: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Loads an Excel or CSV file, handling encryption."""
    if encrypted:
        if not password:
            logging.error(f"Password required for encrypted file: {filepath}")
            return None
        return load_decrypted_excel(filepath, password)
    else:
        try:
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
                logging.info(f"Successfully loaded: {filepath}")
                return df
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                logging.info(f"Successfully loaded: {filepath}")
                return df
            else:
                logging.warning(f"Unsupported file type: {filepath}. Only .xlsx and .csv are supported.")
                return None
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            return None
        except Exception as e:
            logging.error(f"Failed to load {filepath}: {e}")
            return None

def parse_to_list(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    if pd.isna(val):
         return []
    try:
        evaluated = ast.literal_eval(str(val))
        return evaluated if isinstance(evaluated, list) else [evaluated]
    except (ValueError, SyntaxError):
        return [str(val)]    
    
def parse_llm_response(response_text: str):
    """Pases raw LLM text into a title and description dictionary."""
    if not response_text:
        raise ValueError("Received empty response text.")
        
    meta_task_match = re.search(r"Meta-task: (.*)", response_text, re.IGNORECASE | re.DOTALL)
    description_match = re.search(r"Description: (.*)", response_text, re.IGNORECASE | re.DOTALL)

    if not meta_task_match or not description_match:
        raise ValueError(f"Output format incorrect. Response: '{response_text[:100]}...'")
        
    title = meta_task_match.group(1).strip()
    description = description_match.group(1).strip()

    return {'title': title, 'description': description}

def get_meta_tasks_for_job(row, task_map):
    all_task_strings = set()
    task_columns = ['onet_tasks', 'Key Tasks', 'Critical Work Function', 'description']

    for col in task_columns:
        if col not in row.index:
            continue
        val = row[col]
        # skip missing
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue

        if col == 'description' and isinstance(val, str):
            all_task_strings.add(val)
        elif isinstance(val, (list, tuple, set, np.ndarray)):
            all_task_strings.update(val)
        elif isinstance(val, str):
            try:
                evaluated = ast.literal_eval(val)
                if isinstance(evaluated, (list, tuple, set, np.ndarray)):
                    all_task_strings.update(evaluated)
                else:
                    all_task_strings.add(val)
            except (ValueError, SyntaxError):
                all_task_strings.add(val)
    cluster_ids = {task_map.get(task, -1) for task in all_task_strings}
    return sorted([cid for cid in cluster_ids if cid != -1])


def smart_title(text):
    return " ".join([
        word if word.isupper() else word.capitalize()
        for word in str(text).split()
    ])