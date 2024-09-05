import sys
from pathlib import Path
import os
import re
import subprocess
from datetime import datetime
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from config import MODEL_FOLDER, PREDICTION_FOLDER
except ImportError as e:
    logging.error("Could not import config. Ensure the config file exists and is correctly set up.")
    sys.exit(1)

def list_directories(base_path, prefix):
    base_path = Path(base_path)
    directories = [dir for dir in base_path.iterdir() if dir.is_dir() and dir.name.startswith(prefix)]
    
    def extract_datetime(dir_name):
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})', dir_name)
        if match:
            return datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M')
        else:
            return None
    
    sorted_directories = sorted(directories, key=lambda dir: extract_datetime(dir.name))
    return sorted_directories

def extract_number_from_filename(filename):
    match = re.search(r'model_(\d+)\.pth', filename)
    if match:
        return match.group(1)
    return None

def extract_features(run_dir):
    """
    Extract features from the run_dir string.
    
    Args:
    - run_dir (str or Path): The directory path string.
    
    Returns:
    - features (str): The extracted features string.
    """
    run_dir = str(run_dir)  # Ensure the input is a string
    match = re.search(r'\[(.*?)\]', run_dir)
    if match:
        features = match.group(1)
        return features
    else:
        return None
    
def execute_prediction_scripts(base_model_path, prediction_base_path):
    directories = list_directories(base_model_path, 'RAMM')

    for run_dir in directories:
        logging.info(f"Processing directory: {run_dir}")
        task = run_dir.parent.name
        logging.info(f"Task: {task}")
        features = extract_features(run_dir.name)
        logging.info(f"Extracted features: {features}")
        
        for file in run_dir.iterdir():
            if file.is_file():
                logging.info(f"Reading file: {file}")
                number = extract_number_from_filename(file.name)
                if number:
                    logging.info(f"Extracted seed: {number}")
                    csv_path = prediction_base_path / run_dir.name / str(number) / 'predictions_test.csv'
                    if not csv_path.exists():
                        command = [
                            'python', 'main.py',
                            '--task', task,
                            '--features', features,
                            '--encoder', 'RAMM',
                            '--eval_model', run_dir.name,
                            '--eval_seed', number,
                            '--predict'
                        ]
                        logging.info(f"Executing command: {' '.join(command)}")
                        try:
                            subprocess.run(command, check=True)
                        except subprocess.CalledProcessError as e:
                            logging.error(f"Command failed with error: {e}")
                else:
                    logging.warning("No number found in filename.")

def process_prediction_csvs(prediction_base_path):
    directories = list_directories(prediction_base_path, 'RAMM')

    for run_dir in directories:
        logging.info(f"Processing directory: {run_dir}")
        for seed_dir in run_dir.iterdir():
            csv_path = seed_dir / 'predictions_test.csv'
            if csv_path.exists():
                try:
                    data = pd.read_csv(csv_path)
                    if 'label' in data.columns:
                        logging.info(f"Dropping 'label' column in: {csv_path}")
                        data = data.drop(columns=['label'])
                        data.to_csv(csv_path, index=False)
                except Exception as e:
                    logging.error(f"Error processing {csv_path}: {e}")
            else:
                logging.warning(f"File does not exist: {csv_path}")

def main():
    model_base_path = Path(MODEL_FOLDER) / 'humor'
    prediction_base_path = Path(PREDICTION_FOLDER) / 'humor'

    if not model_base_path.exists():
        logging.error(f"Model base path {model_base_path} does not exist.")
        return

    if not prediction_base_path.exists():
        logging.info(f"Prediction base path {prediction_base_path} does not exist. Creating it.")
        prediction_base_path.mkdir(parents=True, exist_ok=True)

    # Execute prediction scripts
    execute_prediction_scripts(model_base_path, prediction_base_path)

    # Process prediction CSVs
    process_prediction_csvs(prediction_base_path)

if __name__ == "__main__":
    main()
