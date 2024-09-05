import wandb
import itertools
from datetime import datetime
from dateutil import tz
import pandas as pd
import os

def initialize_wandb_api():
    """Initialize the WandB API."""
    return wandb.Api()

def create_result_directory(project_root, relative_path='../results'):
    """Create the results directory if it doesn't exist."""
    result_dir = os.path.join(project_root, relative_path)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def get_runs(api, entity_name, project_name):
    """Retrieve all runs for the given project."""
    return api.runs(f'{entity_name}/{project_name}')

def collect_run_data(run, comb):
    """Collect necessary data from a run."""
    return {
        'run_name': run.name,
        'encoder': comb[0],
        'features': comb[1],
        'dev_auc': run.summary.get('best_val_score'),
        'dev_loss': run.summary.get('best_val_loss'),
        'train_auc': run.summary.get('train_score'),
        'seed': run.summary.get('seed'),
        'harmonic_mean': 2 * (run.summary.get('train_score') * run.summary.get('best_val_score')) / 
                         (run.summary.get('train_score') + run.summary.get('best_val_score'))
    }

def save_results_to_csv(df, result_dir):
    """Save the DataFrame to a CSV file."""
    timestamp = datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M")
    filename = f'results_{timestamp}_all.csv'
    df.to_csv(os.path.join(result_dir, filename), index=False)

def main():
    # Initialize WandB API
    api = initialize_wandb_api()

    # Define directories
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = create_result_directory(PROJECT_ROOT)

    # Define project details
    entity_name = 'ssu'
    project_name = "SSU_MuSe2024_HUMOR"

    # Define metric and goal
    metric_name = 'best_val_score'
    metric_goal = 'maximize'

    # Define parameter combinations
    encoders = ['RAMM']
    features = ['vit-fer + w2v-msp + bert-multilingual']
    filter_combinations = itertools.product(encoders, features)

    # Create DataFrame to store results
    df = pd.DataFrame(columns=['run_name', 'encoder', 'features', 'seed', 'dev_loss', 'train_auc', 'dev_auc', 'harmonic_mean'])

    # Retrieve all runs for the project
    runs = get_runs(api, entity_name, project_name)

    # Iterate through filter combinations
    for comb in filter_combinations:
        print(f'Combination: {comb}')
        for run in runs:
            print(f'\tRun[ id: {run.id} | name: {run.name} | status: {run.state}]')
            if metric_name in run.summary:
                row = collect_run_data(run, comb)
                df = df.append(row, ignore_index=True)

    # Sort and save results
    df_sorted = df.sort_values(by='harmonic_mean', ascending=False)
    save_results_to_csv(df_sorted, RESULT_DIR)

if __name__ == "__main__":
    main()
