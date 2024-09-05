import wandb
from datetime import datetime
from dateutil import tz
import pandas as pd
from itertools import product
import os

def run_meets_criteria(run, filters):
    """Check if a run meets the specified criteria."""
    for key, value in filters.items():
        if run.config.get(key) != value:
            return False
    return True

def initialize_wandb_api():
    """Initialize the WandB API."""
    return wandb.Api()

def create_result_directory(project_root, relative_path='../results'):
    """Create the results directory if it doesn't exist."""
    result_dir = os.path.join(project_root, relative_path)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def get_filter_combinations(fixed_filter, combination_range):
    """Generate all combinations of filters."""
    keys = combination_range.keys()
    values = combination_range.values()
    filter_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    for comb in filter_combinations:
        comb.update(fixed_filter)
    return filter_combinations

def fetch_best_run(api, entity_name, project_name, filters, metric_name, metric_goal):
    """Fetch the best run based on the specified metric and filters."""
    order = '+' if metric_goal == 'minimize' else '-'
    order += f"summary_metrics.{metric_name}"
    
    runs = api.runs(path=f'{entity_name}/{project_name}', filters=filters, order=order)
    return runs[0] if len(runs) != 0 else None

def collect_run_data(best_run):
    """Collect necessary data from the best run."""
    return {
        'run_name': best_run.name,
        'dev_loss': best_run.summary.get('best_val_loss'),
        'train_auc': best_run.summary.get('train_score'),
        'dev_auc': best_run.summary.get('best_val_score'),
        'seed': best_run.config.get('seed'),
        'harmonic_mean': 2 * (best_run.summary.get('train_score') * best_run.summary.get('best_val_score')) / 
                         (best_run.summary.get('train_score') + best_run.summary.get('best_val_score'))
    }

def save_results_to_csv(df, result_dir):
    """Save the DataFrame to a CSV file."""
    timestamp = datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M")
    filename = f'results_{timestamp}_best_runs.csv'
    df.to_csv(os.path.join(result_dir, filename), index=False)

# Main execution
def main():
    # Initialize WandB API
    api = initialize_wandb_api()

    # Define directories
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = create_result_directory(PROJECT_ROOT)

    # Define project details
    entity_name = 'ssu'
    project_name = "SSU_MuSe2024_HUMOR"
    metric_name = 'best_val_score'
    metric_goal = 'maximize'

    # Define filters and combinations
    fixed_filter = {'state': 'finished'}
    combination_range = {
        'config.encoder': ['RAMM'],
        'config.features': ['w2v-msp + bert-multilingual', 'vit-fer + w2v-msp', 'vit-fer + bert-multilingual']
    }

    filter_combinations = get_filter_combinations(fixed_filter, combination_range)

    # Create DataFrame to store results
    df = pd.DataFrame(columns=['run_name', 'seed', 'dev_loss', 'train_auc', 'dev_auc', 'harmonic_mean'])

    for filters in filter_combinations:
        print(f'Combination: {filters}')
        row = {key: 0 for key in df.columns}
        row.update(filters)

        best_run = fetch_best_run(api, entity_name, project_name, filters, metric_name, metric_goal)
        if best_run:
            row.update(collect_run_data(best_run))
            print(f"Best run Name: {best_run.name}")
            print(f"Best metric value: {row['dev_auc']}")
            print(f"Run details: {best_run.url}")
        else:
            print("No runs found with the specified metric.")
        print('='*50)
        df = df.append(row, ignore_index=True)

    # Save results to CSV
    save_results_to_csv(df, RESULT_DIR)

if __name__ == "__main__":
    main()
