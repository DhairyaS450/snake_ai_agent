import multiprocessing as mp
import time
import json
import os
from itertools import product
import numpy as np
from agent import train

def run_experiment(params, experiment_id, max_games=1000):
    """Run a single experiment with given parameters"""
    print(f"Starting experiment {experiment_id} with params: {params}")
    result = train(params=params, headless=True, max_games=max_games)
    result['params'] = params
    result['experiment_id'] = experiment_id
    
    # Save experiment results
    os.makedirs('experiments', exist_ok=True)
    with open(f'experiments/exp_{experiment_id}.json', 'w') as f:
        json.dump(result, f)
    
    print(f"Experiment {experiment_id} completed. Max score: {result['max_score']}, Avg score: {result['avg_score']:.2f}")
    return result

def grid_search():
    """Run grid search over hyperparameters"""
    # Define parameter grid
    param_grid = {
        'lr': [0.0005, 0.001, 0.002],
        'gamma': [0.90, 0.95, 0.99],
        'batch_size': [64, 128, 256],
        'epsilon_decay': [200, 400, 600],
        'epsilon_min': [0.01, 0.02, 0.05]
    }
    
    # Create all combinations 
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[key] for key in keys]))
    
    # Convert combinations to list of dictionaries
    param_combinations = []
    for combo in combinations:
        param_dict = {keys[i]: combo[i] for i in range(len(keys))}
        param_combinations.append(param_dict)
    
    print(f"Running grid search with {len(param_combinations)} parameter combinations")
    
    # Define number of processes to run in parallel
    num_processes = mp.cpu_count() - 1  # Leave one CPU for system
    
    # Run experiments in parallel
    with mp.Pool(num_processes) as pool:
        results = []
        for i, params in enumerate(param_combinations):
            results.append(pool.apply_async(run_experiment, args=(params, i)))
        
        # Wait for all experiments to complete
        all_results = [r.get() for r in results]
    
    # Find best parameters
    best_result = max(all_results, key=lambda x: x['avg_score'])
    print("\n=== Best Parameters ===")
    print(f"Parameters: {best_result['params']}")
    print(f"Max score: {best_result['max_score']}")
    print(f"Average score: {best_result['avg_score']:.2f}")
    
    # Save best parameters
    with open('experiments/best_params.json', 'w') as f:
        json.dump(best_result, f)

if __name__ == "__main__":
    start_time = time.time()
    grid_search()
    duration = time.time() - start_time
    print(f"Total time: {duration/3600:.2f} hours")
