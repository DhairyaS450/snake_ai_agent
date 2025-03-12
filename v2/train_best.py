import json
import os
from agent import train

def load_best_params():
    """Load best parameters from previous grid search"""
    try:
        with open('experiments/best_params.json', 'r') as f:
            best_result = json.load(f)
        return best_result['params']
    except FileNotFoundError:
        print("No previous best parameters found. Using defaults.")
        return None

if __name__ == "__main__":
    # Load best parameters and train visually with them
    best_params = load_best_params()
    print(f"Training with parameters: {best_params}")
    train(params=best_params, headless=False)
