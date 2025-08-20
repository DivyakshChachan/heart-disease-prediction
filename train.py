
from __future__ import annotations
import os
import argparse
from src.model_training import train_models

def parse_args():
    p = argparse.ArgumentParser(description="Train heart disease models")
    p.add_argument('--data', default=os.path.join(os.path.dirname(__file__), 'data', 'heart_disease_dataset.csv'))
    p.add_argument('--models_dir', default=os.path.join(os.path.dirname(__file__), 'models'))
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    results = train_models(args.data, args.models_dir)
    print(results)
