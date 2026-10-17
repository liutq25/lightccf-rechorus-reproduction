"""Measure how often official positive interactions share a user in a batch."""
from pathlib import Path
import json

import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
BATCH = 2048


def analyze(dataset):
    path = ROOT / 'framework' / 'ReChorus' / 'data' / dataset / 'train.csv'
    frame = pd.read_csv(path, sep='\t', usecols=['user_id', 'item_id'])
    n = len(frame)
    users = frame['user_id'].value_counts().to_numpy()
    items = frame['item_id'].value_counts().to_numpy()
    def expectation(counts):
        return (BATCH - 1) * (counts * (counts - 1)).sum() / (n * (n - 1))
    # Average over the anchor interaction, not over distinct users.
    p_other_same_user = float((users / n * hypergeom.sf(0, n - 1, users - 1, BATCH - 1)).sum())
    return {'dataset': dataset, 'train_interactions': int(n), 'batch_size': BATCH,
            'mean_train_positives_per_user': float(n / len(users)),
            'expected_other_same_user_positives_per_anchor': float(expectation(users)),
            'probability_anchor_has_another_same_user_positive': p_other_same_user,
            'expected_other_same_item_interactions_per_anchor': float(expectation(items))}


def main():
    rows = [analyze(dataset) for dataset in ('Grocery_and_Gourmet_Food', 'ML_1MTOPK')]
    output = ROOT / 'analysis' / 'batch_collisions.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
