"""Describe the fixed ReChorus splits without inspecting model predictions."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'framework' / 'ReChorus' / 'data'
OUT = ROOT / 'analysis' / 'dataset_profile.json'


def describe_counts(values):
    return {
        'mean': float(values.mean()),
        'median': float(values.median()),
        'p90': float(values.quantile(.9)),
        'maximum': int(values.max()),
    }


def main():
    profile = {}
    for name in ('Grocery_and_Gourmet_Food', 'ML_1MTOPK'):
        split = {phase: pd.read_csv(DATA / name / f'{phase}.csv', sep='\t',
                                    usecols=['user_id', 'item_id'])
                 for phase in ('train', 'dev', 'test')}
        train_degree = split['train'].groupby('user_id').size()
        test_degree = split['test'].groupby('user_id').size()
        dev_users = set(split['dev'].user_id)
        test_users = set(split['test'].user_id)
        all_items = set().union(*(set(frame.item_id) for frame in split.values()))
        test_histories = train_degree.reindex(sorted(test_users), fill_value=0)
        profile[name] = {
            'train_interactions_per_user': describe_counts(train_degree),
            'test_interactions_per_test_user': describe_counts(test_degree),
            'train_interactions_for_test_users': describe_counts(test_histories),
            'test_users': len(test_users),
            'dev_test_user_overlap': len(dev_users & test_users),
            'items_absent_from_train': len(all_items - set(split['train'].item_id)),
        }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
