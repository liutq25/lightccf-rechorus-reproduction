"""Read-only consistency audit for the two supplied ReChorus datasets."""
import ast
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = {}
for name in ["Grocery_and_Gourmet_Food", "ML_1MTOPK"]:
    frames = {split: pd.read_csv(ROOT / "framework" / "ReChorus" / "data" / name / f"{split}.csv", sep="\t") for split in ["train", "dev", "test"]}
    pairs = {split: set(zip(df.user_id, df.item_id)) for split, df in frames.items()}
    train_items = set(frames["train"].item_id)
    all_df = pd.concat(frames.values())
    clicked = {u: set(df.item_id) for u, df in all_df.groupby("user_id")}
    result = {
        "duplicate_user_item_pairs": {split: len(df) - len(pairs[split]) for split, df in frames.items()},
        "pair_overlap": {"train_dev": len(pairs["train"] & pairs["dev"]), "train_test": len(pairs["train"] & pairs["test"]), "dev_test": len(pairs["dev"] & pairs["test"])},
        "items_absent_from_train": len(set(all_df.item_id) - train_items),
        "cold_item_targets": {split: int((~frames[split].item_id.isin(train_items)).sum()) for split in ["dev", "test"]},
        "positive_train_degree_min": {"user": int(frames["train"].groupby("user_id").size().min()), "item": int(frames["train"].groupby("item_id").size().min())},
        "candidate_checks": {},
    }
    for split in ["dev", "test"]:
        duplicate_rows = false_negative_rows = invalid_rows = 0
        for row in frames[split].itertuples():
            neg = ast.literal_eval(row.neg_items)
            duplicate_rows += len(neg) != len(set(neg))
            false_negative_rows += bool(set(neg) & clicked[row.user_id])
            invalid_rows += any(i <= 0 or i > all_df.item_id.max() for i in neg)
        result["candidate_checks"][split] = {"rows_with_duplicate_negatives": duplicate_rows, "rows_with_known_positive_negative": false_negative_rows, "rows_with_out_of_range_negative": invalid_rows}
    if name == "Grocery_and_Gourmet_Food":
        train_last = frames["train"].groupby("user_id").time.max()
        dev = frames["dev"].set_index("user_id").time
        test = frames["test"].set_index("user_id").time
        result["temporal_checks"] = {"train_after_dev": int((train_last > dev).sum()), "dev_after_test": int((dev > test).sum()), "train_dev_ties": int((train_last == dev).sum()), "dev_test_ties": int((dev == test).sum())}
    else:
        result["temporal_checks"] = {"train_max": int(frames["train"].time.max()), "dev_min": int(frames["dev"].time.min()), "dev_max": int(frames["dev"].time.max()), "test_min": int(frames["test"].time.min())}
    out[name] = result
path = ROOT / "research" / "rechorus_data_checks.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
