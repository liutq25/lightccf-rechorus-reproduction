"""Reproduce the official ReChorus MovieLens-1M TopK notebook deterministically.

Source: THUwangcy/ReChorus, commit c164ec4303cc20ddcfbd1b57de366a481811d1e5,
data/MovieLens_1M/MovieLens-1M.ipynb. The source notebook has path/typing
typos; this script implements its TopK data transformation without its unused
CTR/metadata branches. UTC+8 reproduces the notebook's published split counts.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "framework" / "ReChorus"
DATA = REPO / "data"
COMMIT = "c164ec4303cc20ddcfbd1b57de366a481811d1e5"
RAW_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_candidates(frame, all_items, clicked, seed, count=99):
    # Preserve the RandomState sequence used by np.random.seed in the notebook.
    rng = np.random.RandomState(seed)
    negative = rng.choice(all_items, (len(frame), count))
    for row, user in enumerate(frame["user_id"]):
        excluded = clicked[user].copy()
        for col in range(count):
            while negative[row, col] in excluded:
                negative[row, col] = rng.choice(all_items, 1)[0]
            excluded.add(int(negative[row, col]))
    return negative.tolist()


def main():
    raw_path = ROOT / "data" / "raw" / "ml-1m.zip"
    with zipfile.ZipFile(raw_path) as archive:
        raw = pd.read_csv(
            io.BytesIO(archive.read("ml-1m/ratings.dat")), sep="::",
            names=["original_user_id", "original_item_id", "rating", "time"],
            dtype={"original_user_id": str, "original_item_id": str}, engine="python",
        )
    raw["label"] = (raw["rating"] >= 4).astype(int)
    # The original notebook recomputes counts until membership is unchanged.
    # On this dataset it converges after the first filtering pass (min degree 5).
    filtered = raw.copy()
    while True:
        pos = filtered.loc[filtered.label == 1]
        users = pos.original_user_id.value_counts()
        items = pos.original_item_id.value_counts()
        updated = filtered.loc[
            filtered.original_user_id.isin(users[users >= 5].index)
            & filtered.original_item_id.isin(items[items >= 5].index)
        ].copy()
        if len(updated) == len(filtered):
            break
        filtered = updated
    assert (len(filtered), filtered.original_user_id.nunique(), filtered.original_item_id.nunique()) == (994338, 6034, 3125)

    # Explicit fixed timezone; source datetime.fromtimestamp had implicit host timezone.
    timestamp = pd.to_datetime(filtered.time, unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    filtered["c_hour_c"] = timestamp.dt.hour
    filtered["c_weekday_c"] = timestamp.dt.weekday
    hours = filtered.c_hour_c.to_numpy()
    filtered["c_period_c"] = np.select(
        [(hours >= 5) & (hours <= 8), (hours > 8) & (hours < 11),
         (hours >= 11) & (hours <= 12), (hours > 12) & (hours <= 15),
         (hours > 15) & (hours <= 17), (hours >= 18) & (hours <= 19),
         (hours > 19) & (hours <= 21), hours > 21],
        list(range(8)), default=8,
    )
    day = ((filtered.time + 8 * 3600) // 86400).astype(int)
    filtered["c_day_f"] = day - day.min()
    positives = filtered.loc[filtered.label == 1].copy()
    split1 = int(positives.c_day_f.max() * 0.8)
    split2 = int(positives.c_day_f.max() * 0.9)
    train = positives.loc[positives.c_day_f <= split1].copy()
    later = positives.loc[positives.c_day_f > split1].sort_values("time")
    dev = later.loc[later.c_day_f <= split2].copy()
    test = later.loc[later.c_day_f > split2].copy()
    train_users, train_items = set(train.original_user_id), set(train.original_item_id)
    original_sizes = {"dev": len(dev), "test": len(test)}
    dev = dev.loc[dev.original_user_id.isin(train_users) & dev.original_item_id.isin(train_items)].copy()
    test = test.loc[test.original_user_id.isin(train_users) & test.original_item_id.isin(train_items)].copy()
    assert [len(train), len(dev), len(test)] == [568761, 2562, 2874]

    full = pd.concat([train, dev, test])
    # Source ids were strings, so preserve lexicographic sorting for exact ID mapping.
    user_map = {u: index + 1 for index, u in enumerate(sorted(full.original_user_id.unique()))}
    item_map = {item: index + 1 for index, item in enumerate(sorted(full.original_item_id.unique()))}
    for frame in [train, dev, test, full]:
        frame["user_id"] = frame.original_user_id.map(user_map)
        frame["item_id"] = frame.original_item_id.map(item_map)
        frame["time"] = frame["time"].astype(float)
    clicked = {u: set(frame.item_id) for u, frame in full.groupby("user_id")}
    all_items = full.item_id.unique()
    dev["neg_items"] = sample_candidates(dev, all_items, clicked, 1)
    test["neg_items"] = sample_candidates(test, all_items, clicked, 2)
    out = DATA / "ML_1MTOPK"
    out.mkdir(exist_ok=True)
    cols = ["user_id", "item_id", "time", "c_hour_c", "c_weekday_c", "c_period_c", "c_day_f"]
    for name, frame in [("train", train), ("dev", dev), ("test", test)]:
        frame[cols + ([] if name == "train" else ["neg_items"])].to_csv(out / f"{name}.csv", sep="\t", index=False, lineterminator="\r\n")
    (out / "user2newid.json").write_text(json.dumps(user_map), encoding="utf-8")
    (out / "item2newid.json").write_text(json.dumps(item_map), encoding="utf-8")
    manifest = {
        "upstream": "https://github.com/THUwangcy/ReChorus", "commit": COMMIT,
        "raw": {"url": RAW_URL, "path": str(raw_path.relative_to(ROOT)), "sha256": sha256(raw_path)},
        "preprocessing": {
            "notebook": "data/MovieLens_1M/MovieLens-1M.ipynb", "timezone": "Asia/Shanghai (UTC+8)",
            "positive_rating_threshold": 4, "positive_min_degree": 5,
            "split": "global calendar-day 80%/10%/10%; remove dev/test users or items absent in train",
            "split_day_thresholds": [split1, split2], "negative_count": 99, "negative_seeds": {"dev": 1, "test": 2},
            "negative_exclusion": "all retained positive interactions; sampled candidates supplied for upstream compatibility only",
            "cold_rows_removed": {"dev": original_sizes["dev"] - len(dev), "test": original_sizes["test"] - len(test)},
            "deviations": ["fixed timezone instead of implicit host timezone", "correct notebook path and undefined-variable typos", "skip unused CTR and item metadata branches", "enforce 5-core convergence explicitly; identical resulting membership on this input"],
        },
        "serialization": {"encoding": "UTF-8", "line_ending": "CRLF", "scope": "pin the originally measured Windows dataset bytes; no interaction changes"},
        "datasets": {},
    }
    for name in ["Grocery_and_Gourmet_Food", "ML_1MTOPK"]:
        entries = {}
        frames = []
        for split in ["train", "dev", "test"]:
            path = DATA / name / f"{split}.csv"
            frame = pd.read_csv(path, sep="\t")
            frames.append(frame)
            entries[split] = {"rows": len(frame), "users": frame.user_id.nunique(), "items": frame.item_id.nunique(), "sha256": sha256(path), "path": str(path.relative_to(ROOT))}
        full_data = pd.concat(frames)
        entries["total"] = {"rows": len(full_data), "users": full_data.user_id.nunique(), "items": full_data.item_id.nunique(), "density": len(full_data) / (full_data.user_id.nunique() * full_data.item_id.nunique())}
        manifest["datasets"][name] = entries
    manifest_path = ROOT / "research" / "rechorus_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
