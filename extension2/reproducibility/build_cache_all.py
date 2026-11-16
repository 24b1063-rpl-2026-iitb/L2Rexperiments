"""
build_cache_all.py
Generates fast binary (.pt) tensor caches for all 5 folds.
- Memory-bounded (< 800 MB RAM peak).
- Shows live tqdm progress for each split.
- Skips splits that are already cached.
"""
import os, time, torch
import numpy as np
from tqdm import tqdm

BASE_DIR = r"C:\Users\rajde\Downloads\L2R Project\data\MSLR-WEB30K"
CHUNK_LINES = 25000


def cache_split(file_path):
    cache_path = os.path.splitext(file_path)[0] + ".pt"
    base_name = os.path.basename(file_path)

    if os.path.exists(cache_path):
        print(f"  ✓ [Already Cached] {base_name} -> {os.path.basename(cache_path)}", flush=True)
        return

    t0 = time.time()
    file_size = os.path.getsize(file_path)

    X_list, y_list = [], []
    curr_qid = None
    curr_X, curr_y = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=f"  Caching {base_name:<9}", ncols=90, colour="green") as pbar:
            last_pos = 0
            while True:
                lines = [f.readline() for _ in range(CHUNK_LINES)]
                lines = [l for l in lines if l]
                if not lines:
                    break
                curr_pos = f.tell()
                pbar.update(curr_pos - last_pos)
                last_pos = curr_pos

                for line in lines:
                    parts = line.split()
                    label = float(parts[0])
                    qid = int(parts[1][4:])
                    feats = [float(p.split(":")[1]) for p in parts[2:]]

                    if curr_qid is None or qid == curr_qid:
                        curr_X.append(feats)
                        curr_y.append(label)
                        curr_qid = qid
                    else:
                        X_list.append(np.array(curr_X, dtype=np.float32))
                        y_list.append(np.array(curr_y, dtype=np.float32))
                        curr_X = [feats]
                        curr_y = [label]
                        curr_qid = qid

    if curr_X:
        X_list.append(np.array(curr_X, dtype=np.float32))
        y_list.append(np.array(curr_y, dtype=np.float32))

    torch.save({"X_by_qid": X_list, "y_by_qid": y_list}, cache_path)
    elapsed = int(time.time() - t0)
    print(f"  ✓ Saved {len(X_list):,} queries -> {os.path.basename(cache_path)} ({elapsed//60}m {elapsed%60}s)\n", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MSLR-WEB30K Binary Tensor Cache Builder")
    parser.add_argument("--data_dir", default=None, help="Parent directory containing FoldN_normalized folders")
    parser.add_argument("--fold_dir", default=None, help="Specific Fold directory to cache (e.g. /path/to/Fold1_normalized)")
    args = parser.parse_args()

    print("=" * 65)
    print("MSLR-WEB30K Binary Tensor Cache Builder")
    print("=" * 65)
    total_start = time.time()

    if args.fold_dir:
        fold_dirs = [args.fold_dir]
    elif args.data_dir:
        fold_dirs = [os.path.join(args.data_dir, f"Fold{i}_normalized") for i in range(1, 6)]
    else:
        fold_dirs = [os.path.join(BASE_DIR, f"Fold{i}_normalized") for i in range(1, 6)]

    for fold_dir in fold_dirs:
        if not os.path.exists(fold_dir):
            continue

        print(f"\n>>> Processing {os.path.basename(fold_dir)}...")
        for split in ["train.txt", "vali.txt", "test.txt"]:
            file_path = os.path.join(fold_dir, split)
            if os.path.exists(file_path):
                cache_split(file_path)

    total_elapsed = int(time.time() - total_start)
    print("=" * 65)
    print(f"✓ Caching complete in {total_elapsed//60}m {total_elapsed%60}s!")
    print("=" * 65)


if __name__ == "__main__":
    main()
