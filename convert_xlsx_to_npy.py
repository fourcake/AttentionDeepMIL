"""Convert iMiGUE skeleton xlsx files to numpy .npy for fast loading.

Run once: python convert_xlsx_to_npy.py
Input:  /data-store/xingke/iMiGUE/mg_skeleton_only/{vid:04d}/{vid:04d}_2.xlsx
Output: /data-store/xingke/iMiGUE/mg_skeleton_npy/{vid:04d}.npy

Each xlsx is ~21MB (openpyxl slow), each .npy is ~16MB (numpy fast).
"""

import os
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool


SRC_DIR = '/data-store/xingke/iMiGUE/mg_skeleton_only'
DST_DIR = '/data-store/xingke/iMiGUE/mg_skeleton_npy'


def convert_one(vid):
    src = os.path.join(SRC_DIR, f'{vid:04d}', f'{vid:04d}_2.xlsx')
    dst = os.path.join(DST_DIR, f'{vid:04d}.npy')
    if os.path.exists(dst):
        return vid, 'skipped'
    try:
        arr = pd.read_excel(src, header=None).values.astype(np.float32)
        np.save(dst, arr)
        return vid, 'ok'
    except Exception as e:
        return vid, f'error: {e}'


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # Get all video IDs
    vids = sorted([int(d) for d in os.listdir(SRC_DIR)
                   if os.path.isdir(os.path.join(SRC_DIR, d))])
    print(f'Found {len(vids)} skeleton directories')

    t0 = time.time()

    # Use multiprocessing for parallel conversion
    # Each worker reads one xlsx at a time
    with Pool(processes=4) as pool:
        results = []
        for i, (vid, status) in enumerate(pool.imap_unordered(convert_one, vids)):
            results.append((vid, status))
            if (i + 1) % 20 == 0 or (i + 1) == len(vids):
                elapsed = time.time() - t0
                print(f'  [{i+1}/{len(vids)}] {elapsed:.0f}s elapsed')

    ok = sum(1 for _, s in results if s == 'ok')
    skipped = sum(1 for _, s in results if s == 'skipped')
    errors = [(v, s) for v, s in results if s.startswith('error')]

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s: {ok} converted, {skipped} skipped, {len(errors)} errors')
    if errors:
        for v, s in errors:
            print(f'  Error: video {v}: {s}')


if __name__ == '__main__':
    main()
