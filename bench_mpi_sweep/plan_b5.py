#!/usr/bin/env python3
"""Append config B5_2x32 to plan.tsv: 64 readers over 2 nodes, 7.5 GiB fresh
windows from the r400 pool (r387's pool is nearly exhausted; every window in
plan.tsv is read at most once, so new configs need untouched bytes).

Replays plan_gen.py's allocations deterministically (same greedy pool, same
config order, same file sizes) so B5 windows never overlap anything already
assigned, then appends only the new rows.
"""
import os
from plan_gen import Pool, R387_BIG, R400_BIG, GiB

W15 = 15 * GiB
W75 = 15 * GiB // 2

def main():
    r387 = Pool(R387_BIG)
    r400 = Pool(R400_BIG)

    # Replay the original allocation sequence exactly (see plan_gen.py).
    replay = [
        ("B1_1",     1, 1, W15, "seq",  r387),
        ("B1_8",     8, 1, W15, "seq",  r387),
        ("B1_32",   32, 1, W75, "seq",  r387),
        ("B2_2n",   32, 2, W75, "seq",  r387),
        ("B2_4n",   32, 4, W75, "seq",  r387),
        ("B3_rand", 32, 1, W75, "rand", r387),
        ("B4_r387", 32, 1, W75, "seq",  r387),
        ("B4_r400", 32, 1, W75, "seq",  r400),
    ]
    for name, n, nodes, win, pat, pool in replay:
        for i in range(n):
            pool.take(win)

    # New config: 64 readers, 32 per node on 2 nodes, fresh r400 bytes.
    rows = []
    name, n, nodes, win, pat = "B5_2x32", 64, 2, W75, "seq"
    per_node = n // nodes
    for i in range(n):
        path, off = r400.take(win)
        rows.append((name, i, i // per_node, pat, path, off, win))

    plan = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.tsv")
    with open(plan, "a") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"appended {len(rows)} B5_2x32 rows ({len(rows)*win/1e9:.0f} GB fresh)")

if __name__ == "__main__":
    main()
