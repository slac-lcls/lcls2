#!/usr/bin/env python3
"""Generate plan.tsv: globally disjoint byte windows for every io_matrix reader.

Every (config, reader) gets its own file+window; windows never overlap across
configs either, so page cache can never serve a re-read anywhere in the job.
Columns: config, reader_id, node_index, pattern, path, offset, length.
"""
import os

XTC = "/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc"
R387_BIG = [f"{XTC}/mfx101210926-r0387-s{s:03d}-c000.xtc2" for s in range(3, 10)]
R400_BIG = ([f"{XTC}/mfx101210926-r0400-s000-c000.xtc2"] +
            [f"{XTC}/mfx101210926-r0400-s{s:03d}-c000.xtc2" for s in (3, 5, 6, 7, 8, 9)])

GiB = 1024**3
TAIL_MARGIN = 1 * GiB          # never touch the last GiB of a file

class Pool:
    def __init__(self, paths):
        self.rem = {}   # path -> (next_offset, usable_end)
        for p in paths:
            size = os.stat(p).st_size
            self.rem[p] = [0, size - TAIL_MARGIN]

    def take(self, length):
        # greedy: file with most remaining bytes
        best = max(self.rem, key=lambda p: self.rem[p][1] - self.rem[p][0])
        off, end = self.rem[best]
        assert end - off >= length, f"pool exhausted: need {length}, have {end-off}"
        self.rem[best][0] = off + length
        return best, off

def main():
    r387 = Pool(R387_BIG)
    r400 = Pool(R400_BIG)
    W15 = 15 * GiB
    W75 = 15 * GiB // 2         # 7.5 GiB

    # config, nreaders, nodes, window, pattern, pool
    configs = [
        ("B1_1",     1, 1, W15, "seq",  r387),
        ("B1_8",     8, 1, W15, "seq",  r387),
        ("B1_32",   32, 1, W75, "seq",  r387),
        ("B2_2n",   32, 2, W75, "seq",  r387),
        ("B2_4n",   32, 4, W75, "seq",  r387),
        ("B3_rand", 32, 1, W75, "rand", r387),
        ("B4_r387", 32, 1, W75, "seq",  r387),
        ("B4_r400", 32, 1, W75, "seq",  r400),
    ]

    rows = []
    for name, n, nodes, win, pat, pool in configs:
        per_node = n // nodes
        for i in range(n):
            path, off = pool.take(win)
            rows.append((name, i, i // per_node, pat, path, off, win))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.tsv"), "w") as f:
        f.write("config\treader\tnode\tpattern\tpath\toffset\tlength\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    tot = sum(r[6] for r in rows)
    print(f"plan.tsv: {len(rows)} windows, {tot/1e12:.2f} TB total")
    for p, (off, end) in sorted(r387.rem.items()):
        print(f"  r387 {os.path.basename(p)}: used {off/GiB:.1f} GiB, headroom {(end-off)/GiB:.1f} GiB")

if __name__ == "__main__":
    main()
