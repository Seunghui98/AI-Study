#!/usr/bin/env python3
"""Paired comparison of two NNTR_PPL_DECODE forced runs on the same ids file.

Usage: python3 ppl_compare.py A.log B.log [more B logs...]

Reads the "[PPL] decode step=N target=T nll=X" lines of each log. The runs
must score the same targets (same ids file); a mismatch stops the script.
For each B against A, step by step:

  sum ratio      nll_sum(B) / nll_sum(A), what the gate used so far
  95% CI         paired bootstrap of that ratio (steps resampled together,
                 2000 draws, fixed seed)
  trimmed ratio  the same without the 5 % of steps whose nll differs most
                 between the runs (a few tokens can carry a 128-token sum)
  B better       steps where B's nll is lower by more than 1e-6, against
                 steps where it is higher; sign-test p (two-sided)
  median diff    median of nll(B) - nll(A)
  top steps      the steps that move the sum most

Verdict line: "B not worse" when the CI's upper end is <= 1.02 or the
sign test does not reject (p >= 0.05) with B winning at least as many
steps; "B worse" when the CI's lower end is > 1.0 and the sign test
rejects with B losing more; otherwise "undecided".
"""
import math
import os
import random
import re
import sys

STEP = re.compile(r"\[PPL\] decode step=(\d+) target=(\d+) nll=([-+0-9.eEinfa]+)")


def read(path):
    steps = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP.search(line)
            if m:
                steps[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
    if not steps:
        raise SystemExit(f"{path}: no [PPL] decode step lines")
    return steps


def sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def compare(a, b, name_a, name_b):
    common = sorted(set(a) & set(b))
    if len(common) != len(a) or len(common) != len(b):
        print(f"  note: {name_a} has {len(a)} steps, {name_b} {len(b)}; "
              f"{len(common)} compared")
    for s in common:
        if a[s][0] != b[s][0]:
            raise SystemExit(f"step {s}: target {a[s][0]} vs {b[s][0]} "
                             "(not the same ids file)")
    xa = [a[s][1] for s in common]
    xb = [b[s][1] for s in common]
    n = len(common)
    sa, sb = sum(xa), sum(xb)
    ratio = sb / sa if sa > 0 else float("nan")

    rng = random.Random(4428)
    draws = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(xa[i] for i in idx)
        db = sum(xb[i] for i in idx)
        if da > 0:
            draws.append(db / da)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]

    cut = max(1, int(round(0.05 * n)))
    hard = sorted(range(n), key=lambda i: abs(xb[i] - xa[i]), reverse=True)
    keep = set(range(n)) - set(hard[:cut])
    ta = sum(xa[i] for i in keep)
    tb = sum(xb[i] for i in keep)
    trimmed = tb / ta if ta > 0 else float("nan")

    d = [xb[i] - xa[i] for i in range(n)]
    wins = sum(1 for v in d if v < -1e-6)
    losses = sum(1 for v in d if v > 1e-6)
    p = sign_p(wins, losses)
    med = sorted(d)[n // 2] if n % 2 else 0.5 * (sorted(d)[n // 2 - 1] +
                                                   sorted(d)[n // 2])

    print(f"{name_b} vs {name_a}: steps={n}")
    print(f"  sum ratio      {ratio:.4f}  (A {sa:.4f}, B {sb:.4f})")
    print(f"  95% CI         [{lo:.4f}, {hi:.4f}]")
    print(f"  trimmed ratio  {trimmed:.4f}  (without the {cut} steps that differ most)")
    print(f"  B better       {wins} / worse {losses} / tie {n - wins - losses}"
          f"  sign-test p={p:.3g}")
    print(f"  median diff    {med:+.5f}")
    top = sorted(range(n), key=lambda i: abs(d[i]), reverse=True)[:5]
    print("  top steps      " + ", ".join(
        f"step {common[i]} {xa[i]:.3f}->{xb[i]:.3f}" for i in top))
    if hi <= 1.02 or (p >= 0.05 and wins >= losses):
        verdict = "B not worse"
    elif lo > 1.0 and p < 0.05 and losses > wins:
        verdict = "B worse"
    else:
        verdict = "undecided"
    print(f"  verdict        {verdict}")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    a = read(argv[1])
    for path in argv[2:]:
        compare(a, read(path), os.path.basename(argv[1]),
                os.path.basename(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
