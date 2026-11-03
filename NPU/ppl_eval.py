#!/usr/bin/env python3
"""Accuracy gate over many prompts: forced NNTR_PPL_DECODE runs of two
cells against a CPU-attention reference (round 13 request).

Usage: python3 ppl_eval.py LOGDIR [A_CELL B_CELL]   (default: off default)

LOGDIR holds, per prompt name P:
  P__ref.log      the reference's self run (fp16 pack, NNTR_HTP_ATTN_F16=0)
                  that wrote P.cpu.ids
  P__<cell>.log   a forced run of that cell on P.cpu.ids
Each reads the "[PPL] decode step=N target=T nll=X top1=I margin=M gap=G"
lines (margin / gap: b0b3c80 on; older logs give no margin columns).

Prints, per prompt and pooled over all prompts:
  ratio     nll_sum(B) / nll_sum(A)
  top1      steps whose pick is the reference's token, for A and B
  pooled 95% CI of the ratio by a two-level bootstrap: prompts resampled,
            then steps within each (2000 draws, fixed seed), so one long
            or odd prompt does not decide it
  sign test over all steps (B lower nll vs higher, ties < 1e-6 dropped)
Then, at the steps A or B picks other than the reference: the reference's
own margin there (a small one is a near tie, any change may flip it) and
the cell's gap (how far its pick is above the reference token).

Verdict: "B not worse" when the pooled CI's upper end is <= 1.03 and B's
top1 agreement is no more than 1 point below A's; "B worse" when the
CI's lower end is > 1.0; else "undecided".
Self-test: python3 ppl_eval.py --self-test
"""
import math
import os
import random
import re
import sys

STEP = re.compile(r"\[PPL\] decode step=(\d+) target=(\d+) nll=([-+0-9.eEinfa]+)"
                  r" top1=(\d+)(?: margin=([-+0-9.eEinfa]+) gap=([-+0-9.eEinfa]+))?")
TIE = 0.5  # logits: a reference margin below this is a near tie


def read(path):
    steps = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP.search(line)
            if m:
                margin = float(m.group(5)) if m.group(5) else None
                gap = float(m.group(6)) if m.group(6) else None
                steps[int(m.group(1))] = (int(m.group(2)), float(m.group(3)),
                                          int(m.group(4)), margin, gap)
    if not steps:
        raise SystemExit(f"{path}: no [PPL] decode step lines")
    return steps


def sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # integers to the last step: 2.0**n overflows a float past n = 1023
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def load(logdir, a, b):
    prompts = {}
    for name in sorted(os.listdir(logdir)):
        if not name.endswith("__ref.log"):
            continue
        p = name[: -len("__ref.log")]
        fa = os.path.join(logdir, f"{p}__{a}.log")
        fb = os.path.join(logdir, f"{p}__{b}.log")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"  skip {p}: missing {a} or {b} log")
            continue
        ref, ra, rb = read(os.path.join(logdir, name)), read(fa), read(fb)
        common = sorted(set(ref) & set(ra) & set(rb))
        for s in common:
            if not (ref[s][0] == ra[s][0] == rb[s][0]):
                raise SystemExit(f"{p}: step {s} targets differ -- not the "
                                 "same ids file")
        prompts[p] = [(ref[s], ra[s], rb[s]) for s in common]
    if not prompts:
        raise SystemExit(f"{logdir}: no prompt with ref, {a} and {b} logs")
    return prompts


def ratio(rows):
    sa = sum(r[1][1] for r in rows)
    sb = sum(r[2][1] for r in rows)
    return sb / sa if sa > 0 else float("nan")


def boot(prompts, draws=2000, seed=1):
    rng = random.Random(seed)
    names = list(prompts)
    out = []
    for _ in range(draws):
        sa = sb = 0.0
        for _ in names:
            rows = prompts[rng.choice(names)]
            for _ in rows:
                r = rows[rng.randrange(len(rows))]
                sa += r[1][1]
                sb += r[2][1]
        out.append(sb / sa if sa > 0 else float("nan"))
    out.sort()
    return out[int(0.025 * draws)], out[int(0.975 * draws) - 1]


def report(prompts, a, b):
    print(f"{'prompt':<14} {'steps':>5} {'nll ' + a:>10} {'nll ' + b:>10} "
          f"{'ratio':>7} {'top1 ' + a:>8} {'top1 ' + b:>8}")
    allrows = []
    for p, rows in prompts.items():
        allrows += rows
        ta = sum(r[1][2] == r[0][0] for r in rows)
        tb = sum(r[2][2] == r[0][0] for r in rows)
        print(f"{p:<14} {len(rows):>5} {sum(r[1][1] for r in rows):>10.3f} "
              f"{sum(r[2][1] for r in rows):>10.3f} {ratio(rows):>7.4f} "
              f"{ta:>8} {tb:>8}")
    n = len(allrows)
    ta = sum(r[1][2] == r[0][0] for r in allrows)
    tb = sum(r[2][2] == r[0][0] for r in allrows)
    lo, hi = boot(prompts)
    wins = sum(r[2][1] < r[1][1] - 1e-6 for r in allrows)
    losses = sum(r[2][1] > r[1][1] + 1e-6 for r in allrows)
    print(f"\npooled: {len(prompts)} prompts, {n} steps")
    print(f"  ratio {ratio(allrows):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  top1  {a} {ta}/{n} ({100.0 * ta / n:.1f} %)  "
          f"{b} {tb}/{n} ({100.0 * tb / n:.1f} %)")
    print(f"  {b} better {wins} / worse {losses}  sign-test "
          f"p={sign_p(wins, losses):.3f}")
    for cell, k in ((a, 1), (b, 2)):
        miss = [r for r in allrows if r[k][2] != r[0][0]]
        margins = [r[0][3] for r in miss if r[0][3] is not None]
        gaps = sorted((r[k][4] for r in miss if r[k][4] is not None),
                      reverse=True)
        if not miss:
            print(f"  {cell}: no step off the reference")
            continue
        if not margins:
            print(f"  {cell}: {len(miss)} steps off the reference (no margin "
                  "columns in these logs)")
            continue
        tie = sum(m < TIE for m in margins)
        print(f"  {cell}: {len(miss)} steps off the reference; the reference's "
              f"margin there < {TIE} (a near tie) at {tie}, median "
              f"{sorted(margins)[len(margins) // 2]:.3f}; {cell}'s largest gaps "
              + ", ".join(f"{g:.3f}" for g in gaps[:5]))
    if hi <= 1.03 and tb >= ta - 0.01 * n:
        verdict = f"{b} not worse"
    elif lo > 1.0:
        verdict = f"{b} worse"
    else:
        verdict = "undecided"
    print(f"  verdict {verdict}")
    return verdict


def self_test():
    import tempfile
    rng = random.Random(7)
    with tempfile.TemporaryDirectory() as d:
        def write(path, rows):
            with open(path, "w") as f:
                for s, (t, nll, top1, m, g) in enumerate(rows, 1):
                    f.write(f"[PPL] decode step={s} target={t} nll={nll!r} "
                            f"top1={top1} margin={m} gap={g}\n")
        for p in range(6):
            ref, ra, rb = [], [], []
            for s in range(128):
                t = rng.randrange(1000)
                m = rng.uniform(0.0, 5.0)
                ref.append((t, rng.uniform(0, 1), t, m, 0.0))
                x = rng.uniform(0.0, 1.0)
                pa = t if m > 0.2 else t + 1
                ra.append((t, x, pa, m, 0.0 if pa == t else 0.1))
                rb.append((t, x * 1.001, pa, m, 0.0 if pa == t else 0.1))
            write(os.path.join(d, f"p{p}__ref.log"), ref)
            write(os.path.join(d, f"p{p}__off.log"), ra)
            write(os.path.join(d, f"p{p}__default.log"), rb)
        v = report(load(d, "off", "default"), "off", "default")
        assert v == "default not worse", v
        # B 10 % worse everywhere must be called worse
        for p in range(6):
            rows = read(os.path.join(d, f"p{p}__default.log"))
            write(os.path.join(d, f"p{p}__default.log"),
                  [(t, n * 1.1, i, m, g) for _, (t, n, i, m, g) in
                   sorted(rows.items())])
        v = report(load(d, "off", "default"), "off", "default")
        assert v == "default worse", v
    # round 13's 1664 steps: 2.0**n overflowed (n > 1023)
    assert 0.0 < sign_p(800, 860) < 1.0 and sign_p(0, 0) == 1.0
    print("SELF-TEST OK")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) in (2, 4):
        a, b = (sys.argv[2], sys.argv[3]) if len(sys.argv) == 4 else ("off",
                                                                      "default")
        report(load(sys.argv[1], a, b), a, b)
    else:
        print(__doc__)
        sys.exit(1)
