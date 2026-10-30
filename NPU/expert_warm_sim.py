#!/usr/bin/env python3
"""How many of the first decode tokens' experts NNTR_MOE_WARM=<n> would
have in the pool, from one run's routing logs.

Usage: python3 expert_warm_sim.py moe_trace.txt route_log.txt [n ...]

moe_trace.txt  NNTR_MOE_TRACE: one line per prefill MoE layer call,
               "<layer> <tokens> | <routed> | <top-(k+5) per token>"
route_log.txt  NNTR_HTP_ROUTE_LOG: one line per MoE op per decode token,
               "<op> 1 | <ids> | ..."

The warm list is causallm::warmList's rule (expert_preload.h): the last
prompt token's top-(k+5) in rank order, then the experts with the most
prefill rows (ties to the lower id), the layer's last prefill call only.
For each n (default 0 4 8 13 16) and each of the first decode tokens,
prints how many of the token's routed experts are in the warm sets: at
token 1 that is the misses the warm-up takes off it (less the few the
prefill tail already holds), against reads of n x layers experts under the
graph init.
"""
import sys
from collections import defaultdict


def fields(line):
    parts = line.rstrip("\n").split("|")
    head = parts[0].split()
    lists = [[int(x) for x in p.split()] for p in parts[1:]]
    return head, lists


def warm_list(ranked, per_token, rows, n):
    out, taken = [], set()
    for e in ranked[len(ranked) - per_token:] if per_token else []:
        if len(out) >= n:
            break
        if e not in taken:
            taken.add(e)
            out.append(e)
    rest = sorted((e for e in rows if e not in taken and rows[e]),
                  key=lambda e: (-rows[e], e))
    out.extend(rest[: max(0, n - len(out))])
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    sizes = [int(x) for x in argv[3:]] or [0, 4, 8, 13, 16]
    last = {}  # layer -> (ranked list, tokens) of its last prefill call
    with open(argv[1]) as f:
        for line in f:
            if "|" not in line:
                continue
            head, lists = fields(line)
            layer, tokens = int(head[0]), int(head[1])
            if tokens > 1 and len(lists) >= 2:
                last[layer] = (lists[1], tokens)
    tokens = []  # decode tokens: list of per-op routed sets
    cur = {}
    with open(argv[2]) as f:
        for line in f:
            if "|" not in line:
                continue
            head, lists = fields(line)
            op = int(head[0])
            if op in cur:
                tokens.append(cur)
                cur = {}
            cur[op] = set(lists[0]) if lists else set()
    if cur:
        tokens.append(cur)
    if not last or not tokens:
        raise SystemExit("no prefill calls or no decode tokens in the logs")
    layers = sorted(last)
    if sorted(tokens[0]) != list(range(len(layers))):
        print(f"note: {len(layers)} prefill layers, decode ops "
              f"{sorted(tokens[0])[:3]}..; matched by order")
    ops = sorted(tokens[0])
    per_token = {}
    for layer in layers:
        ranked, n_tok = last[layer]
        per_token[layer] = len(ranked) // n_tok
    k = len(next(iter(tokens[0].values())))
    print(f"prefill layers {len(layers)}, decode tokens {len(tokens)}, "
          f"top-k {k}, ranked per token {per_token[layers[0]]}")
    show = tokens[:4]
    print("n    reads " + " ".join(f"tok{t + 1:<2} hits" for t in
                                   range(len(show))))
    for n in sizes:
        warm = {}
        for layer in layers:
            ranked, n_tok = last[layer]
            pt = per_token[layer]
            rows = defaultdict(int)
            for t in range(n_tok):
                for e in ranked[t * pt: t * pt + k]:
                    rows[e] += 1
            warm[layer] = set(warm_list(ranked, pt, rows, n))
        cells = []
        for tok in show:
            hit = sum(len(tok[op] & warm[layers[i]])
                      for i, op in enumerate(ops) if i < len(layers))
            total = sum(len(tok[op]) for op in ops)
            cells.append(f"{hit:>4}/{total:<4}")
        print(f"{n:<4} {n * len(layers):>5} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
