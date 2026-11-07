"""Write tests/parity_golden/<name>.pt for every instance of parity_instances.tsv and log the seconds each took.
Run from complete_verifier on CPU: python tests/make_parity_goldens.py [name ...]"""
import os
import sys
import time

import torch

import parity_harness as ph


def main(only):
    os.makedirs(ph.GOLDEN, exist_ok=True)
    for bench, name in ph.read_list():
        if only and name not in only:
            continue
        t = time.time()
        torch.save(ph.record(bench, name), ph.golden_path(name))
        print(f'{name}\t{time.time() - t:.1f}s', flush=True)


if __name__ == '__main__':
    main(set(sys.argv[1:]))
