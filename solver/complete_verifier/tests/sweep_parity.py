"""Compare two sweep run directories (<run>/<bench>/<name>.log).

Usage: python tests/sweep_parity.py BASE_RUN_DIR NEW_RUN_DIR [--noise tests/parity_noise.txt]
Prints gains, losses (verified in BASE, not in NEW) and slowdowns (both verified, new > 1.5 * old + 10 s) and
exits with status 1 when a loss is not on the noise list.
"""
import argparse
import glob
import os
import re
import sys


def read_run(run):
    """{(bench, name): (status, elapsed)}; a log without the #RUNNER line has no status"""
    out = {}
    for path in glob.glob(os.path.join(run, '*', '*.log')):
        with open(path, errors='replace') as f:
            text = f.read()
        status = elapsed = None
        if '#RUNNER' in text:
            s, e = re.findall(r'status=(\w+)', text), re.findall(r'elapsed=([\d.]+)', text)
            status = s[-1] if s else 'none'
            elapsed = float(e[-1]) if e else float('nan')
        out[(os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-4])] = (status, elapsed)
    return out


def read_noise(path):
    if not path:
        return set()
    with open(path) as f:
        return {l.strip() for l in f if l.strip() and not l.startswith('#')}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('new')
    ap.add_argument('--noise')
    a = ap.parse_args(argv)
    base, new, noise = read_run(a.base), read_run(a.new), read_noise(a.noise)
    ok = lambda r, k: r.get(k, (None, None))[0] == 'verified'
    keys = sorted(set(base) | set(new))
    gains = [k for k in keys if ok(new, k) and not ok(base, k)]
    losses = [k for k in keys if ok(base, k) and not ok(new, k)]
    slow = [k for k in keys if ok(base, k) and ok(new, k) and new[k][1] > 1.5 * base[k][1] + 10]
    print(f'n base={len(base)} new={len(new)}  verified base={sum(ok(base, k) for k in keys)} '
          f'new={sum(ok(new, k) for k in keys)}')
    print(f'gains {len(gains)}: ' + ' '.join(n for _, n in gains))
    print(f'losses {len(losses)}: ' + ' '.join(n + ('(noise)' if n in noise else '') for _, n in losses))
    print(f'slowdowns {len(slow)}: ' + ' '.join(f'{n}({base[(b, n)][1]:.0f}s->{new[(b, n)][1]:.0f}s)'
                                                 for b, n in slow))
    return 1 if any(n not in noise for _, n in losses) else 0


if __name__ == '__main__':
    sys.exit(main())
