"""Per-test outcomes of a pytest run, to compare a branch against tests/parity_baseline.txt.

Usage (run pytest with --junitxml=RUN.xml):
  python tests/suite_parity.py write RUN.xml OUT.txt           one line 'outcome<TAB>test id' per test
  python tests/suite_parity.py compare tests/parity_baseline.txt RUN.xml
compare lists every baseline test that is missing or has another outcome, and exits with status 1 if there is one.
Tests that are new in RUN.xml are listed but do not fail the comparison.
"""
import sys
import xml.etree.ElementTree as ET


def outcomes(xml_path):
    out = {}
    for case in ET.parse(xml_path).getroot().iter('testcase'):
        state = 'passed'
        for tag in ('failure', 'error', 'skipped'):
            if case.find(tag) is not None:
                state = {'failure': 'failed'}.get(tag, tag)
                break
        out[f"{case.get('classname')}::{case.get('name')}"] = state
    return out


def read_baseline(path):
    with open(path) as f:
        return dict(l.rstrip('\n').split('\t')[::-1] for l in f if l.strip() and not l.startswith('#'))


def main(argv):
    if argv[0] == 'write':
        with open(argv[2], 'w') as f:
            f.writelines(f'{s}\t{t}\n' for t, s in sorted(outcomes(argv[1]).items()))
        return 0
    base, new = read_baseline(argv[1]), outcomes(argv[2])
    bad = [(t, s, new.get(t, 'missing')) for t, s in sorted(base.items()) if new.get(t) != s]
    for t, was, now in bad:
        print(f'{t}: {was} -> {now}')
    added = sorted(set(new) - set(base))
    print(f'{len(bad)} baseline tests differ; {len(added)} tests are new')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
