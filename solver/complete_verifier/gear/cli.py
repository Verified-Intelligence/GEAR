"""The ``gear`` command (G5 DESIGN section 1). Run from ``complete_verifier/``:

    python -m gear.cli solve --legacy hs071 --family gams_world --root /path/to/benchmarks_eqexact \
        [--preset gear-f0] [--timing legacy|solve] [--time_limit S] [--device cpu|cuda[:k]] [--set key=value ...] \
        [--result_json out.json]

The last stdout line is ``problem_name=<name> status=<solver status> success=<bool>``, as ``nlp_example.py`` prints
it, so the 505 runners and scorers read either command. ``--timing`` defaults to ``legacy`` here: the CLI keeps the
runner's clock (section 1 Timing).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path


def _value(text: str):
    """A ``--set`` value: a Python literal (number, bool, None, string in quotes), else the raw string."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gear", description="GEAR global solver.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("solve", help="solve one problem")
    s.add_argument("--legacy", metavar="NAME", required=True, help="frozen legacy artifact (the only v1 CLI source)")
    s.add_argument("--family", required=True, choices=["gams_world", "minlplib_linear"])
    s.add_argument("--root", required=True, help="benchmark root holding the artifact families")
    s.add_argument("--preset", default=None, help="preset name or file (default: gear.DEFAULT_PRESET)")
    s.add_argument("--timing", choices=["legacy", "solve"], default="legacy")
    s.add_argument("--time_limit", type=float, default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--feas_tol", type=float, default=None)
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="advanced dotted legacy key, e.g. --set solver.nlp.ipopt_bab_topk=3")
    s.add_argument("--result_json", default=None, help="write Result.to_json() here")
    s.add_argument("--log", default=None, help="append the solver transcript to this file instead of stdout")
    return p


def options_from_args(a):
    import gear as gr
    advanced = {}
    for item in a.set:
        key, sep, text = item.partition("=")
        if not sep:
            raise gr.OptionError(f"--set expects KEY=VALUE, got {item!r}")
        advanced[key.strip()] = _value(text)
    fields = dict(timing=a.timing, advanced=advanced, log=a.log)
    if a.preset is not None:
        fields["preset"] = a.preset
    for name in ("time_limit", "device", "seed", "feas_tol"):
        if getattr(a, name) is not None:
            fields[name] = getattr(a, name)
    return gr.Options(**fields)


def finish(status: int, controller: bool):
    """As ``nlp_example._finish_cli``: with the controller on, reap the warm pool and leave without module teardown."""
    if controller:
        from input_split.native_lane import _close_unused
        _close_unused()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    raise SystemExit(status)


def main(argv=None) -> tuple[int, bool]:
    import gear as gr
    a = build_parser().parse_args(argv)
    options = options_from_args(a)
    nlp = gr.options.resolve(options).config["solver"]["nlp"]
    controller = bool(nlp.get("controller", False) and nlp.get("enabled", False))
    result = gr.solve(gr.read_legacy(a.legacy, root=a.root, family=a.family), options)
    if a.result_json:
        Path(a.result_json).write_text(json.dumps(result.to_json(), sort_keys=True, indent=1) + "\n")
    print(f"problem_name={a.legacy} status={result.internal_status} success={result.internal_status != 'unknown'}")
    failed = any(x.startswith("exception:") for x in result.anomalies)
    return (1 if failed else 0), controller


if __name__ == "__main__":
    finish(*main())
