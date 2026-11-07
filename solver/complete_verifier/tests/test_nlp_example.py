#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
#########################################################################
"""Tests for the NLP API example runner."""

from pathlib import Path
import sys


COMPLETE_VERIFIER_DIR = Path(__file__).resolve().parents[1]
if str(COMPLETE_VERIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(COMPLETE_VERIFIER_DIR))

import nlp_example  # noqa: E402


def test_nlp_bab_br_budget_cli_override_reaches_config():
    parser = nlp_example._build_arg_parser()
    config_path = COMPLETE_VERIFIER_DIR / "exp_configs/NLP_solver/nlplib_new.yaml"

    args = parser.parse_args([
        "--config", str(config_path),
        "--device", "cpu",
        "--nlp_bab_br_budget", "123",
    ])

    config = nlp_example._build_config(args).to_dict()

    assert args.nlp_bab_br_budget == 123
    assert config["solver"]["nlp"]["bab_br_budget"] == 123



def test_controller_cli_reaps_workers_and_flushes_before_process_exit(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from input_split import native_lane
    events = []
    class Exit(Exception):
        pass
    def exit_process(status):
        events.append(('exit', status))
        raise Exit()
    monkeypatch.setattr(nlp_example, '_CONTROLLER_CLI', True)
    monkeypatch.setattr(native_lane, '_close_unused', lambda: events.append('reap'))
    monkeypatch.setattr(nlp_example.os, '_exit', exit_process)
    with monkeypatch.context() as streams:
        streams.setattr(nlp_example.sys, 'stdout', SimpleNamespace(flush=lambda: events.append('stdout')))
        streams.setattr(nlp_example.sys, 'stderr', SimpleNamespace(flush=lambda: events.append('stderr')))
        with pytest.raises(Exit):
            nlp_example._finish_cli(0)
    assert events == ['reap', 'stdout', 'stderr', ('exit', 0)]


def test_flag_off_cli_keeps_normal_python_exit(monkeypatch):
    import pytest
    monkeypatch.setattr(nlp_example, '_CONTROLLER_CLI', False)
    with pytest.raises(SystemExit) as caught:
        nlp_example._finish_cli(3)
    assert caught.value.code == 3
