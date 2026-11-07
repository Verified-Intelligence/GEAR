#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
# The PGD attack drivers (attack_pgd, general_spec_attack, attack_interface) are not part of this release; their entry
# points raise.
from .attack_utils import (
    default_adv_saver, default_adv_verifier, PGDAttackResult,
    reset_attack_stats, get_attack_stats, check_and_save_cex,
    verify_onnxruntime_output_matches_pytorch)


def _not_in_release(name):
    def removed(*args, **kwargs):
        from auto_LiRPA.not_in_release import not_in_release
        not_in_release(f"the PGD attack ({name})")
    return removed


attack = _not_in_release("attack")
attack_with_general_specs = _not_in_release("attack_with_general_specs")
default_pgd_loss = _not_in_release("default_pgd_loss")
default_early_stop_condition = _not_in_release("default_early_stop_condition")
default_adv_example_finalizer = _not_in_release("default_adv_example_finalizer")
pgd_attack_with_general_specs = _not_in_release("pgd_attack_with_general_specs")
