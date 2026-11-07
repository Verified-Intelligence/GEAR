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
"""
Arguments parser and config file loader.

When adding new commandline parameters, please make sure to provide a clear and descriptive help message and put it in under a related hierarchy.
"""

import re
import os
import sys
import yaml
import time
import argparse
from collections import defaultdict
import torch


class ConfigHandler:
    OP_BUDGET_DEFAULTS = {"hidden_squares": 1 << 20, "complete_squares": 1 << 24,
                          "prove_rows": 1 << 23, "commit": "partial"}
    # solver.nlp keys that the code reads with .get(key, default) and the schema lacked (G5 WP2). The schema default is
    # the code's default, and the "Configurations:" dump omits them while they are at it, so logs stay equal.
    NLP_GAP_DEFAULTS = {"bab_primal_dual_abs_tol": 1e-5, "bab_primal_dual_rel_tol": 1e-4, "crown_engine_tiered": False,
                        "enable_ipopt_in_bab": False, "ipopt_bab_max_fails": 2, "ipopt_bab_timeout": 3.0,
                        "ipopt_bab_topk": 2, "qtb_tangent_timeout": 10.0}

    def __init__(self):
        self.config_file_hierarchies = {
            # Given a hierarchy for each commandline option. This hierarchy is used in yaml config.
            # For example: "batch_size": ["solver", "propagation", "batch_size"] will be an element in this dictionary.
            # The entries will be created in add_argument() method.
        }
        # Stores all arguments according to their hierarchy.
        self.all_args = {}
        # Parses all arguments with their defaults.
        self.defaults_parser = argparse.ArgumentParser()
        # Parses the specified arguments only. Not specified arguments will be ignored.
        self.no_defaults_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        # Help message for each configuration entry.
        self.help_messages = defaultdict(str)
        # Add all common arguments.
        self.add_common_options()
        # Path to the config file
        self.file = None
        # Parse an empty commandline to get all default arguments.
        self.default_args = vars(self.defaults_parser.parse_args([]))

    def add_common_options(self):
        """
        Add all parameters that are shared by different front-ends.
        """

        # We must set how each parameter will be presented in the config file, via the "hierarchy" parameter.
        # Global Configurations, not specific for a particular algorithm.

        # The "--config" option does not exist in our parameter dictionary.
        self.add_argument('--config', type=str, help='Path to YAML format config file.', hierarchy=None)

        h = ["general"]
        self.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                          help='Select device to run verifier, cpu or cuda (GPU).',
                          hierarchy=h + ["device"])
        self.add_argument("--seed", type=int, default=100, help='Random seed.',
                          hierarchy=h + ["seed"])
        self.add_argument("--conv_mode", default="patches", choices=["patches", "matrix"],
                          help='Convolution mode during bound propagation: "patches" mode (default) is very efficient, but may not support all architecture; "matrix" mode is slow but supports all architectures.',
                          hierarchy=h + ["conv_mode"])
        self.add_argument("--deterministic", action='store_true',
                          help='Run code in CUDA deterministic mode, which has slower performance but better reproducibility.',
                          hierarchy=h + ["deterministic"])
        self.add_argument("--double_fp", action='store_true',
                          help='Use double precision floating point. GPUs with good double precision support are preferable (NVIDIA P100, V100, A100, H100; AMD Radeon Instinct MI50, MI100).',
                          hierarchy=h + ["double_fp"])
        self.add_argument("--loss_reduction_func", default="sum",
                          help='When batch size is not 1, this reduction function is applied to reduce the bounds into a scalar (options are "sum" and "min").',
                          hierarchy=h + ["loss_reduction_func"])
        self.add_argument("--no_sparse_alpha", action='store_false',
                          help='Enable/disable sparse alpha.',
                          hierarchy=h + ["sparse_alpha"])
        self.add_argument("--no_sparse_interm", action='store_false',
                          help='Enable/disable sparse intermediate bounds.',
                          hierarchy=h + ["sparse_interm"])
        self.add_argument("--save_adv_example", action='store_true',
                          help='Save returned adversarial example in file.',
                          hierarchy=h + ["save_adv_example"])
        self.add_argument("--verify_onnxruntime_output", action='store_true',
                          help='Validate counterexamples with ONNXRuntime; '
                          'fail if validation cannot run.',
                          hierarchy=h + ["verify_onnxruntime_output"])
        self.add_argument("--eval_adv_example", action='store_true',
                          help='Whether to validate the saved adversarial example.',
                          hierarchy=h + ["eval_adv_example"])
        self.add_argument("--onnx_adv_example", action='store_true',
                          help='Whether to use the ONNX inference result for saving adversarial examples.',
                          hierarchy=h + ["onnx_adv_example"])
        self.add_argument("--show_adv_example", action='store_true',
                          help='Print the adversarial example.',
                          hierarchy=h + ["show_adv_example"])
        self.add_argument("--precompile_jit", action='store_true',
                          help='Precompile jit kernels to speed up after jit-wrapped functions, but will cost extra time at the beginning.',
                          hierarchy=h + ["precompile_jit"])
        self.add_argument("--reset_seed_after_precompile", action='store_true',
                          help='Reset the random seed after precompilation to ensure reproducibility.',
                          hierarchy=h + ["reset_seed_after_precompile"])
        self.add_argument("--prepare_only", action='store_true',
                          help='Prepare to run the instance (e.g., cache vnnlib and converted onnx files) without running the actual verification.',
                          hierarchy=h + ["prepare_only"])
        self.add_argument('--complete_verifier',
                          choices=["bab", "mip", "bab-refine", "skip", "auto", "Customized"], default="bab",
                          help='Complete verification verifier. "bab": branch and bound with beta-CROWN or GCP-CROWN; '
                               '"mip": mixed integer programming (MIP) formulation; '
                               '"bab-refine": branch and bound with intermediate layer bounds computed by MIP; '
                               '"Customized": customized verifier, need to specially def by user.',
                          hierarchy=h + ["complete_verifier"])
        self.add_argument("--csv_name", type=str, default=None,
                          help='Name of .csv file containing a list of properties to verify (VNN-COMP specific).',
                          hierarchy=h + ["csv_name"])
        self.add_argument("--results_file", type=str, default='out.txt',
                          help='Path to results file.',
                          hierarchy=h + ["results_file"])
        self.add_argument("--root_path", type=str, default='',
                          help='Root path of the specification folder if using vnnlib.',
                          hierarchy=h + ["root_path"])
        self.add_argument("--graph_optimizer", type=str, default='Customized("custom_graph_optimizer", "default_optimizer")',
                          help='BoundedModule model graph optimizer function name. For examples of customized graph optimizer, please see the config files for '
                               'the gtrsb benchmark in VNN-COMP 2023.',
                          hierarchy=h + ["graph_optimizer"])
        self.add_argument("--buffer_has_batchdim", action="store_true",
                          help='In most cases, the shape of buffers in an ONNX graph do not have a batch dimension. '
                               'Enabling this option will help load models with a buffer object that has a batch dimension. '
                               'In most case this can be inferred in the verifier automatically, and this option is not needed.',
                          hierarchy=h + ["buffer_has_batchdim"])
        self.add_argument("--save_output", action='store_true',
                          help='Save output for test.',
                          hierarchy=h + ["save_output"])
        self.add_argument("--output_file", type=str, default='out.pkl',
                          help="Path to the output file.",
                          hierarchy=h + ["output_file"])
        self.add_argument("--return_optimized_model", action='store_true',
                          help="Return the model with optimized bounds after incomplete verification is done.",
                          hierarchy=h + ["return_optimized_model"])
        self.add_argument("--store_all_specs_on_cpu", action='store_true',
                          help='Store all specifications on CPU to save GPU memory. '
                               'This is useful when the number of specifications is large or the model is large. but it will lead additional transfer overhead.',
                            hierarchy=h + ["store_all_specs_on_cpu"])
        self.add_argument("--adhoc_tuning", type=str, default=None,
                          help='Ad-hoc tuning function name. This function will be called with the model and vnnlib_handler as arguments.',
                          hierarchy=h + ["adhoc_tuning"])

        h = ["model"]
        self.add_argument("--model", type=str, default=None, help='Model name. Will be evaluated as a python statement.',
                          hierarchy=h + ["name"])
        self.add_argument("--load_model", type=str, default=None,
                          help='Load pretrained model from this specified path.',
                          hierarchy=h + ["path"])
        self.add_argument("--onnx_path", type=str, default=None, help='Path to .onnx model file.',
                          hierarchy=h + ["onnx_path"])
        self.add_argument("--cache_onnx_conversion", action='store_true',
                          help='Cache the model converted from ONNX.',
                          hierarchy=h + ["cache_onnx_conversion"])
        self.add_argument("--debug_onnx", action='store_true',
                          help='Debug onnx conversion.',
                          hierarchy=h + ["debug_onnx"])
        self.add_argument('--onnx_quirks', type=str, default=None,
                          help="Load onnx model with quirks to workaround onnx model issue. "
                               "This string will be passed to onnx2pytorch as the 'quirks' argument, and it is typically a literal of a python dict, e.g., \"{'Reshape': {'fix_batch_size: True'}}\".",
                          hierarchy=h + ["onnx_quirks"])
        self.add_argument("--input_shape", nargs='+', type=int, default=None,
                          help='Specified input shape of the model.'
                               'Usually the shape can be automatically determined from dataset or onnx model, but some onnx models may have an incompatible shape (without batch dim). '
                               'You can specify shape explicitly here.'
                               'The shape should be (-1, input_shape) like (-1, 3, 32, 32) and -1 indicates the batch dim.',
                          hierarchy=h + ["input_shape"])
        self.add_argument("--onnx_loader", type=str, default='default_onnx_and_vnnlib_loader',
                          help='ONNX model loader function name. Can be the Customized() primitive; for examples of customized model loaders, please see the config files for '
                               'the marabou-cifar10 benchmark in VNN-COMP 2021 and the Carvana benchmark in VNN-COMP 2022.',
                          hierarchy=h + ["onnx_loader"])
        self.add_argument("--onnx_optimization_flags", nargs='+',
                          choices=["merge_bn", "merge_linear", "merge_gemm",
                                   "remove_ineffective_layers",
                                   "remove_relu_in_last_layer",
                                   "optimize_relu_relationship"],
                          default=[], help='Onnx graph optimization config.',
                          hierarchy=h + ["onnx_optimization_flags"])
        self.add_argument("--onnx_vnnlib_joint_optimization_flags", nargs='+',
                          choices=["peel_off_last_softmax_layer", "none"],
                          default="none", help='Joint optimization that changes both onnx model and vnnlib.',
                          hierarchy=h + ["onnx_vnnlib_joint_optimization_flags"])
        self.add_argument('--check_optimized', action='store_true',
                          help='Check the optimized onnx file instead the original one when converting to pytorch. This is used when input shape is changed during optimization.',
                          hierarchy=h + ["check_optimized"])
        self.add_argument("--use_ort_optimized_model", action='store_true',
                          help='Load the ONNX Runtime optimized graph before ONNX-to-PyTorch conversion.',
                          hierarchy=h + ["use_ort_optimized_model"])
        self.add_argument("--flatten_final_output", action='store_true',
                          help='Manually add a flatten layer at the end of the model.',
                          hierarchy=h + ["flatten_final_output"])
        self.add_argument("--optimize_graph", type=str,
                          help="Specify a custom function for optimizing the graph on the BoundedModule.",
                          hierarchy=h + ["optimize_graph"])
        self.add_argument('--model_with_jacobian', action='store_true',
                          help='Indicate that the model contains JacobianOP.',
                          hierarchy=h + ['with_jacobian'])
        self.add_argument("--bound_opts", type=str, default=None,
                          help='Dictionary of options to be passed to BoundedModule. Should be a valid Python dictionary literal.',
                          hierarchy=h + ["bound_opts"])
        # Operation budgets of the in-graph proof stages (G5 section 4.1). One op unit is a
        # polynomial built or checked, a term pair multiplied, a build() call, a scales()
        # element, a node visited in the hidden-square loop, or a row proved in prove_rows.
        # Each default is the smallest power of 2 >= 2x the largest count that the 505
        # frozen instances used where the former 5 s / 15 s / 5 s deadline did not fire
        # (370162 / 8190888 / 2753500; WP4 report). They are resource caps: the graph and
        # root bounds are identical on 505/505 with and without them.
        self.add_argument("--op_budget_hidden_squares", type=int, default=self.OP_BUDGET_DEFAULTS["hidden_squares"],
                          help='Operation budget of rewrite_hidden_squares (replaces a 5 s deadline).',
                          hierarchy=h + ["op_budget", "hidden_squares"])
        self.add_argument("--op_budget_complete_squares", type=int, default=self.OP_BUDGET_DEFAULTS["complete_squares"],
                          help='Operation budget of rewrite_complete_squares (replaces a 15 s deadline).',
                          hierarchy=h + ["op_budget", "complete_squares"])
        self.add_argument("--op_budget_prove_rows", type=int, default=self.OP_BUDGET_DEFAULTS["prove_rows"],
                          help='Operation budget of each prove_rows call (replaces a 5 s deadline).',
                          hierarchy=h + ["op_budget", "prove_rows"])
        self.add_argument("--op_budget_commit", type=str, default=self.OP_BUDGET_DEFAULTS["commit"], choices=["partial", "whole"],
                          help='Commit mode: partial keeps the rewrites applied before a stage exhausts its budget; '
                               'whole raises so that the stage is dropped entirely.',
                          hierarchy=h + ["op_budget", "commit"])

        h = ["data"]
        self.add_argument("--start", type=int, default=0, help='Start from the i-th property in specified dataset.',
                          hierarchy=h + ["start"])
        self.add_argument("--end", type=int, default=10000, help='End with the (i-1)-th property in the dataset.',
                          hierarchy=h + ["end"])
        self.add_argument("--select_instance", type=int, nargs='+', default=None,
                          help='Select a list of instances to verify.',
                          hierarchy=h + ["select_instance"])
        # FIXME --num_outputs should be inferred automatically.
        self.add_argument('--num_outputs', type=int, default=10,
                          help="Number of classes for classification problem.",
                          hierarchy=h + ["num_outputs"])
        self.add_argument("--mean", nargs='+', type=float, default=0.0,
                          help='Mean vector used in data preprocessing.',
                          hierarchy=h + ["mean"])
        self.add_argument("--std", nargs='+', type=float, default=1.0,
                          help='Std vector used in data preprocessing.',
                          hierarchy=h + ["std"])
        self.add_argument('--pkl_path', type=str, default=None,
                          help="Load properties to verify from a .pkl file (only used for oval20 dataset).",
                          hierarchy=h + ["pkl_path"])
        self.add_argument("--dataset", type=str, default=None,
                          help="Dataset name (only if not using specifications from a .csv file). Dataset must be defined in data_utils.py. For customized data, checkout custom/custom_model_data.py.",
                          hierarchy=h + ["dataset"])
        self.add_argument("--filter_path", type=str, default=None,
                          help='A filter in pkl format contains examples that will be skipped (not used).',
                          hierarchy=h + ["data_filter_path"])
        self.add_argument("--data_idx_file", type=str, default=None,
                          help='A text file with a list of example IDs to run.',
                          hierarchy=h + ["data_idx_file"])

        h = ["specification"]
        self.add_argument("--spec_type", type=str, default='lp', choices=['lp', 'box', 'bound'],
                          help='Type of verification specification. "lp" = L_p norm, "box" = element-wise lower and upper bound provided by dataloader.',
                          hierarchy=h + ["type"])
        self.add_argument("--robustness_type", type=str, default="verified-acc",
                          choices=["verified-acc", "runnerup", "clean-acc", "specify-target", "all-positive"],
                          help='For robustness verification: verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), '
                               'or using a specified label in dataset ("specify-target" mode, only used for oval20). Not used when a VNNLIB spec is used.',
                          hierarchy=h + ["robustness_type"])
        self.add_argument("--norm", type=float, default=float('inf'),
                          help='Lp-norm for epsilon perturbation in robustness verification (1, 2, inf).',
                          hierarchy=h + ["norm"])
        self.add_argument("--epsilon", type=float, default=None,
                          help='Set perturbation size (Lp norm). If not set, a default value may be used based on dataset loader.',
                          hierarchy=h + ["epsilon"])
        self.add_argument("--epsilon_min", type=float, default=0.,
                          help='Set an optional minimum perturbation size (Lp norm).',
                          hierarchy=h + ["epsilon_min"])
        self.add_argument("--vnnlib_path", type=str, default=None,
                          help='Path to .vnnlib specification file. Will override any Lp/robustness verification arguments.', hierarchy=h + ["vnnlib_path"])
        self.add_argument("--shrink_eps", type=float, default=None,
                          help='Shrink the input bounds.',
                          hierarchy=h + ['shrink_eps'])
        self.add_argument("--rhs_offset", type=float, default=None,
                          help='Adding an offset to RHS.',
                          hierarchy=h + ['rhs_offset'])

        h = ["solver"]
        self.add_argument("--batch_size", type=int, default=64,
                          help='Batch size in bound solver (number of parallel splits).',
                          hierarchy=h + ["batch_size"])
        self.add_argument("--no_auto_enlarge_batch_size", dest='auto_enlarge_batch_size', action='store_false', default=True,
                          help='Disable automatic increase of batch size based on --batch_size in bab '
                               'when current VRAM usage < 45%%; only supports input_split.',
                          hierarchy=h + ["auto_enlarge_batch_size"])
        self.add_argument("--auto_enlarge_batch_size", dest='auto_enlarge_batch_size', action='store_true',
                          help='Override to enable auto enlarge batch size (for internal tests).',
                          hierarchy=h + ["auto_enlarge_batch_size"], private=True)
        self.add_argument("--batched_crown_auto_enlarge_max_vram_ratio", type=float, default=0.9,
                          help='Automatically increase batch size based on --batch_size in bab '
                               'if current VRAM usage < (batched_crown_auto_enlarge_max_vram_ratio / 2)%%, only support batched crown.',
                          hierarchy=h + ["batched_crown_auto_enlarge_max_vram_ratio"])
        self.add_argument('--min_batch_size_ratio', type=float, default=0.1,
                          help='The minimum batch size ratio in each iteration (splitting multiple layers if the number of domains is smaller than min_batch_size_ratio * batch_size).',
                          hierarchy=h + ["min_batch_size_ratio"])
        self.add_argument('--early_stop_patience', type=int, default=10,
                          help='Number of iterations that we will start considering early stop if tracking no improvement.',
                          hierarchy=h + ["early_stop_patience"])
        self.add_argument('--start_save_best', type=float, default=0.5,
                          help='Start to save best optimized bounds when i > int(iteration*start_save_best). Early iterations are skipped for better efficiency.',
                          hierarchy=h + ["start_save_best"])
        self.add_argument("--reproduce_alpha_bounds", action="store_true",
                          help='Try to ensure that returned optimization '
                               'parameters reproduce the returned bounds during '
                               'debugging. This also preserves the alpha storage '
                               'dtype even when alpha_dtype is "float16"; it '
                               'does not select the arithmetic precision used '
                               'by alpha-CROWN.',
                          hierarchy=h + ["reproduce_alpha_bounds"])
        self.add_argument('--bound_prop_method', default="alpha-crown",
                          choices=["lp", "lp+crown", "alpha-crown", "crown", "forward", "forward+crown",
                                   "alpha-forward", "crown-ibp", "init-crown", "ibp",
                                   "dynamic-forward", "dynamic-forward+crown", "dynamic-forward+backward"],
                          help='Bound propagation method used for incomplete verification and input split based branch and bound.',
                          hierarchy=h + ["bound_prop_method"])
        self.add_argument('--init_bound_prop_method', default="same",
                          choices=["same", "alpha-crown", "crown", "forward",
                                   "forward+crown", "alpha-forward", "crown-ibp",
                                   "init-crown"],
                          help='Bound propagation method used for the initial bound in input split based branch and bound. '
                               'If "same" is specified, then it will use the same method as "bound_prop_method".',
                          hierarchy=h + ["init_bound_prop_method"])
        self.add_argument("--no_prune_after_crown", action='store_false', dest='prune_after_crown',
                          help='By default, we prune verified specifications before starting the alpha-CROWN pass after CROWN pass. Set it to disable this feature.',
                          hierarchy=h + ["prune_after_crown"])
        self.add_argument("--optimize_disjuncts_separately", action='store_true',
                          help="If set, each neuron computes separate bounds for each disjunct. If set, do not set prune_after_crown=True.",
                          hierarchy=h + ["optimize_disjuncts_separately"])
        self.add_argument('--build_batch_size', type=int,
                          help='Maximum number of specifications to handle. Decrease it when OOM.',
                          default=1000000, hierarchy=h + ["build_batch_size"])
        self.add_argument('--no_skip_with_refined_bound', action='store_false', dest='skip_with_refined_bound',
                          hierarchy=h + ['skip_with_refined_bound'],
                          help='In build_with_refined_bounds(), by default we skip alpha-CROWN if all alphas are already initialized. Setting this to avoid this feature.')
        self.add_argument("--forward_before_compute_bounds", action="store_true",
                          help="If set, run forward pass before computing bounds (Sometimes useful for updating the model state before computing bounds).",
                          hierarchy=h + ["forward_before_compute_bounds"])

        h = ["solver", "nlp"]
        self.add_argument('--nlp_crown_engine_sigma_main', action='store_true', default=False,
                          help='Select native or sparse main bounds by measured reward rate and memory.',
                          hierarchy=h+['crown_engine_sigma_main'])
        self.add_argument('--nlp_crown_engine_m1', action='store_true', default=False,
                          help='Cold restarted sparse diagonal-QP node bounds; no CPU LP or simplex.',
                          hierarchy=h+['crown_engine_m1'])
        self.add_argument('--nlp_crown_engine_m1_steps', type=int, default=1000,
                          help='Fixed PDHG node iteration cap (at most 4000).', hierarchy=h+['crown_engine_m1_steps'])
        self.add_argument('--nlp_crown_engine_halpern', action='store_true', default=False,
                          help='Use reflected restarted Halpern PDHG for M1 and linear M2 tables.',
                          hierarchy=h+['crown_engine_halpern'])
        self.add_argument('--nlp_crown_engine_implied_box', action='store_true', default=False,
                          help='Tighten each linear/diagonal QP table box by certified row FBBT.',
                          hierarchy=h+['crown_engine_implied_box'])
        self.add_argument('--nlp_crown_engine_m1_trial_rental', action='store_true', default=False,
                          help='Ski-rent M1 trials by seconds: after f unpaid trials wait 2^(f-1) x the last trial\'s '
                               'seconds of native time before the next one.',
                          hierarchy=h+['crown_engine_m1_trial_rental'])
        self.add_argument('--nlp_crown_engine_m2', action='store_true', default=False,
                          help='Persistent shared-table root OBBT, interleaved by measured time.',
                          hierarchy=h+['crown_engine_m2'])
        self.add_argument('--nlp_crown_engine_m2_rounds', type=int, default=100,
                          help='Legacy Jacobi setting; ignored by asynchronous M2.', hierarchy=h+['crown_engine_m2_rounds'])
        self.add_argument('--nlp_crown_engine_m2_steps', type=int, default=1000,
                          help='Legacy Jacobi setting; ignored by asynchronous M2.', hierarchy=h+['crown_engine_m2_steps'])
        self.add_argument('--no_nlp_crown_engine_m2_convergence', action='store_false', default=True,
                          help='Disable support KKT stopping relative to achieved tightening and GEAR gap tolerance.',
                          hierarchy=h+['crown_engine_m2_convergence'])
        self.add_argument('--no_nlp_crown_engine_m2_stall', action='store_false', default=True,
                          help='Legacy Jacobi setting; asynchronous M2 uses per-lane retirement.',
                          hierarchy=h+['crown_engine_m2_stall'])
        self.add_argument("--enable_nlp", action='store_true',
                          help='Enable NLP-based falsification at root node for input split.',
                          hierarchy=h + ["enabled"])
        self.add_argument("--nlp_max_iter", type=int, default=1000,
                          help='Maximum NLP local-solver iterations.',
                          hierarchy=h + ["max_iter"])
        self.add_argument("--nlp_tol", type=float, default=1e-9,
                          help='NLP solver optimality tolerance.',
                          hierarchy=h + ["tol"])
        self.add_argument("--nlp_constr_viol_tol", type=float, default=1e-4,
                          help='NLP solver constraint violation tolerance for feasibility.',
                          hierarchy=h + ["constr_viol_tol"])
        self.add_argument("--nlp_timeout", type=float, default=60.0,
                          help='NLP local-solver timeout (seconds).',
                          hierarchy=h + ["timeout"])
        self.add_argument("--nlp_verbose", action='store_true',
                          help='Print NLP solver output.',
                          hierarchy=h + ["verbose"])
        self.add_argument("--nlp_method", type=str, default='ipopt',
                          choices=['ipopt'],
                          help='Root local solver method.',
                          hierarchy=h + ["method"])
        self.add_argument("--nlp_enable_local_solver", action='store_true',
                          help='Enable the selected root local solver before or alongside ALM.',
                          hierarchy=h + ["enable_local_solver"])
        # ALM-PGD specific arguments
        self.add_argument("--nlp_enable_alm", action='store_true',
                          help='Enable ALM-PGD solver for comparison with IPOPT.',
                          hierarchy=h + ["enable_alm"])
        self.add_argument("--nlp_expand_input_constraints_as_outputs", action='store_true',
                          default=False,
                          help='Expose shared input linear constraints as proof-time output rows '
                               'in the new API to match legacy expanded NLP benchmarks.',
                          hierarchy=h + ["expand_input_constraints_as_outputs"])
        self.add_argument("--nlp_root_local_solver_warmstart", action='store_true', default=True,
                          help='Run the selected root local solver to warm-start ALM (default: on).',
                          hierarchy=h + ["root_local_solver_warmstart"])
        self.add_argument("--nlp_root_parallel", action='store_true',
                          help='Run the selected root local solver and ALM in parallel at root.',
                          hierarchy=h + ["root_parallel"])
        self.add_argument("--nlp_root_budget_frac", type=float, default=0.5,
                          help='Cap root NLP/IPOPT wall time at this fraction of the remaining '
                               'global timeout (<=0 disables the cap).',
                          hierarchy=h + ["root_budget_frac"])
        self.add_argument("--nlp_controller", action='store_true', default=False,
                          help="Publish root candidates and certified LP bounds independently, with receipt accounting.",
                          hierarchy=h + ["controller"])
        self.add_argument("--nlp_unified_root", action='store_true', default=False,
                          help='Run the root operations under the run deadline alone (no share of the remaining '
                               'time per phase), check the gap rule after each, bound the lifted LP before its '
                               'OBBT and B&B, and stop the root ALM once its last outer iteration did not beat '
                               "IPOPT's verified incumbent.",
                          hierarchy=h + ["unified_root"])
        self.add_argument("--nlp_root_hybrid_term_bound", action='store_true', default=False,
                          help='Tighten the root objective with per-term IBP/CROWN bounds.',
                          hierarchy=h + ["root_hybrid_term_bound"])
        self.add_argument("--nlp_root_hybrid_max_terms", type=int, default=64,
                          help='Maximum number of scalar terms in the root hybrid objective bound.',
                          hierarchy=h + ["root_hybrid_max_terms"])
        self.add_argument("--nlp_pre_incomplete", action='store_true', default=False,
                          help='Run the root primal NLP/ALM search before incomplete bound '
                               'construction. Useful when bound propagation is singular or '
                               'consumes the primal-search budget.',
                          hierarchy=h + ["pre_incomplete"])
        self.add_argument("--nlp_pre_incomplete_stop_on_verified",
                          action='store_true', default=False,
                          help='In solving mode, stop after pre-incomplete search finds a '
                               'raw-verified primal instead of starting dual proof.',
                          hierarchy=h + ["pre_incomplete_stop_on_verified"])
        self.add_argument("--nlp_alm_rho0", type=float, default=10,
                          help='Initial penalty parameter for ALM.',
                          hierarchy=h + ["alm_rho0"])
        self.add_argument("--nlp_alm_kappa", type=float, default=5,
                          help='Penalty increase factor for ALM.',
                          hierarchy=h + ["alm_kappa"])
        self.add_argument("--nlp_alm_outer_max_iter", type=int, default=20,
                          help='Maximum outer iterations for ALM.',
                          hierarchy=h + ["alm_outer_max_iter"])
        self.add_argument("--nlp_alm_inner_steps", type=int, default=100,
                          help='Inner PGD steps per ALM outer iteration.',
                          hierarchy=h + ["alm_inner_steps"])
        self.add_argument("--nlp_alm_inner_lr", type=float, default=1,
                          help='Learning rate for inner PGD optimizer.',
                          hierarchy=h + ["alm_inner_lr"])
        self.add_argument("--nlp_alm_inner_lr_outer_decay", type=float, default=1.0,
                          help='Per-outer multiplicative decay for ALM inner learning rate.',
                          hierarchy=h + ["alm_inner_lr_outer_decay"])
        self.add_argument("--nlp_alm_inner_lr_min_factor", type=float, default=1e-6,
                          help='Minimum outer-decay factor applied to ALM inner learning rate.',
                          hierarchy=h + ["alm_inner_lr_min_factor"])
        self.add_argument("--nlp_alm_obj_patience", type=int, default=5,
                          help='Early stop outer loop if best_obj unchanged for n iters (-1 to disable).',
                          hierarchy=h + ["alm_obj_patience"])
        self.add_argument("--nlp_alm_obj_improvement_tol", type=float, default=1e-4,
                          help='Absolute tolerance for objective improvement detection in outer loop.',
                          hierarchy=h + ["alm_obj_improvement_tol"])
        self.add_argument("--nlp_alm_obj_gap_threshold", type=float, default=0.1,
                          help='Min gap between best and 2nd-best domain obj for outer early stop (0=disabled).',
                          hierarchy=h + ["alm_obj_gap_threshold"])
        self.add_argument("--nlp_alm_first_feas_patience", type=int, default=1,
                          help='Max extra outer iters after ANY domain first becomes feasible '
                               'without global best feasible obj improving (-1 to disable).',
                          hierarchy=h + ["alm_first_feas_patience"])
        self.add_argument("--nlp_alm_inner_plateau_check", type=int, default=10,
                          help='Check L_val plateau every N inner steps (0=disabled).',
                          hierarchy=h + ["alm_inner_plateau_check"])
        self.add_argument("--nlp_alm_inner_plateau_rtol", type=float, default=1e-3,
                          help='Relative tolerance for L_val plateau in inner loop.',
                          hierarchy=h + ["alm_inner_plateau_rtol"])
        self.add_argument("--nlp_alm_inner_plateau_patience", type=int, default=2,
                          help='Consecutive plateau checks before inner early stop.',
                          hierarchy=h + ["alm_inner_plateau_patience"])
        self.add_argument("--nlp_alm_scale_eps_with_bias", action='store_true', default=True,
                          help='Use the legacy Adam first-step clipping-cap expansion.',
                          hierarchy=h + ["alm_scale_eps_with_bias"])
        self.add_argument("--nlp_alm_constraint_normalize", action='store_true', default=False,
                          help='Optimize constraints after positive per-row RMS scaling; '
                               'final feasibility remains checked on raw residuals.',
                          hierarchy=h + ["alm_constraint_normalize"])
        self.add_argument("--nlp_alm_constraint_scale_floor", type=float, default=1.0,
                          help='Minimum positive residual RMS scale used by ALM preconditioning.',
                          hierarchy=h + ["alm_constraint_scale_floor"])
        self.add_argument("--nlp_alm_constraint_scale_decay", type=float, default=1.0,
                          help='Per-outer decay of ALM row scales toward the scale floor.',
                          hierarchy=h + ["alm_constraint_scale_decay"])
        self.add_argument("--nlp_alm_update_multipliers_always", action='store_true', default=False,
                          help='Update inequality multipliers every outer iteration instead of '
                               'requiring the whole restart to pass the epsilon gate.',
                          hierarchy=h + ["alm_update_multipliers_always"])
        self.add_argument("--nlp_alm_output_affine_project_init",
                          action='store_true', default=False,
                          help='Detect affine generated equality pairs from constant opposite '
                               'Jacobians and project root ALM seeds onto their manifold.',
                          hierarchy=h + ["alm_output_affine_project_init"])
        self.add_argument("--nlp_alm_output_affine_project_fraction", type=float, default=1.0,
                          help='Fraction of root ALM restarts constrained to the detected '
                               'generated affine-equality manifold.',
                          hierarchy=h + ["alm_output_affine_project_fraction"])
        self.add_argument("--nlp_alm_output_affine_project_sweeps", type=int, default=3,
                          help='Generated affine-equality projection / box-clamp sweeps.',
                          hierarchy=h + ["alm_output_affine_project_sweeps"])
        self.add_argument("--nlp_alm_output_affine_inner_sweeps", type=int, default=0,
                          help='Hard-manifold projection sweeps after every inner PGD step.',
                          hierarchy=h + ["alm_output_affine_inner_sweeps"])
        self.add_argument("--nlp_alm_output_affine_detection_tol", type=float, default=1e-9,
                          help='Jacobian constancy and opposite-pair detection tolerance.',
                          hierarchy=h + ["alm_output_affine_detection_tol"])
        self.add_argument("--nlp_alm_output_affine_pair_slack_tol", type=float, default=1e-4,
                          help='Maximum normalized width for a generated affine '
                               'paired-equality slab.',
                          hierarchy=h + ["alm_output_affine_pair_slack_tol"])
        self.add_argument("--nlp_alm_raw_polish_steps", type=int, default=0,
                          help='Local trust-region PGD steps minimizing the exact raw max constraint '
                               'violation after global ALM restoration.',
                          hierarchy=h + ["alm_raw_polish_steps"])
        self.add_argument("--nlp_alm_raw_polish_lr", type=float, default=1e-3,
                          help='Initial raw-minimax trust radius in unit-box coordinates.',
                          hierarchy=h + ["alm_raw_polish_lr"])
        self.add_argument("--nlp_alm_raw_polish_restarts", type=int, default=8,
                          help='Number of jittered local clones for raw-minimax polish.',
                          hierarchy=h + ["alm_raw_polish_restarts"])
        self.add_argument("--nlp_alm_raw_polish_max_parallel", type=int, default=65536,
                          help='Maximum total domain-times-clone candidates in raw polish.',
                          hierarchy=h + ["alm_raw_polish_max_parallel"])
        self.add_argument("--nlp_alm_raw_polish_jitter", type=float, default=1e-3,
                          help='Local clone jitter as a fraction of each box width.',
                          hierarchy=h + ["alm_raw_polish_jitter"])
        self.add_argument("--nlp_alm_raw_polish_max_viol", type=float, default=10.0,
                          help='Skip raw-minimax polish when archived violation exceeds this.',
                          hierarchy=h + ["alm_raw_polish_max_viol"])
        self.add_argument("--nlp_alm_endgame_time_frac", type=float, default=0.0,
                          help='Wall-clock share reserved for the raw polish and Gauss-Newton '
                               'restoration endgame. The ALM outer loop stops at '
                               '(1-frac)*timeout so the endgame is never entered expired.',
                          hierarchy=h + ["alm_endgame_time_frac"])
        self.add_argument("--nlp_alm_endgame_polish_frac", type=float, default=1.0,
                          help='Share of the reserved endgame budget the raw minimax polish may '
                               'consume; the remainder is kept for Gauss-Newton. 1.0 reproduces '
                               'the legacy behaviour where polish can starve GN.',
                          hierarchy=h + ["alm_endgame_polish_frac"])
        self.add_argument("--nlp_alm_batched_gn_steps", type=int, default=0,
                          help='Batched Gauss-Newton feasibility restoration iterations run '
                               'after the ALM outer loop for every infeasible domain (0=off).',
                          hierarchy=h + ["alm_batched_gn_steps"])
        self.add_argument("--nlp_alm_batched_gn_seeds", type=int, default=16,
                          help='Seeds per infeasible domain for batched Gauss-Newton restoration.',
                          hierarchy=h + ["alm_batched_gn_seeds"])
        self.add_argument("--nlp_alm_batched_gn_damping", type=float, default=1e-6,
                          help='Batched GN Levenberg-Marquardt damping, relative to max diag(J^T J).',
                          hierarchy=h + ["alm_batched_gn_damping"])
        self.add_argument("--nlp_alm_batched_gn_trust_radius", type=float, default=1.0,
                          help='Batched GN maximum unit-box step (infinity norm).',
                          hierarchy=h + ["alm_batched_gn_trust_radius"])
        self.add_argument("--nlp_alm_batched_gn_max_viol", type=float, default=1e12,
                          help='Batched GN skips seeds with a larger raw violation.',
                          hierarchy=h + ["alm_batched_gn_max_viol"])
        self.add_argument("--nlp_alm_batched_gn_max_elems", type=int, default=1 << 24,
                          help='Batched GN cap on candidates*rows*vars of the Jacobian.',
                          hierarchy=h + ["alm_batched_gn_max_elems"])
        self.add_argument("--nlp_alm_init_scaled_fraction", type=float, default=0.0,
                          help='Share of ALM random restarts replaced by multi-scale seeds '
                               'around the projection of the origin onto the box (0=off).',
                          hierarchy=h + ["alm_init_scaled_fraction"])
        self.add_argument("--nlp_alm_init_scaled_min", type=float, default=1e-3,
                          help='Smallest relative scale of the multi-scale restarts '
                               '(log-uniform in [min, 1]).',
                          hierarchy=h + ["alm_init_scaled_min"])
        self.add_argument("--nlp_alm_batched_gn_improve", action='store_true', default=False,
                          help='Batched GN also searches objective <= incumbent - margin in every '
                               'domain without such a point (search-only row; raw acceptance).',
                          hierarchy=h + ["alm_batched_gn_improve"])
        self.add_argument("--nlp_alm_batched_gn_improve_rel", type=float, default=1e-4,
                          help='Relative margin of the batched GN objective level row.',
                          hierarchy=h + ["alm_batched_gn_improve_rel"])
        self.add_argument("--nlp_alm_batched_gn_max_time", type=float, default=0.0,
                          help='Wall-clock cap (s) of one batched GN restoration call '
                               '(0 = until the ALM deadline).',
                          hierarchy=h + ["alm_batched_gn_max_time"])
        self.add_argument("--nlp_alm_batched_gn_interior", type=float, default=0.0,
                          help='Batched GN keeps iterates this absolute distance inside the box '
                               '(sqrt-like derivatives are infinite on a bound; 0=off).',
                          hierarchy=h + ["alm_batched_gn_interior"])
        self.add_argument("--nlp_alm_opt_polish_candidates", type=int, default=0,
                          help='Local optimality phase after ALM: run a batched SQP (damped BFGS, '
                               'l1 merit) from this many distinct final restart points and keep a '
                               'polished point when it is raw-feasible and better. 0 disables.',
                          hierarchy=h + ["alm_opt_polish_candidates"])
        self.add_argument("--nlp_alm_opt_polish_iters", type=int, default=50,
                          help='SQP iterations of the ALM optimality polish.',
                          hierarchy=h + ["alm_opt_polish_iters"])
        self.add_argument("--nlp_alm_opt_polish_time", type=float, default=5.0,
                          help='Wall-clock cap in seconds of the ALM optimality polish.',
                          hierarchy=h + ["alm_opt_polish_time"])
        self.add_argument("--nlp_alm_opt_polish_target_frac", type=float, default=0.9,
                          help='The optimality polish drives constraints to g <= frac * constr_viol_tol.',
                          hierarchy=h + ["alm_opt_polish_target_frac"])
        self.add_argument("--nlp_alm_opt_polish_near_quantile", type=float, default=0.05,
                          help='Near-feasible candidate pool of the optimality polish: the '
                               'least-violated share of the final restart points.',
                          hierarchy=h + ["alm_opt_polish_near_quantile"])
        self.add_argument("--nlp_alm_opt_polish_seeds", type=int, default=0,
                          help='The optimality polish also starts from this many restart start points '
                               '(restart 0 is the box midpoint or the warm start).',
                          hierarchy=h + ["alm_opt_polish_seeds"])
        self.add_argument("--nlp_alm_deduplicate_expanded_input_constraints",
                          action='store_true', default=False,
                          help='When structural input constraints were expanded into model outputs, '
                               'do not append the identical input-side rows again inside ALM.',
                          hierarchy=h + ["alm_deduplicate_expanded_input_constraints"])
        # ALM-PGD in BaB loop
        self.add_argument("--nlp_in_bab", action='store_true',
                          help='Enable ALM-PGD attacks during BaB iterations (not just root).',
                          hierarchy=h + ["in_bab"])
        self.add_argument("--nlp_bab_interval", type=int, default=5,
                          help='Run ALM-PGD every N BaB iterations.',
                          hierarchy=h + ["bab_interval"])
        self.add_argument("--nlp_bab_num_restarts", type=int, default=1000,
                          help='Number of restarts for ALM-PGD in BaB (smaller for speed).',
                          hierarchy=h + ["bab_num_restarts"])
        self.add_argument("--nlp_bab_timeout", type=float, default=10.0,
                          help='Timeout for ALM-PGD in BaB iterations.',
                          hierarchy=h + ["bab_timeout"])
        self.add_argument("--nlp_bab_feas_tol", type=float, default=1e-4,
                          help='Constraint violation tolerance when accepting ALM-PGD BaB primal candidates.',
                          hierarchy=h + ["bab_feas_tol"])
        self.add_argument("--nlp_bab_outer_max_iter", type=int, default=20,
                          help='Max outer iterations for ALM-PGD in BaB (smaller for speed).',
                          hierarchy=h + ["bab_outer_max_iter"])
        self.add_argument("--nlp_bab_inner_steps", type=int, default=50,
                          help='Inner PGD steps per ALM outer iteration in BaB (smaller than root for speed).',
                          hierarchy=h + ["bab_inner_steps"])
        self.add_argument("--nlp_bab_alm_opt_polish_candidates", type=int, default=-1,
                          help='ALM optimality polish candidates in BaB; negative keeps alm_opt_polish_candidates.',
                          hierarchy=h + ["bab_alm_opt_polish_candidates"])
        self.add_argument("--nlp_bab_alm_opt_polish_time", type=float, default=-1.0,
                          help='ALM optimality polish time cap (s) in BaB; negative keeps alm_opt_polish_time.',
                          hierarchy=h + ["bab_alm_opt_polish_time"])
        self.add_argument("--nlp_bab_br_budget", type=int, default=0,
                          help='Shared ALM B*R budget for both root and BaB. Root ALM restarts resolve to floor(budget/domains); BaB chunks domain batches to stay within the same limit. If <= 0, fall back to defaults or root-profile auto-estimation.',
                          hierarchy=h + ["bab_br_budget"])
        self.add_argument("--nlp_bab_cap_batch_by_br_budget", action='store_true', default=False,
                          help='Chunk ALM domain batches so each call respects the shared BaB B*R budget.',
                          hierarchy=h + ["bab_cap_batch_by_br_budget"])
        self.add_argument("--nlp_bab_budget_utilization", type=float, default=0.9,
                          help='Target GPU utilization ratio when estimating BaB B*R budget from root profile.',
                          hierarchy=h + ["bab_budget_utilization"])
        self.add_argument("--nlp_bab_budget_min_scale", type=float, default=1.0,
                          help='Minimum scaling factor from root B*R when estimating BaB budget.',
                          hierarchy=h + ["bab_budget_min_scale"])
        self.add_argument("--nlp_bab_budget_max_scale", type=float, default=4.0,
                          help='Maximum scaling factor from root B*R when estimating BaB budget.',
                          hierarchy=h + ["bab_budget_max_scale"])
        self.add_argument("--nlp_bab_skip_outer_width", type=float, default=0.0,
                          help='Skip ALM outer loop (set outer_max_iter=1) when avg max domain '
                               'width < this threshold. 0 = disabled.',
                          hierarchy=h + ["bab_skip_outer_width"])
        self.add_argument("--nlp_bab_adaptive_interval", action='store_true', default=True,
                          help='Adaptively increase ALM interval when no primal improvement.',
                          hierarchy=h + ["bab_adaptive_interval"])
        self.add_argument("--nlp_bab_max_interval", type=int, default=20,
                          help='Maximum ALM interval when using adaptive interval.',
                          hierarchy=h + ["bab_max_interval"])
        self.add_argument("--nlp_bab_interval_patience", type=int, default=3,
                          help='Consecutive no-improvement ALM calls before doubling interval.',
                          hierarchy=h + ["bab_interval_patience"])
        self.add_argument("--nlp_bab_alm_max_time_share", type=float, default=0.0,
                          help='Cap the in-BaB ALM (with IPOPT-in-BaB) at this fraction of the BaB '
                               'wall time since its last incumbent improvement: a scheduled call is '
                               'skipped while the calls since then used more. 0 disables the cap.',
                          hierarchy=h + ["bab_alm_max_time_share"])
        self.add_argument("--nlp_bab_alm_stop_after_failures", type=int, default=0,
                          help='Skip scheduled in-BaB ALM calls after this many consecutive calls '
                               'without a better incumbent, unless the relative gap (P - D) / '
                               '(1 + |P|) is below bab_alm_stop_gap; an improvement resets the '
                               'count. 0 disables.',
                          hierarchy=h + ["bab_alm_stop_after_failures"])
        self.add_argument("--nlp_bab_alm_stop_gap", type=float, default=0.01,
                          help='Relative gap below which bab_alm_stop_after_failures keeps calling.',
                          hierarchy=h + ["bab_alm_stop_gap"])
        self.add_argument("--nlp_bab_alm_release_cache", action='store_true', default=False,
                          help='Empty the CUDA allocator cache after each in-BaB ALM call, so the '
                               'memory its restarts reserved does not stop AutoBatchSize from '
                               'enlarging the BaB batch.',
                          hierarchy=h + ["bab_alm_release_cache"])
        # IPOPT solver arguments (consolidated under solver.nlp)
        self.add_argument("--nlp_ipopt_enabled", action='store_true',
                          help='Enable IPOPT-based falsification at root node for input split.',
                          hierarchy=h + ["ipopt_enabled"])
        self.add_argument("--nlp_ipopt_max_iter", type=int, default=1000,
                          help='Maximum IPOPT solver iterations.',
                          hierarchy=h + ["ipopt_max_iter"])
        self.add_argument("--nlp_ipopt_tol", type=float, default=1e-6,
                          help='IPOPT solver optimality tolerance.',
                          hierarchy=h + ["ipopt_tol"])
        self.add_argument("--nlp_ipopt_constr_viol_tol", type=float, default=1e-4,
                          help='IPOPT solver constraint violation tolerance for feasibility.',
                          hierarchy=h + ["ipopt_constr_viol_tol"])
        self.add_argument("--nlp_ipopt_timeout", type=float, default=10.0,
                          help='IPOPT solve timeout (seconds).',
                          hierarchy=h + ["ipopt_timeout"])
        self.add_argument("--nlp_ipopt_verbose", action='store_true',
                          help='Print IPOPT solver output.',
                          hierarchy=h + ["ipopt_verbose"])
        self.add_argument("--nlp_vector_lr", action='store_true',
                          help="Use vectorized lr during PGD optimization.",
                          hierarchy=h + ["vector_lr"])
        self.add_argument("--nlp_dual_ascent_enabled", action='store_true', default=False,
                          help='Enable dual ascent to tighten CROWN lower bounds at BnB nodes. '
                               'Requires bounding_method=alpha-crown (needs lA/lbias).',
                          hierarchy=h + ["dual_ascent_enabled"])
        self.add_argument("--nlp_dual_ascent_steps", type=int, default=20,
                          help='Projected gradient ascent steps for lambda optimization in dual ascent.',
                          hierarchy=h + ["dual_ascent_steps"])
        for _key, _help in (
                ("bab_primal_dual_abs_tol", 'Absolute primal-dual gap at which the search counts as closed.'),
                ("bab_primal_dual_rel_tol", 'Relative primal-dual gap, scaled by max(1, abs(primal)), at which the search counts as closed.'),
                ("crown_engine_tiered", 'Let the CROWN engine keep tiered tangent points for the seed engine.'),
                ("enable_ipopt_in_bab", 'Run IPOPT on picked domains inside the BaB loop.'),
                ("ipopt_bab_max_fails", 'Skip IPOPT-in-BaB after this many consecutive calls without a better incumbent.'),
                ("ipopt_bab_timeout", 'Seconds granted to one IPOPT-in-BaB call.'),
                ("ipopt_bab_topk", 'Number of picked domains that IPOPT-in-BaB starts from.'),
                ("qtb_tangent_timeout", 'Seconds granted to the quadratic tangent bound in the BaB loop.')):
            _default = self.NLP_GAP_DEFAULTS[_key]
            self.add_argument("--nlp_" + _key, type=(lambda v: v.lower() in ("1", "true", "yes")) if isinstance(_default, bool) else type(_default),
                              default=_default, help=_help, hierarchy=h + [_key])
        self.add_argument("--nlp_dual_ascent_lr", type=float, default=0.05,
                          help='Adam learning rate for dual ascent lambda optimization.',
                          hierarchy=h + ["dual_ascent_lr"])
        self.add_argument("--nlp_dual_ascent_verbose", action='store_true', default=False,
                          help='Print dual ascent d(lambda) progress per step (debug only).',
                          hierarchy=h + ["dual_ascent_verbose"])
        self.add_argument("--nlp_dual_lp_pdhg_iters", type=int, default=0,
                          help='PDHG iterations for the LP form of the dual-ascent bound '
                               '(objective plane over the box and the node rows); 0 disables.',
                          hierarchy=h + ["dual_lp_pdhg_iters"])
        self.add_argument("--nlp_dual_lp_exact_max", type=int, default=0,
                          help='Re-solve this many lowest-bound domains per batch exactly (scipy '
                               'HiGHS) after PDHG, for LPs with at most 2e5 coefficients.',
                          hierarchy=h + ["dual_lp_exact_max"])
        self.add_argument("--nlp_dual_lp_halpern", action='store_true', default=False,
                          help='Solve the plane LP of each box as a restarted-Halpern lane (stops at KKT '
                               'convergence or the objective threshold; no HiGHS, exact_max ignored).',
                          hierarchy=h + ["dual_lp_halpern"])
        self.add_argument("--nlp_dual_bound_record_all", action='store_true', default=False,
                          help='Record the dual-ascent/LP objective bound on every domain, not only '
                               'on domains it prunes or nearly certifies.',
                          hierarchy=h + ["dual_bound_record_all"])
        self.add_argument("--nlp_quadratic_tangent_bound", action='store_true', default=False,
                          help='When the objective is provably quadratic, also bound it by its '
                               'tangent plane at the incumbent (needs dual_lp_pdhg_iters > 0).',
                          hierarchy=h + ["quadratic_tangent_bound"])
        self.add_argument("--nlp_taylor2_bound", action='store_true', default=False,
                          help='Use interval-Hessian Taylor planes when the objective degree '
                               'proof fails (needs dual_lp_pdhg_iters > 0).',
                          hierarchy=h + ["taylor2_bound"])
        self.add_argument("--nlp_taylor2_max_vars", type=int, default=40,
                          help='Maximum number of inputs for interval-Hessian Taylor planes.',
                          hierarchy=h + ["taylor2_max_vars"])
        self.add_argument("--nlp_taylor2_max_domains", type=int, default=1024,
                          help='Evaluate interval Hessians for this many lowest-bound domains '
                               'per batch; other domains keep their existing bounds.',
                          hierarchy=h + ["taylor2_max_domains"])
        self.add_argument("--nlp_convex_row_cuts", action='store_true', default=False,
                          help='Add global tangent cuts for provably convex quadratic constraint '
                               'rows at the root and on incumbent improvements (up to four rounds).',
                          hierarchy=h + ["convex_row_cuts"])
        self.add_argument("--nlp_convex_tangent_objective", action='store_true', default=False,
                          help='Use global tangent planes for nonquadratic objectives proved '
                               'convex by composition rules on the root box (needs dual_lp_pdhg_iters > 0).',
                          hierarchy=h + ["convex_tangent_objective"])
        self.add_argument("--nlp_convex_tangent_rows", action='store_true', default=False,
                          help='Add global tangent cuts for signed constraint rows proved convex '
                               'by composition rules on the root box (up to four incumbent rounds).',
                          hierarchy=h + ["convex_tangent_rows"])
        self.add_argument("--nlp_qtb_quadratic_dual", action='store_true', default=False,
                          help='Retain positive curvature in the quadratic tangent bound '
                               'and minimize its Lagrangian over the box in closed form.',
                          hierarchy=h + ["qtb_quadratic_dual"])
        self.add_argument("--nlp_qtb_crown_rewrite", action='store_true', default=False,
                          help='Bound a provably quadratic objective with native CROWN on its Hessian eigenbasis '
                               'rewrite at the QTB anchor, per box, instead of the single QTB tangent plane.',
                          hierarchy=h + ["qtb_crown_rewrite"])
        self.add_argument("--nlp_qtb_crown_rewrite_max_vars", type=int, default=2000,
                          help='Largest number of free inputs for which the eigenbasis rewrite is built '
                               '(its Hessian and eigendecomposition are dense); above it QTB\'s plane is used.',
                          hierarchy=h + ["qtb_crown_rewrite_max_vars"])
        self.add_argument("--nlp_qtb_accept_blend", action='store_true', default=False,
                          help='Also offer points on the segment from the QTB relaxed KKT point '
                               'to the incumbent as incumbents (each verified).',
                          hierarchy=h + ["qtb_accept_blend"])
        self.add_argument("--nlp_structure_report", action='store_true', default=True,
                          help='Build and log the shared structure report of the root problem.',
                          hierarchy=h + ["structure_report"])
        self.add_argument("--nlp_gp_certificate", action='store_true', default=False,
                          help='Try a structural geometric-program proof and certified log-space '
                               'outer approximation after the root primal solve.',
                          hierarchy=h + ["gp_certificate"])
        self.add_argument("--nlp_gp_crown", action='store_true', default=False,
                          help='Generate the GP certificate\'s cutting planes with native CROWN on the log-coordinate '
                               'rewrite of its posynomials (exp tangents at the current point).',
                          hierarchy=h + ["gp_crown"])
        self.add_argument("--nlp_gp_certificate_time", type=float, default=20.0,
                          help='CPU time budget in seconds for the root geometric-program certificate.',
                          hierarchy=h + ["gp_certificate_time"])
        self.add_argument("--nlp_cert_crown_dual", action='store_true', default=False,
                          help='The GP, exp and structural root certificates take their LP duals from the restarted '
                               'Halpern kernel (no HiGHS/linprog); the GP epigraph column is capped by the cutoff.',
                          hierarchy=h + ["cert_crown_dual"])
        self.add_argument("--nlp_exp_certificate", action='store_true', default=False,
                          help='Try a structural relative-entropy proof after the substitution '
                               'y = exp(x), with a certified root LP bound.',
                          hierarchy=h + ["exp_certificate"])
        self.add_argument("--nlp_exp_certificate_time", type=float, default=20.0,
                          help='CPU time budget in seconds for the root exp-substitution certificate.',
                          hierarchy=h + ["exp_certificate_time"])
        self.add_argument("--nlp_m3_reduce", action='store_true', default=False,
                          help='M3: when every row is an exact affine, one-product or signed-square row and the '
                               'objective is affine, replace bounding and splitting by outward interval '
                               'propagation of both midpoint children (input_split/m3_reduce.py).',
                          hierarchy=h + ["m3_reduce"])
        for kind, what in (("diameter", "ordered polar polygon area with a complete diameter graph"),
                           ("assignment", "almost-binary cardinality assignment with squared-affine group costs"),
                           ("eigenblock", "normalized symmetric eigenpair rows")):
            self.add_argument(f"--nlp_struct_cert_{kind}", action='store_true', default=False,
                              help=f'Root structure certificate ({what}): a certified objective floor on every '
                                   'box and a constructive point verified on the original model '
                                   '(nlp_solver/structure/certificates.py).',
                              hierarchy=h + [f"struct_cert_{kind}"])
        self.add_argument("--nlp_aux_reference_bounds", action='store_true', default=False,
                          help="Transfer scoped lifted auxiliary bounds to CROWN intermediate nodes.",
                          hierarchy=h + ["aux_reference_bounds"])
        self.add_argument("--nlp_input_cutoff_cuts", action='store_true', default=False,
                          help="Use scoped sign-safe lifted objective cutoff aggregates in input clipping.",
                          hierarchy=h + ["input_cutoff_cuts"])
        self.add_argument("--nlp_lp_certificate_transfer", action='store_true', default=False,
                          help="Reevaluate scoped compressed lifted LP certificates on descendants.",
                          hierarchy=h + ["lp_certificate_transfer"])
        self.add_argument("--nlp_lifted_lp_bound", action='store_true', default=False,
                          help='Also bound the objective of each domain by the lifted McCormick '
                               'LP over the spec rows that are provably quadratic (one variable '
                               'per product, both McCormick planes; nlp_solver/lifted_lp.py).',
                          hierarchy=h + ["lifted_lp_bound"])
        self.add_argument("--nlp_lifted_lp_convex_objective", action='store_true', default=False,
                          help='In float64, lift structurally proved convex functions of quadratic terms '
                               'in the objective using epigraphs and adaptive tangent cuts.',
                          hierarchy=h + ["lifted_lp_convex_objective"])
        self.add_argument("--nlp_clear_denominators", action='store_true', default=False,
                          help='Add quadratic lifted LP rows by clearing monomial denominators '
                               'whose variables have strictly positive root lower bounds.',
                          hierarchy=h + ["clear_denominators"])
        self.add_argument("--nlp_lifted_lp_obbt", action='store_true', default=False,
                          help='Tighten the root box by OBBT over the lifted McCormick LP '
                               '(objective cut off at the incumbent) before the first split.',
                          hierarchy=h + ["lifted_lp_obbt"])
        self.add_argument("--no_nlp_lifted_lp_node_cutoff", action='store_false', default=True,
                          dest="nlp_lifted_lp_node_cutoff",
                          help='Include the incumbent cutoff row in spatial node LPs and FBBT.',
                          hierarchy=h + ["lifted_lp_node_cutoff"])
        self.add_argument("--nlp_lifted_lp_node_obbt_gain", action='store_true', default=False,
                          help='Admit node OBBT by measured certified bound gain per paid second '
                               'relative to ordinary node solves.',
                          hierarchy=h + ["lifted_lp_node_obbt_gain"])
        self.add_argument("--nlp_lifted_lp_rlt", action='store_true', default=False,
                          help='Multiply linear equalities by product variables in the lifted LP, '
                               'with relaxation derived from both tolerance-relaxed source rows.',
                          hierarchy=h + ["lifted_lp_rlt"])
        self.add_argument("--nlp_lifted_lp_rlt_full_budget", action='store_true', default=False,
                          help='Charge RLT rows, columns and structural nonzeros, including product '
                               'envelopes, to the existing RLT row budget.',
                          hierarchy=h + ["lifted_lp_rlt_full_budget"])
        self.add_argument("--nlp_lifted_lp_rlt_max_rows", type=int, default=1200,
                          help='RLT rows (two per equality product) allowed, after vertex-cover '
                               'multipliers; more needed means no RLT rows at all. Zero disables RLT.',
                          hierarchy=h + ["lifted_lp_rlt_max_rows"])
        self.add_argument("--nlp_lifted_lp_rlt_max_columns", type=int, default=7000,
                          help='No RLT rows when the lifted LP would exceed this many columns '
                               '(inputs plus products, including RLT products).',
                          hierarchy=h + ["lifted_lp_rlt_max_columns"])
        self.add_argument("--nlp_lifted_lp_branching", action='store_true', default=False,
                          help='SB branching takes the variable with the most violated '
                               'McCormick products at the lifted LP point of the domain.',
                          hierarchy=h + ["lifted_lp_branching"])
        self.add_argument("--nlp_lifted_lp_split_at_point", action='store_true', default=False,
                          help='With lifted_lp_branching, split at the LP point (clamped to the '
                               'middle 80%%) instead of the midpoint (reordered BaB).',
                          hierarchy=h + ["lifted_lp_split_at_point"])
        self.add_argument("--nlp_lifted_lp_bab_time", type=float, default=0.0,
                          help='Root-stage budget (s) of a B&B over the lifted LP alone, run when '
                               'every constraint row is lifted and the objective is quadratic; '
                               'it certifies the incumbent when it closes the gap. 0 disables.',
                          hierarchy=h + ["lifted_lp_bab_time"])
        self.add_argument("--nlp_lifted_lp_bab_handoff", action='store_true', default=False,
                          help='When the root lifted-LP B&B does not close, start BaB from its '
                               'open leaves (with their bounds) instead of the root box.',
                          hierarchy=h + ["lifted_lp_bab_handoff"])
        self.add_argument("--nlp_lifted_lp_leaf_ipopt", action='store_true', default=False,
                          help='When the lifted-LP B&B stays open, start the root local solver '
                               'from the LP points of its three lowest-bound leaves.',
                          hierarchy=h + ["lifted_lp_leaf_ipopt"])
        self.add_argument("--nlp_lifted_lp_leaf_ipopt_time", type=float, default=5.0,
                          help='Timeout (s) of each such local solve.',
                          hierarchy=h + ["lifted_lp_leaf_ipopt_time"])
        self.add_argument("--nlp_lifted_lp_bab_workers", type=int, default=1,
                          help='CPU worker processes of the lifted-LP B&B (forked; each expands '
                               'one of the best nodes per step). 1 = serial.',
                          hierarchy=h + ["lifted_lp_bab_workers"])
        self.add_argument("--nlp_lifted_lp_bab_max_nodes", type=int, default=200000,
                          help='Node cap of the lifted-LP B&B.',
                          hierarchy=h + ["lifted_lp_bab_max_nodes"])
        self.add_argument("--nlp_lifted_lp_obbt_rounds", type=int, default=5,
                          help='Maximum OBBT rounds over the product variables.',
                          hierarchy=h + ["lifted_lp_obbt_rounds"])
        self.add_argument("--nlp_lifted_lp_obbt_time", type=float, default=10.0,
                          help='Wall-time budget (s) of the root OBBT.',
                          hierarchy=h + ["lifted_lp_obbt_time"])
        self.add_argument("--nlp_lifted_lp_exact_max", type=int, default=16,
                          help='Domains per batch whose lifted LP is re-solved exactly (HiGHS).',
                          hierarchy=h + ["lifted_lp_exact_max"])
        self.add_argument("--nlp_lifted_lp_exact_time", type=float, default=0.5,
                          help='Wall-time cap (s) of the exact lifted-LP re-solves per bounding '
                               'call; the remaining domains keep their PDHG bound.',
                          hierarchy=h + ["lifted_lp_exact_time"])
        self.add_argument("--nlp_lifted_lp_pdhg_iters", type=int, default=500,
                          help='PDHG iterations of the batched lifted LP (0: HiGHS only).',
                          hierarchy=h + ["lifted_lp_pdhg_iters"])
        self.add_argument("--nlp_lifted_lp_max_elems", type=int, default=25_000_000,
                          help='Dense coefficients of the batched lifted LP; the lowest-bound '
                               'domains that fit are bounded by PDHG.',
                          hierarchy=h + ["lifted_lp_max_elems"])
        self.add_argument("--nlp_lifted_lp_max_vars", type=int, default=2000,
                          help='No lifted LP above this many input variables.',
                          hierarchy=h + ["lifted_lp_max_vars"])
        self.add_argument("--nlp_ipopt_constraint_slack_frac", type=float, default=0.0,
                          help='IPOPT aims at g <= frac * constr_viol_tol instead of g <= 0, '
                               'using the feasibility tolerance the dual cuts already allow.',
                          hierarchy=h + ["ipopt_constraint_slack_frac"])
        self.add_argument("--nlp_sparse_derivatives", action='store_true', default=False,
                          help='IPOPT callbacks from the exact forms of the structure walk: sparse '
                               'Jacobians, the exact Hessian of the Lagrangian, and rows merged by '
                               'exact direction (an equality is one ranged row).',
                          hierarchy=h + ["sparse_derivatives"])
        self.add_argument("--nlp_lp_alternation", action='store_true', default=False,
                          help='Root primal on bilinear structure: block alternation of exact LPs '
                               'with an elastic phase 1, joint trust-region LP steps, then the '
                               'sparse exact-Hessian IPOPT; only verified points become incumbents.',
                          hierarchy=h + ["lp_alternation"])
        self.add_argument("--nlp_root_lp_local_start", action='store_true', default=False,
                          help='Also start the root local solver (IPOPT) from the minimizer of '
                               'the root objective plane over the box and the root rows.',
                          hierarchy=h + ["root_lp_local_start"])
        self.add_argument("--nlp_root_primal_first_order", action='store_true', default=False,
                          help='Start the root local solver from the first-order (PDHG, on the device) primal of the root '
                               'plane LP instead of a HiGHS solve. Does not touch lp_alternation.',
                          hierarchy=h + ["root_primal_first_order"])
        self.add_argument("--nlp_concretize_max_constraint_rows", type=int, default=0,
                          help='At most this many binding constraint rows per domain enter the constrained-concretization dual sweep (the most binding ones); a subset only weakens the bound. 0 keeps all.',
                          hierarchy=h + ["concretize_max_constraint_rows"])
        self.add_argument("--nlp_concretize_fused_kernel", action='store_true', default=False,
                          help='Run the constrained-concretization dual sweep as one fused Triton kernel (CUDA, float64, no gradient); same multipliers and bounds up to rounding.',
                          hierarchy=h + ["concretize_fused_kernel"])
        self.add_argument("--nlp_alm_root_mu_warmstart", action='store_true', default=False,
                          help='Run dual ascent at the root node and use lambda* as mu_init for '
                               'root ALM. Requires dual_ascent_enabled and alpha-crown / crown lA.',
                          hierarchy=h + ["alm_root_mu_warmstart"])
        self.add_argument("--no_nlp_dual_to_alm_mu_warmstart",
                          dest='dual_to_alm_mu_warmstart', action='store_false', default=True,
                          help='Disable warm-starting ALM mu from dual-ascent lambda* in BaB.',
                          hierarchy=h + ["dual_to_alm_mu_warmstart"])
        self.add_argument("--nlp_dual_to_alm_mu_warmstart",
                          dest='dual_to_alm_mu_warmstart', action='store_true',
                          help='Enable warm-starting ALM mu from dual-ascent lambda* in BaB.',
                          hierarchy=h + ["dual_to_alm_mu_warmstart"], private=True)
        self.add_argument("--nlp_alm_dual_cuts_enabled", action='store_true', default=True,
                          help='Append a small set of useful dual-derived linear cuts as temporary ALM input constraints during BaB.',
                          hierarchy=h + ["alm_dual_cuts_enabled"])
        self.add_argument("--nlp_alm_dual_cut_topk", type=int, default=10,
                          help='Maximum number of useful dual-derived cuts kept per domain for temporary ALM use. 0 disables cut passing.',
                          hierarchy=h + ["alm_dual_cut_topk"])
        self.add_argument("--nlp_alm_dual_cut_inactive_tol", type=float, default=1e-6,
                          help='Drop a candidate dual cut when its max slack on the evaluation domain is <= -tol.',
                          hierarchy=h + ["alm_dual_cut_inactive_tol"])
        self.add_argument("--nlp_alm_dual_cut_eval_domain", type=str, default='complete',
                          choices=['complete', 'base', 'relaxed', 'box'],
                          help='Domain used to judge whether a candidate dual cut is still active: '
                               '"relaxed"/"box" uses only the current box, while '
                               '"complete"/"base" uses the current box intersected with the immutable base input constraints. '
                               'We intentionally do not evaluate against the full domain-constraint tuple because it contains the candidate row itself.',
                          hierarchy=h + ["alm_dual_cut_eval_domain"])
        self.add_argument("--nlp_alm_dual_cut_score", type=str, default='ub',
                          choices=['ub', 'lambda_ub'],
                          help='Ranking heuristic for useful dual cuts before top-k selection.',
                          hierarchy=h + ["alm_dual_cut_score"])
        self.add_argument("--nlp_alm_dual_cut_lambda_floor", type=float, default=0.0,
                          help='Lower clamp for lambda weights when --nlp_alm_dual_cut_score=lambda_ub.',
                          hierarchy=h + ["alm_dual_cut_lambda_floor"])
        self.add_argument("--nlp_dual_ascent_lam_warmstart", action='store_true',
                          help='Warm-start dual ascent lambda from ALM mu, and vice versa.',
                          hierarchy=h + ["dual_ascent_lam_warmstart"])
        h = ["solver", "crown"]
        self.add_argument('--crown_batch_size', type=int, default=int(1e9),
                          help='Batch size in batched CROWN.',
                          hierarchy=h + ["batch_size"])
        self.add_argument('--crown_spec_batch_size', type=int, default=int(1e9),
                          help='Maximum number of final specification rows per CROWN compute_bounds call. '
                               'Default 1e9 = no batching (zero regression); set a small value (e.g. via '
                               'solver.crown.spec_batch_size in YAML) only for instances whose initial CROWN '
                               'OOMs on a huge number of specs (e.g. allen_cahn_g1 with 15367 specs).',
                          hierarchy=h + ["spec_batch_size"])
        self.add_argument('--max_crown_size', type=int, default=int(1e9),
                          help='Max output size in CROWN (when there are too many output neurons, only part of them will be bounded by CROWN).',
                          hierarchy=h + ["max_crown_size"])

        h = ["solver", "multi_facet"]
        self.add_argument('--multi_facet_enabled', action='store_true', dest='multi_facet_enabled',
                          help='Enable multi-facet relaxation cuts: use the convex-hull facets that a '
                               'single CROWN plane discards (see auto_LiRPA/multi_facet.py). Default off '
                               '(byte-identical default path, zero regression).',
                          hierarchy=h + ["enabled"])
        # Restore the crown hierarchy: the two arguments below belong to solver.crown,
        # and beta_CROWN_solver.py reads solver_args['crown']['compare_crown_with_ibp'].
        h = ["solver", "crown"]
        self.add_argument("--activation_bound_option", default='adaptive',
                          choices=["adaptive", "same-slope", "zero-lb", "one-lb"],
                          help='Options for specifying the way to initialize CROWN bounds for activations.',
                          hierarchy=h + ["activation_bound_option"])
        self.add_argument("--compare_crown_with_ibp", action="store_true",
                          help="Compare CROWN bounds with IBP bounds given existing intermediate bounds.",
                          hierarchy=h + ["compare_crown_with_ibp"])

        h = ["solver", "alpha-crown"]
        self.add_argument('--no_alpha', action='store_false', dest='alpha',
                          help='Disable/Enable alpha crown.',
                          hierarchy=h + ["alpha"])
        self.add_argument("--lr_init_alpha", type=float, default=0.1,
                          help='Learning rate for the optimizable parameter alpha in alpha-CROWN bound.',
                          hierarchy=h + ["lr_alpha"])
        self.add_argument('--init_iteration', type=int, default=100,
                          help='Number of iterations for alpha-CROWN incomplete verifier.',
                          hierarchy=h + ["iteration"])
        self.add_argument("--share_alphas", action='store_true',
                          help='Share some alpha variables to save memory at the cost of slightly looser bounds.',
                          hierarchy=h + ["share_alphas"])
        self.add_argument("--alpha_lr_decay", type=float, default=0.98,
                          help='Learning rate decay factor during alpha-CROWN optimization. Need to use a larger value like 0.99 or 0.995 when you increase the number of iterations.',
                          hierarchy=h + ["lr_decay"])
        self.add_argument("--no_full_conv_alpha", action='store_false', dest='full_conv_alpha',
                          help='Enable/disable the use of independent alpha for conv layers.',
                          hierarchy=h + ["full_conv_alpha"])
        self.add_argument("--max_coeff_mul", type=float, default=torch.inf,
                          help='Maximum coefficient value in the optimizable parameters for BoundMul.',
                          hierarchy=h + ["max_coeff_mul"])
        self.add_argument("--matmul_share_alphas", action='store_true',
                          help='Check alpha sharing for matmul.',
                          hierarchy=h + ['matmul_share_alphas'])
        self.add_argument("--disable_optimization", type=str, nargs='+', default=[],
                          help='A list of the names of operators which have bound optimization disabled.',
                          hierarchy=h + ['disable_optimization'])
        self.add_argument('--init_max_time', type=float, default=1.0,
                           help='Maximum time for the initial bound optimization (relative to the total timeout).',
                           hierarchy=h + ['max_time'])
        self.add_argument('--alpha_dtype', type=str, default='float16',
                          choices = ['float16', 'auto'],
                          help='The dtype of alpha stored in branch-and-bound '
                               'domains. Alpha values may take a lot of space, so '
                               'float16 is used by default. Setting this to '
                               '"auto" preserves the graph dtype (useful for '
                               'debugging). solver.reproduce_alpha_bounds also '
                               'preserves the source dtype and overrides this '
                               'setting.',
                          hierarchy=h + ['alpha_dtype'])
        self.add_argument('--input_split_alpha_iteration', type=int, default=5,
                          help='Number of alpha-CROWN iterations during input-split based branch and bound.',
                          hierarchy=h + ['input_split_alpha_iteration'])
        self.add_argument('--input_split_lr_alpha', type=float, default=0.05,
                          help='Learning rate for alpha-CROWN during input-split based branch and bound.',
                          hierarchy=h + ['input_split_lr_alpha'])
        self.add_argument('--input_split_objective_iteration', type=int, default=0,
                          help='Solving mode, input split: after the CROWN pass, re-bound only the objective '
                               'row with this many alpha-CROWN iterations (intermediate bounds fixed). 0 disables it.',
                          hierarchy=h + ['input_split_objective_iteration'])

        h = ["solver", "invprop"]
        self.add_argument("--apply_output_constraints_to", type=str, nargs='*', default=[],
                          help='Includes the output constraint in the optimization of linear layers. Can be a comma separated list of layer types (e.g. "BoundLinear"), layer names (e.g. "/input.7") or "all". This will disable patch mode for listed conv layers. When set, --optimize_disjuncts_separately must be set, too, if the safety property uses a disjunction.',
                          hierarchy=h + ['apply_output_constraints_to'])
        self.add_argument("--tighten_input_bounds", action='store_true',
                          help='Tighten input bounds using output constraints. If set, --apply_output_constraints_to should contain "BoundInput" or the corresponding layer name.',
                          hierarchy=h + ['tighten_input_bounds'])
        self.add_argument("--best_of_oc_and_no_oc", action='store_true',
                          help='Computes bounds of each layer both with and without output constraints. The best of both is used. Increases the runtime, but may improve the bounds.',
                          hierarchy=h + ['best_of_oc_and_no_oc'])
        self.add_argument("--directly_optimize", type=str, nargs='*', default=[],
                          help='A list of layer names whose bounds should be directly optimized for. IBP is disabled for these layers. Should only be needed for backward verification (approximation of input bounds) where the first linear layer defines the cs and should be optimized.',
                          hierarchy = h + ['directly_optimize'])
        self.add_argument("--oc_lr", type=float, default=0.1,
                          help='The learning rate for the dualized output constraints.',
                          hierarchy = h + ['oc_lr'])
        self.add_argument("--share_gammas", action='store_true',
                          help='Shares gammas across neurons in the optimized layer.',
                          hierarchy = h + ['share_gammas'])

        h = ["solver", "beta-crown"]
        self.add_argument("--lr_alpha", type=float, default=0.01,
                          help='Learning rate for optimizing alpha during branch and bound.',
                          hierarchy=h + ["lr_alpha"])
        self.add_argument("--lr_beta", type=float, default=0.05,
                          help='Learning rate for optimizing beta during branch and bound.',
                          hierarchy=h + ["lr_beta"])
        self.add_argument("--lr_decay", type=float, default=0.98,
                          help='Learning rate decay factor during beta-CROWN optimization. Need to use a larger value like 0.99 or 0.995 when you increase the number of iterations.',
                          hierarchy=h + ["lr_decay"])
        self.add_argument("--optimizer", default="adam", help='Optimizer used for alpha and beta optimization.',
                          hierarchy=h + ["optimizer"])
        self.add_argument("--iteration", type=int, default=50,
                          help='Number of iteration for optimizing alpha and beta during branch and bound.',
                          hierarchy=h + ["iteration"])
        self.add_argument('--no_beta', action='store_false', dest='beta',
                          help='Enable/disable beta split constraint (this option is for ablation study only and should not be used normally).',
                          hierarchy=h + ["beta"])
        self.add_argument('--no_beta_warmup', action='store_false', dest='beta_warmup',
                          help='Enable/disable beta warmup from branching history (this option is for ablation study only and should not be used normally).',
                          hierarchy=h + ["beta_warmup"])
        self.add_argument('--enable_opt_interm_bounds', action='store_true', default=False,
                          help='Enable optimizing intermediate bounds for beta-CROWN, only used when mip refine for now.',
                          hierarchy=h + ["enable_opt_interm_bounds"])
        self.add_argument('--enable_all_node_split_LP', action='store_true',
                          dest="all_node_split_LP", default=False,
                          help='When all nodes are split during Bab but not verified, using LP to check.',
                          hierarchy=h + ["all_node_split_LP"])

        h = ["solver", "forward"]
        self.add_argument('--forward_refine', action='store_true',
                          help='Refine forward bound with CROWN for unstable neurons.',
                          hierarchy=h + ["refine"])
        self.add_argument('--forward_max_dim', type=int, default=10000,
                          help='Maximum input dimension for forward bounds in a batch.',
                          hierarchy=h + ["max_dim"])
        self.add_argument('--reset_start_layer_threshold', type=float, default=1.0,
                          help='Reset the start layer if timeout neurons are above the threshold.',
                          hierarchy=h + ["reset_threshold"])

        ### preprocessor-hint: private-section-start
        h = ["solver", "intermediate_refinement"]
        self.add_argument('--opt_intermediate_beta', action='store_true', dest='opt_intermediate_beta',
                          help='Optimize constraint bias in compute bounds.',
                          hierarchy=h + ["enabled"], private=True)
        self.add_argument("--refinement_batch_size", type=int, default=10,
                          help='Batch size used for intermediate beta refinement.',
                          hierarchy=h + ["batch_size"],
                          private=True)
        self.add_argument('--opt_coeffs', action='store_true', dest='opt_coeffs',
                          help='Optimize coeffs during intermediate_refinement.',
                          hierarchy=h + ["opt_coeffs"], private=True)
        self.add_argument('--opt_bias', action='store_true', dest='opt_bias',
                          help='Optimize constraint bias during intermediate_refinement.',
                          hierarchy=h + ["opt_bias"], private=True)
        self.add_argument("--lr_intermediate_beta", type=float, default=0.05,
                          help='Learning rate for intermediate layer beta for refinement.',
                          hierarchy=h + ["lr"],
                          private=True)
        self.add_argument('--intermediate_refinement_layers', nargs='+', type=int, default=[-1],
                          help='Layers to be refined, separated by commas. -1 means preactivation before last relu.',
                          hierarchy=h + ["layers"], private=True)
        self.add_argument('--max_refinement_domains', type=int, default=1000,
                          help='The max length of domains that we can use the opt_intermediate_beta.',
                          hierarchy=h + ["max_domains"], private=True)
        ### preprocessor-hint: private-section-end

        h = ["solver", "mip"]
        self.add_argument('--mip_unstable_neuron_threshold', type=int, default=0,
                          help='When complete_verifier=auto, enable MIP refinement when the number of unstable neurons exceeds this threshold.',
                          hierarchy=h + ["unstable_neuron_threshold"])
        self.add_argument('--mip_multi_proc', type=int, default=None,
                          help='Number of multi-processes for mip solver. Each process computes a mip bound for an intermediate neuron. Default (None) is to auto detect the number of CPU cores (note that each process may use multiple threads, see the next option).',
                          hierarchy=h + ["parallel_solvers"])
        self.add_argument('--mip_threads', type=int, default=1,
                          help='Number of threads for echo mip solver process (default is to use 1 thread for each solver process).',
                          hierarchy=h + ["solver_threads"])
        self.add_argument('--mip_perneuron_refine_timeout', type=float, default=15,
                          help='MIP timeout threshold for improving each intermediate layer bound (in seconds).',
                          hierarchy=h + ["refine_neuron_timeout"])
        self.add_argument('--mip_refine_timeout', type=float, default=0.8,
                          help='Percentage (x100%%) of time used for improving all intermediate layer bounds using mip. Default to be 0.8*timeout.',
                          hierarchy=h + ["refine_neuron_time_percentage"])
        self.add_argument('--no_mip_early_stop', action='store_false',
                          dest='mip_early_stop',
                          help='Enable/disable early stop when finding a positive lower bound or a adversarial example during MIP.',
                          hierarchy=h + ["early_stop"])
        self.add_argument('--disable_adv_warmup', action='store_false', dest='adv_warmup',
                          help='Disable using PGD adv as MIP refinement warmup starts.',
                          hierarchy=h + ["adv_warmup"])
        self.add_argument('--mip_solver', type=str, default='gurobi', help="MLP/LP solver package (SCIP support is experimental).",
                          choices=["gurobi", "scip"], hierarchy=h + ["mip_solver"])
        self.add_argument('--mip_skip_unsafe', action='store_true', dest='unsafe',
                          help='Do not use MIP solver to find counterexamples. This is useful in the case the MIP formulation is a relaxation and not complete.',
                          hierarchy=h + ["skip_unsafe"])
        self.add_argument('--topk_percentage', type=float, default=1., help="Refine top k neurons in mip solver.",
                          hierarchy=h + ["topk_filter"])
        self.add_argument('--mip_refine_sliding_window', type=int, default=-1,
                          help="Only when the neural network is very deep or too many timeout neurons that we use it to simplify the problem.",
                          hierarchy=h + ["sliding_window"])
        self.add_argument('--mip_refine_extra_input_constraints', action='store_true', dest='extra_constraint',
                          help="Add extra contraints to the MIP solver between the input and the starting layer, only activate when sliding window is enabled.",
                          hierarchy=h + ["extra_constraint"])
        self.add_argument('--mip_perneuron_refine_timeout_increasement', type=int, default=0,
                          help='MIP timeout threshold increasment for improving each intermediate layer bound (in seconds) if the remaining time is sufficient.',
                          hierarchy=h + ["refine_neuron_timeout_increasement"])
        self.add_argument('--mip_timeout_neuron', type=float, default=0.3,
                          help='Threshold for the timeout neurons to reset the perneuron timeout.',
                          hierarchy=h + ["timeout_neuron_percentage"])
        self.add_argument('--mip_remaining_timeout_coefficient_for_increasement', type=float, default=1,
                           help='MIP remaining timeout multiplier to determine whether following layer should increase the perneuron timeout.',
                           hierarchy=h + ["remaining_timeout_threshold"])
        self.add_argument('--mip_refine_remove_neurons', action='store_true', dest='remove_unstable_neurons',
                           help='Remove unstable neurons from MIP based on A_dict from compute_bounds.',
                           hierarchy=h + ['remove_unstable_neurons'])
        self.add_argument('--mip_formulation', default="mip", choices= ["lp", "lp_integer", "mip"],
                          help=(
                              "Create (mixed integer) linear programming model. "
                              "lp: linear programming model without planet relaxation; "
                              "lp_integer: linear programming model with relaxed integer variables z; "
                              "mip: mixed integer programming model with binary variables z."
                          ),
                          hierarchy=h + ["formulation"])
        self.add_argument('--mip_add_output_cut', action='store_true',
                          help="Get the output <= 0 as a cut from the last layer.",
                          hierarchy=h + ["output_cut"])

        h = ["bab"]
        self.add_argument("--max_domains", type=int, default=float("inf"),
                          help='Max number of subproblems in branch and bound.',
                          hierarchy=h + ["max_domains"])
        self.add_argument("--decision_thresh", type=float, default=0,
                          help='Decision threshold of lower bounds. When lower bounds are greater than this value, verification is successful. Set to 0 for robustness verification.',
                          hierarchy=h + ["decision_thresh"])
        self.add_argument("--timeout", type=float, default=360,
                          help='Timeout (in second) for verifying one image/property.',
                          hierarchy=h + ["timeout"])
        self.add_argument("--timeout_scale", type=float, default=1, help='Scale the timeout for development purpose.',
                          hierarchy=h + ["timeout_scale"])
        self.add_argument("--max_iterations", type=int, default=-1,
                          help='Maximum number of BaB iterations.',
                          hierarchy=h + ["max_iterations"])
        self.add_argument("--override_timeout", type=float, default=None,
                          help='Override timeout.',
                          hierarchy=h + ["override_timeout"])
        self.add_argument("--solve_deadline", type=float, default=None,
                          help='Wall-clock time (time.time()) by which the solve ends; set only by the GEAR facade '
                               'under timing="solve" (G5 DESIGN section 1 Timing). None keeps today\'s clocks.',
                          hierarchy=h + ["solve_deadline"])
        self.add_argument("--get_upper_bound", action='store_true',
                          help='Update global upper bound during BaB (has extra overhead, typically the upper bound is not used).',
                          hierarchy=h + ["get_upper_bound"])
        self.add_argument("--disable_pruning_in_iteration", action='store_false',
                          dest='pruning_in_iteration',
                          help='Disable verified domain pruning within iteration.',
                          hierarchy=h + ["pruning_in_iteration"])
        self.add_argument("--pruning_in_iteration_ratio", type=float, default=0.2,
                          help='When ratio of positive domains >= this ratio, pruning in iteration optimization is open.',
                          hierarchy=h + ["pruning_in_iteration_ratio"])
        self.add_argument('--sort_targets', action='store_true',
                          help='Sort targets before BaB.',
                          hierarchy=h + ["sort_targets"])
        self.add_argument("--disable_batched_domain_list", action='store_false',
                          dest='batched_domain_list',
                          help='Disable batched domain list. Batched domain list is faster but picks domain to split in an unsorted way.',
                          hierarchy=h + ['batched_domain_list'])
        self.add_argument("--optimized_interm", type=str, default="",
                          help='A list of layer names that will be optimized during branch and bound, separated by comma.',
                          hierarchy=h + ['optimized_interm'])
        self.add_argument("--no_interm_transfer", action='store_false',
                          dest='interm_transfer',
                          help='Skip the intermediate bound transfer to save transfer-to-CPU time. Require intermediate bound does not change. Caution: cannot be used with cplex cut or intermediate beta refinement.',
                          hierarchy=h + ['interm_transfer'])
        self.add_argument("--recompute_interm", action="store_true",
                          dest="recompute_interm",
                          help="Recompute all the intermediate bounds during BaB.",
                          hierarchy=h + ['recompute_interm'])
        self.add_argument("--sort_domain_interval", type=int, default=-1,
                          help='If unsorted domains are used, sort the domains every sort_domain_interval iterations.',
                          hierarchy=h + ["sort_domain_interval"])
        self.add_argument("--lazy_sort_domains", action="store_true", default=False,
                          help='Input BaB: a scheduled domain sort only records the sort; pick_out_batch '
                               'then selects and orders just the rows it pops. The same domains are '
                               'popped as after a full sort (up to ties), without permuting every storage.',
                          hierarchy=h + ["lazy_sort_domains"])
        self.add_argument("--domain_storage_gpu_mb", type=int, default=0,
                          help='Input BaB: keep the per-domain storages (bounds, boxes, cut rows) on the '
                               'GPU while they take less than this many MB, instead of host memory; past '
                               'it they move to the host for good. 0 keeps them on the host.',
                          hierarchy=h + ["domain_storage_gpu_mb"])
        self.add_argument("--tree_traversal", default="depth_first",
                          choices=["depth_first", "breadth_first"],
                          help='During BaB, unknown domains can continue being split (deepening the tree, i.e. depth first traversal) or split only when all other unknown domains have the same number of splits (keeping the tree shallow for as long as possible, i.e. breadth first traversal). Depth first traversal minimizes memory access time, breadth first traversal is beneficial for BICCOS.',
                          hierarchy=h + ["tree_traversal"])
        self.add_argument("--disable_hugetensor_allocator", action="store_false",
                          help='Use the default tensor allocator instead of the HugeTensor allocator.',
                          hierarchy=h + ["hugetensor_allocator"])

        h = ["bab", "mixed_precision"]
        self.add_argument("--no_tf32_branching_decision", action="store_false",
                          dest="enable_tf32_branching_decision",
                          help='Whether to use TF32 precision during branching decision computation.',
                          hierarchy=h + ["enable_tf32_branching_decision"])
        self.add_argument("--no_tf32_clip_decision", action="store_false",
                          dest="enable_tf32_clip_decision",
                          help='Whether to use TF32 precision during clip decision computation.',
                          hierarchy=h + ["enable_tf32_clip_decision"])
        self.add_argument("--check_mixed_decision_bound_soundness", action="store_true",
                          dest="check_mixed_decision_bound_soundness",
                          help='Recompute bounds in full precision after mixed-precision BaB bound and check for unsoundness, for debug.',
                          hierarchy=h + ["check_mixed_decision_bound_soundness"])
        self.add_argument("--enable_bab_bound_mixed_precision", action="store_true",
                          dest="enable_bab_bound_mixed_precision",
                          help='Enable mixed-precision bound computation during BaB. A fp32 bounds is computed for soundness.',
                          hierarchy=h + ["enable_bab_bound_mixed_precision"])
        self.add_argument("--bab_bound_mixed_precision_dtype", type=str, default="tf32",
                          choices=["fp16", "bf16", "tf32", "fp16+tf32", "bf16+tf32", "fp32", "none", ""],
                          help='Mixed Precision dtype for bound CROWN. A fp32 bounds is computed for soundness. Options: fp16, bf16, tf32, fp16+tf32, bf16+tf32, fp32, none, "".',
                          hierarchy=h + ["bab_bound_mixed_precision_dtype"])
        self.add_argument("--disable_prune_even_under_low_precision", action="store_false",
                          dest="prune_even_under_low_precision",
                          help='Whether to speculatively prune verified domains to cut workload even under the low-precision iterations of alpha-CROWN using the low-precision. Meaningful only if prune_in_iteration is enabled.',
                          hierarchy=h + ["prune_even_under_low_precision"])
        self.add_argument("--low_precision_prune_margin_abs", type=float, default=0.01,
                          help='Additional absolute margin for speculative pruning to decide verified. The default suits TF32 bound error.',
                          hierarchy=h + ["low_precision_prune_margin_abs"])

        # FIXME: cut should not be under bab. We don't have to use bab with cuts. It should be under "solver" category.
        h = ["bab", "cut"]
        self.add_argument('--enable_cut', action='store_true', dest='enable_cut',
                          help='Enable cutting planes using GCP-CROWN.',
                          hierarchy=h + ["enabled"])
        self.add_argument('--cuts_path', default=None, type=str,
                          help='For cuts from CPLEX, specify the path for saving intermediate files with generated cuts.',
                          hierarchy=h + ["cuts_path"])
        self.add_argument('--enable_bab_cut', action='store_true', dest='enable_bab_cut',
                          help='Enable cut constraints optimization during BaB.',
                          hierarchy=h + ["bab_cut"])
        self.add_argument('--enable_lp_cut', action='store_true', dest='enable_lp_cut',
                          help='enable lp with cut constraints to debug',
                          hierarchy=h + ["lp_cut"], private=True)
        self.add_argument("--lr_cuts", type=float, default=0.01, help='Learning rate for optimizing cuts.',
                          hierarchy=h + ["lr"])
        self.add_argument("--cut_lr_decay", type=float, default=1.0,
                          help='Learning rate decay for optimizing betas in GCP-CROWN.',
                          hierarchy=h + ["lr_decay"])
        self.add_argument("--cut_iteration", type=int, default=100,
                          help='Iterations for optimizing betas in GCP-CROWN.',
                          hierarchy=h + ["iteration"])
        self.add_argument("--cut_bab_iteration", type=int, default=-1,
                          help='Iterations for optimizing betas in GCP-CROWN during branch and bound. Set to -1 to use the same number of iterations without cuts.',
                          hierarchy=h + ["bab_iteration"])
        self.add_argument("--cut_early_stop_patience", type=int, default=-1,
                          help='Early stop patience for optimizing cuts. Set to -1 to use the same value when cuts are not used.',
                          hierarchy=h + ["early_stop_patience"])
        self.add_argument("--cut_lr_beta", type=float, default=0.02,
                          help='Learning rate for optimizing betas in GCP-CROWN.',
                          hierarchy=h + ["lr_beta"])
        self.add_argument("--number_cuts", type=int, default=50,
                          help='Maximum number of cuts that we want to add.',
                          hierarchy=h + ["number_cuts"])
        self.add_argument("--topk_implication", type=int, default=1000,
                          help='Only keep top K constraints when filtering cuts.',
                          hierarchy=h + ["topk_cuts_in_filter"])
        self.add_argument("--batch_size_primal", type=int, default=100,
                          help='Batch size when calculate primals, should be negative correlated to number of unstable neurons.',
                          hierarchy=h + ["batch_size_primal"])
        self.add_argument("--cut_max_num", type=int, default=int(1e9),
                          help='Maximum number of cuts.',
                          hierarchy=h + ["max_num"])
        self.add_argument("--enable_patches_cut", action='store_true',
                          help='Enable GCP-CROWN optimization for intermediate layer bounds in patches mode.',
                          hierarchy=h + ["patches_cut"])
        self.add_argument("--cplex_cuts", action='store_true',
                          help='Build and save mip mps models, let cplex find cuts, and use found cuts to improve lower bounds.',
                          hierarchy=h + ["cplex_cuts"])
        self.add_argument("--cplex_cuts_wait", type=float, default=0,
                          help='Wait a bit after cplex warmup in seconds, so that we tend to get some cuts at early stage of branch and bound.',
                          hierarchy=h + ["cplex_cuts_wait"])
        self.add_argument("--no_cplex_cuts_revpickup", action='store_false',
                          help='Enable/disable the inverse order domain pickout when cplex is enabled.',
                          hierarchy=h + ["cplex_cuts_revpickup"])
        self.add_argument("--no_cut_reference_bounds", action='store_false',
                          help='Enable/disable using reference bounds when GCP-CROWN cuts are used.',
                          hierarchy=h + ["cut_reference_bounds"])
        self.add_argument("--fix_cut_intermediate_bounds", action='store_true',
                          help='Fix intermediate bounds when GCP-CROWN cuts are used.',
                          hierarchy=h + ["fix_intermediate_bounds"])
        self.add_argument("--manual_cuts", type=str, default=None,
                          help='Feed manual cuts using this argument. The cut file should be named xxx.cuts.',
                          hierarchy=h + ["manual_cuts"])
        self.add_argument("--record_cplex_cut_to", type=str, default=None,
                          help='Path to CSV file for recording CPLEX cuts for replay. If specified, all cuts fetched from CPLEX will be saved. The batch ID will be automatically appended to the filename (e.g., cut_records.csv becomes cut_records_batch_0.csv). Pickle files will be saved in a folder next to the CSV file.',
                          hierarchy=h + ["record_cplex_cut_to"])
        self.add_argument("--read_cplex_cut_from", type=str, default=None,
                          help='Path to CSV file for reading CPLEX cuts for replay. If specified, cuts will be replayed from this CSV file instead of fetching from CPLEX. The batch ID will be automatically appended to the filename (e.g., cut_records.csv becomes cut_records_batch_0.csv).',
                          hierarchy=h + ["read_cplex_cut_from"])

        h = ["bab", "cut", "biccos"]
        self.add_argument("--biccos_cuts", action='store_true',
                          help='BICCOS z cuts from the Branch and Bound process.',
                          hierarchy=h + ["enabled"])
        self.add_argument("--biccos_disable_auto_param", action='store_false',
                          help='BICCOS auto start the mts and set parameters.',
                          hierarchy=h + ["auto_param"])
        self.add_argument("--biccos_no_constraint_strengthening", action='store_false',
                          dest='constraint_strengthening',
                          help='Do not use constraint strengthening in BICCOS.',
                          hierarchy=h + ["constraint_strengthening"])
        self.add_argument("--biccos_drop_ratio", type=float, default=0.5,
                          help='Neuron drop ratio parameter when using inferred cut and neuron influence score heuristic.',
                          hierarchy=h + ["drop_ratio"])
        self.add_argument("--biccos_max_infer_iter", type=int, default=20,
                          help='After max_infer_iter iterations, we will stop inferencing cuts.',
                          hierarchy=h + ["max_infer_iter"])
        self.add_argument("--biccos_max_domain", type=int, default=1e4,
                          help='After max_domain visited, we will stop inferencing cuts.',
                          hierarchy=h + ["max_domain"])
        self.add_argument("--biccos_dropping_heuristics", default="neuron_influence_score",
                          choices=["neuron_influence_score", "random_drop", "sparse_opt", "None"],
                          help='Neuron dropping heuristic.',
                          hierarchy=h + ["heuristic"])
        self.add_argument("--biccos_save_cuts", action='store_true',
                          help='Save cuts to log/biccos.txt.',
                          hierarchy=h + ["save_cuts"])

        h = ["bab", "cut", "biccos", "multi_tree_branching"]
        self.add_argument("--multi_tree_branching", action="store_true",
                          help="Enables multi-tree branching. Instead of picking one new split per unknown domain, multiple splits are tested. In the end, the best (measured in bound tightness) regular BaB tree is selected.",
                          hierarchy=h + ["enabled"])
        self.add_argument("--multi_tree_branching_disable_restore_best_tree", action="store_false",
                          help="Restore the best tree after the multi-tree search.",
                          hierarchy=h + ["restore_best_tree"])
        self.add_argument("--multi_tree_branching_keep_n_best_domains", type=int, default=50,
                          help="Number of best domains to keep in multi-tree branching.",
                          hierarchy=h + ["keep_n_best_domains"])
        self.add_argument("--multi_tree_branching_target_batch_size", type=int, default=200,
                          help="Target batch size. Determines the branching factor: target_batch_size // keep_n_best_domains.",
                          hierarchy=h + ["target_batch_size"])
        self.add_argument("--multi_tree_branching_iterations", type=int, default=3,
                          help="Number of iterations for multi-tree branching. Too large values may lead to slow speed.",
                          hierarchy=h + ["iterations"])

        h = ["bab", "clip_n_verify"]
        self.add_argument("--clip_constraints_rearrange", action='store_true',
                          help="Before solving LP problems with coordinate descent, rearrange linear constraints based on their distances to the input domain centroids.",
                          hierarchy=h + ["rearrange_constraints"])
        self.add_argument("--clip_alpha_crown", action ='store_true',
                          help="Clip the input domain and intermediate bounds during alpha-CROWN optimization.",
                          hierarchy=h + ["alpha_crown"])
        self.add_argument("--clip_enable_prune", action="store_true",
                          help="Use lp to check the feasibility of the clipped domain.",
                          hierarchy=h + ['prune'], private=True)
        self.add_argument("--clip_using_final_layer", action="store_true",
                          help="Use lp to check the feasibility of the clipped domain.",
                          hierarchy=h + ['final_layer'], private=True)

        h = ["bab", "clip_n_verify", "clip_input_domain"]
        self.add_argument("--enable_clip_input", action='store_true',
                          help='Shrinks subdomains based on their specification by using relaxed clip.',
                          hierarchy=h + ["enabled"])
        self.add_argument("--clip_input_type", type=str, 
                          default="relaxed", choices=["relaxed", "complete"],
                          help="Specify the type of input clip. If set to be 'complete', linear constraints will be applied when concretizing all hidden layers.",
                          hierarchy=h + ["clip_type"])
        self.add_argument("--clip_input_neuron_selection_type", type=str, default="ratio", choices=["ratio", "number"],
                          help="How to select the neuron to concretize with the constraints (complete clip)." ,
                          hierarchy=h + ["clip_neuron_selection_type"])
        self.add_argument("--clip_input_neuron_selection_value", type=float, default=-1.0,
                          help="Number of neurons to concretize with the constraints (complete clip). The rest will use the relaxed method. " \
                               "The default value is -1.0. A negative value means neuron selection will be disabled. " \
                               "If select neurons with ratio, the legal value range would be [0.0, 1.0]. " \
                               "(The ratio selection is based on the number of UNSTABLE neurons per layer) "
                               "Otherwise, directly set the number of neurons to select in each layer.",
                          hierarchy=h + ["clip_neuron_selection_value"])
        self.add_argument("--clip_iterations", type=int, default=1,
                          help="The number of times to call the clipping procedure per BaB round.",
                          hierarchy=h + ["clip_iterations"])
        self.add_argument("--clip_calculate_volume_metrics", action='store_true',
                          help="If enabled, will also calculate the total box volume on a batch of unverified domains before "
                               "and after clipping in a round of BaB. Though this prints additional information to the "
                               "console that may be insightful, calculating the volume in a numerically stable manner "
                               "requires performing a series of calculations that can add quite a bit of overhead.",
                          hierarchy=h + ["clip_calculate_volume_metrics"])

        h = ["bab", "clip_n_verify", "clip_interm_domain"]
        self.add_argument("--enable_complete_clip", action='store_true',
                          help='Using neuron split constraints to improve the interm bounds.',
                          hierarchy=h + ["enabled"])
        self.add_argument("--clip_interm_w_input", action='store_true',
                          help='Using neuron split constraints to improve the interm bounds.',
                          hierarchy=h + ["with_input"])
        self.add_argument("--clip_in_alpha_crown", action='store_true',
                          help='Using neuron split constraints to improve the interm bounds.',
                          hierarchy=h + ["clip_in_alpha_crown"])
        self.add_argument('--clip_topk_objective', type=int, default=20,
                          help='Select top-k objective based on their kfsb-score.',
                          hierarchy=h + ['topk_objective'])

        self.add_argument("--clipper_only_bab_timeout", action='store_true',
                          help="Enable the pre-β-CROWN α-zono substitution BaB "
                               "(clipper_bab.try_clipper_only_bab): build a lifted α-zono "
                               "generator state + per-neuron slope-opt, then a factored dual "
                               "BaB; no compute_bounds. No time early-stop is applied.",
                          hierarchy=["bab"] + ["clipper_only_bab_timeout"])
        self.add_argument("--clipper_only_K", type=int, default=1,
                          help="Dual-ascent steps per node in the factored clipper kernel "
                               "(K=1 warm-start is the default; higher K = tighter per node).",
                          hierarchy=["bab"] + ["clipper_only_K"])
        self.add_argument("--clipper_only_fast_Kb", type=int, default=256,
                          help="Number of line-search buckets for the compiled clipper kernel.",
                          hierarchy=["bab"] + ["clipper_only_fast_Kb"])
        self.add_argument("--clipper_only_slope_iters", type=int, default=80,
                          help="Adam iterations for per-neuron zonotope slope (alpha) "
                               "optimization when building the alpha-zono gen-state.  Higher "
                               "→ tighter root bound → fewer BaB nodes; ~0.06s/iter.",
                          hierarchy=["bab"] + ["clipper_only_slope_iters"])
        self.add_argument("--clipper_only_shared_slope_iters", type=int, default=-1,
                          help="Slope optimization iterations for the shared multi-query gen-state. "
                               "Negative means 0 for multi-query and clipper_only_slope_iters for single-query.",
                          hierarchy=["bab"] + ["clipper_only_shared_slope_iters"])
        self.add_argument("--clipper_only_shared_frontier_cap", type=int, default=250000,
                          help="Frontier cap for the cheap shared multi-query clipper attempt before per-query fallback.",
                          hierarchy=["bab"] + ["clipper_only_shared_frontier_cap"])

        h = ["bab", "branching"]
        self.add_argument("--branching_method", default="kfsb",
                          choices=["babsr", "fsb", "kfsb", "kfsb-intercept-only",
                                   "random", "intercept", "naive", "sb",
                                   "nonlinear", "brute-force"],
                          help='Branching heuristic. babsr is fast but less accurate; fsb is slow but most accurate; kfsb is usually a balance; kfsb-intercept-only is faster but may lead to worse branching; sb is fast smart branching which relies on the A matrix.',
                          hierarchy=h + ["method"])
        self.add_argument("--branching_candidates", type=int, default=3,
                          help='Number of candidates to consider when using fsb or kfsb. More candidates lead to slower but better branching.',
                          hierarchy=h + ["candidates"])
        self.add_argument("--branching_reduceop", choices=["min", "max", "mean", "auto"], default="min",
                          help='Reduction operation to compute branching scores from two sides of a branch (min or max). max can work better on some models.',
                          hierarchy=h + ["reduceop"])
        self.add_argument('--enable_intermediate_bound_opt', action='store_true',
                          default=False, help='Enable optimizing intermediate bounds for during bab.',
                          hierarchy=h + ["enable_intermediate_bound_opt"])
        self.add_argument("--branching_input_and_activation", action='store_true',
                          help="Branching input domains and relu domains (experimental).",
                          hierarchy=h + ["branching_input_and_activation"])
        self.add_argument("--branching_input_and_activation_order", nargs='+',
                          type=str, default=['input', 'relu'],
                          help="Order of branching input domains and relu domains (experimental).",
                          hierarchy=h + ["branching_input_and_activation_order"])
        self.add_argument("--branching_input_iterations", type=int, default=30,
                          help="Number of iterations to run input split before we run relu split.",
                          hierarchy=h + ["branching_input_iterations"])
        self.add_argument("--branching_relu_iterations", type=int, default=50,
                          help="Number of iterations to run relu split before we run input split.",
                          hierarchy=h + ["branching_relu_iterations"])
        self.add_argument("--save_visited_domains_to", type=str, default="",
                          help="Save visited domains to the specified path for visualization.",
                          hierarchy=h + ["save_visited_domains_to"])
        self.add_argument('--decision_sub_batch_size', type=int, default=None,
                          help='Maximum sub-batch size when computing bounds for branching candidates (only applies to KFSB and FSB). Useful to avoid OOM.',
                          hierarchy=h + ["decision_sub_batch_size"])

        h = ["bab", "branching", "nonlinear_split"]
        self.add_argument("--nonlinear_split_method", type=str, default='bbps',
                          choices=['bbps','babsr-like'],
                          help='Branching heuristic for the general nonlinear functions (either the default BBPS heuristic, or a BaBSR-like as a baseline).',
                          hierarchy=h + ['method'])
        self.add_argument("--disable_nonlinear_split", action='store_true',
                          help='Disable GenBaB even if there are non-ReLU functions to split.',
                          hierarchy=h + ['disable'])
        self.add_argument("--branching_point_method", type=str, default='uniform',
                          choices=['uniform', 'opt'],
                          help='For general nonlinear functions, the method for choosing the branching point ("uniform" stands for branching in the middle while "opt" stands for using pre-optimized branching points).',
                          hierarchy=h + ['branching_point_method'])
        self.add_argument("--nonlinear_branches", type=int, default=2,
                          help='Number of branches for nonlinear branching.',
                          hierarchy=h + ['num_branches'])
        self.add_argument("--nonlinear_split_filtering", action='store_true',
                          dest='nonlinear_split_filtering',
                          help='KFSB-like filtering in general nonlinear branching.',
                          hierarchy=h + ['filter'])
        self.add_argument("--nonlinear_split_filtering_beta", action='store_true',
                          dest='nonlinear_split_filtering_beta',
                          help='Use beta in the KFSB-like filtering.',
                          hierarchy=h + ['filter_beta'])
        self.add_argument("--nonlinear_split_filtering_clamp", action='store_true',
                          dest='nonlinear_split_filtering_clamp',
                          help='Clamping scores to 0 in BBPS filtering.',
                          hierarchy=h + ['filter_clamp'])
        self.add_argument("--filter_batch_size", type=int, default=10000,
                          help='Batch size for filtering.',
                          hierarchy=h + ['filter_batch_size'])
        self.add_argument("--filter_iterations", type=int, default=25,
                          help="Number of iterations for filtering.",
                          hierarchy=h + ['filter_iterations'])
        self.add_argument("--nonlinear_split_relu_only",
                          action="store_true",
                          help="When using BBPS, only consider branching ReLU instead of all the nonlinearities.",
                          hierarchy=h + ['relu_only'])
        self.add_argument("--loose_tanh_threshold", type=float, default=None,
                          help='Set a threshold for tanh/sigmoid to use a different relaxation when the pre-activation bounds are too loose.',
                          hierarchy=h + ['loose_tanh_threshold'])

        h = ["bab", "branching", "nonlinear_split", "branching_point"]
        self.add_argument("--branching_point_db", type=str, default='branching_points.pt',
                          help='Path to store pre-computed branching points.',
                          hierarchy=h + ['db_path'])
        self.add_argument('--branching_point_num_iterations', type=int, default=1000,
                          help='Number of iterations for the optimization.',
                          hierarchy=h + ['num_iterations'])
        self.add_argument('--branching_point_batch_size', type=int, default=1000000,
                          help='Batch size for the optimization.',
                          hierarchy=h + ['batch_size'])
        self.add_argument('--branching_point_range_l', type=float, default=-5.0,
                          help='Range (lower bound) for the optimization.',
                          hierarchy=h + ['range_l'])
        self.add_argument('--branching_point_range_u', type=float, default=5.0,
                          help='Range (upper bound) for the optimization.',
                          hierarchy=h + ['range_u'])
        self.add_argument('--branching_point_log_interval', type=int, default=100,
                          help='Log interval for the optimization.',
                          hierarchy=h + ['log_interval'])
        self.add_argument('--branching_point_step_size_1d', type=float, default=0.01,
                          help='Step size for 1d nonlinearities.',
                          hierarchy=h + ['step_size_1d'])
        self.add_argument('--branching_point_step_size', type=float, default=0.2,
                          help='Step size for 2d and above nonlinearities.',
                          hierarchy=h + ['step_size'])

        h = ["bab", "branching_decision_precompute"]
        self.add_argument('--no_branching_decision_precompute', action='store_false',
                          help='Try to enable branching decision precomputing.',
                          hierarchy=h + ["enable"])

        h = ["bab", "branching", "input_split"]
        self.add_argument("--split_hint", type=float, nargs='+',
                          help="Specifies value to split at.",
                          hierarchy=h + ["split_hint"])
        self.add_argument("--enable_input_split", action='store_true',
                          help='Branch on input domain rather than unstable neurons.',
                          hierarchy=h + ["enable"])
        self.add_argument("--enable_beta", action='store_true',
                          help='Branch on input domain rather than unstable neurons with beta crown for constraints.',
                          hierarchy=h + ["beta"])
        self.add_argument("--use_relaxation_warmstart", action='store_true',
                          help='Use CROWN relaxation minimizer x* to warm-start PGD attack. '
                               'Extracts the minimizer of the linear relaxation and uses it as initial point.',
                          hierarchy=h + ["use_relaxation_warmstart"])
        self.add_argument("--reorder_bab", action="store_true",
                          help="Uses a reordered implementation of the input BaB procedure that bounds, splits and "
                               "clips domains rather than splitting, bounding, and clipping domains.",
                          hierarchy= h + ["reorder_bab"])
        self.add_argument("--input_split_input_dim_threshold", type=int, default=20,
                          help='When complete_verifier=auto, use input split when the input dimension is no larger than this threshold.',
                          hierarchy=h + ["input_dim_threshold"])
        self.add_argument("--input_split_adv_check", type=int, default=0,
                          help='After the number of visited nodes, we will run adv_check in input split.',
                          hierarchy=h + ["adv_check"])
        self.add_argument("--input_split_partitions", type=int, default=2,
                          help='How many domains to split to for each dimension at each time. By default, it is 2. In very few limited experimental cases, can change to larger numbers.',
                          hierarchy=h + ["split_partitions"])
        self.add_argument("--sb_margin_weight", default=1.0, type=float,
                          help='Weight for the margin term in the sb heuristic.',
                          hierarchy=h + ["sb_margin_weight"])
        self.add_argument('--sb_sum', action='store_true',
                          help='Use sum for multiple specs in sb.',
                          hierarchy=h + ['sb_sum'])
        self.add_argument('--sb_primary_spec', type=int, default=None,
                          help='Focus on one particular spec for the SB score.',
                          hierarchy=h + ['sb_primary_spec'])
        self.add_argument('--bf_backup_thresh', type=float, default=-1,
                          help='Threshold for using the SB score as the backup when the brute force score is too bad.',
                          hierarchy=h + ['bf_backup_thresh'])
        self.add_argument('--bf_rhs_offset', type=float, default=0,
                          help='An offset on RHS used in computing the brute force heuristic.',
                          hierarchy=h + ['bf_rhs_offset'])
        self.add_argument('--bf_iters', type=int, default=1e9,
                          help='Number of iterations to use brute force.',
                          hierarchy=h + ['bf_iters'])
        self.add_argument('--bf_batch_size', type=int, default=100000,
                          help='A special batch size for brute force.',
                          hierarchy=h + ['bf_batch_size'])
        self.add_argument('--bf_zero_crossing_score', action='store_true',
                          help='A zero crossing score in BF.',
                          hierarchy=h + ['bf_zero_crossing_score'])
        self.add_argument('--touch_zero_score', type=float, default=0,
                          help='A touch-zero score in BF.',
                          hierarchy=h + ['touch_zero_score'])
        self.add_argument('--no_ibp_enhancement', action='store_false',
                          help='Use IBP bounds to enhance.',
                          hierarchy=h + ['ibp_enhancement'])
        self.add_argument('--no_compare_input_split_with_old_bounds',
                          action='store_false',
                          dest='compare_with_old_bounds',
                          help='Compare bounds after an input split with bounds before the split and take the better one.',
                          hierarchy=h + ['compare_with_old_bounds'])
        self.add_argument('--input_split_update_rhs_with_attack',
                          action='store_true',
                          help='Run attack during input split and update RHS. BaB does not stop even if any counterexample is found.',
                          hierarchy=h + ['update_rhs_with_attack'])
        self.add_argument("--sb_coeff_thresh", default=1e-3, type=float,
                          help='Clamp values of coefficient matrix (A matrix) for sb branching heuristic.',
                          hierarchy=h + ["sb_coeff_thresh"])
        self.add_argument("--sb_min_width_ratio", default=0.0, type=float,
                          help='For sb branching, split the widest input (relative to the root box) instead of the chosen one when the chosen one is narrower than this fraction of it; |a_i| w_i / 2 never selects an input whose coefficient is 0, and spatial BaB needs every width to shrink. 0 disables.',
                          hierarchy=h + ["sb_min_width_ratio"])
        self.add_argument("--angle_first_ratio", default=0.0, type=float,
                          help='For sb branching, split the widest variable of a circle equality x_i^2 + x_j^2 = const (cos/sin of a rotation in rigid-body models) while its width exceeds this fraction of its root width; the relaxation of a circle is informative only on angle sectors. Overrides the lifted violation choice and the width guard. 0 disables.',
                          hierarchy=h + ["angle_first_ratio"])
        self.add_argument("--sb_strong_candidates", default=0, type=int,
                          help='Objective-aware strong branching (input_split/strong_branching.py): bound both midpoint children of the top-k sb candidates of every domain and split the one with the best product of objective-bound gains. -1 = every input, 0 disables.',
                          hierarchy=h + ["sb_strong_candidates"])
        self.add_argument("--sb_strong_max_domains", default=-1, type=int,
                          help='Strong-branch only this many domains of each batch with the lowest objective bound; the rest keep the sb choice (-1 = all).',
                          hierarchy=h + ["sb_strong_max_domains"])
        self.add_argument("--sb_strong_max_children", default=65536, type=int,
                          help='Candidate children bounded per strong-branching call (chunk size).',
                          hierarchy=h + ["sb_strong_max_children"])
        self.add_argument("--sb_strong_candidates_unconstrained", default=0, type=int,
                          help='Like sb_strong_candidates, for problems whose spec is the objective row alone (no constraint rows, no input linear constraints); it replaces sb_strong_candidates there.',
                          hierarchy=h + ["sb_strong_candidates_unconstrained"])
        self.add_argument("--sb_strong_max_inputs_unconstrained", default=0, type=int,
                          help='No strong branching on objective-only problems with more inputs than this (0 = no limit).',
                          hierarchy=h + ["sb_strong_max_inputs_unconstrained"])
        self.add_argument("--sb_strong_domain_frac", default=1.0, type=float,
                          help='Strong-branch only this fraction of each batch (the domains with the lowest objective bound); the rest keep the sb choice.',
                          hierarchy=h + ["sb_strong_domain_frac"])
        self.add_argument("--sb_strong_domain_frac_unconstrained", default=1.0, type=float,
                          help='Like sb_strong_domain_frac, for problems whose spec is the objective row alone.',
                          hierarchy=h + ["sb_strong_domain_frac_unconstrained"])
        self.add_argument("--sb_strong_min_domains", default=0, type=int,
                          help='Strong-branch at least this many domains of a batch (when it has that many), whatever the fraction.',
                          hierarchy=h + ["sb_strong_min_domains"])
        self.add_argument('--input_split_sort_index', type=int,
                          help='The output index to use for sorting domains in the input split.',
                          hierarchy=h + ["sort_index"])
        self.add_argument('--input_split_sort_heuristic', type=str,
                          default='lb_rhs', choices=['lb_rhs', 'lb_only'],
                          help='The heuristic type for sorting domains in input bab. lb_rhs means (lb-rhs).',
                          hierarchy=h + ["sort_heuristic"])
        self.add_argument('--no_input_split_sort_descending', action='store_false',
                          dest="input_split_sort_descending",
                          help='Sort input split domains in an ascending/descending way.',
                          hierarchy=h + ["sort_descending"])
        self.add_argument('--input_split_show_progress', action='store_true',
                          help='Show progress during input split.',
                          hierarchy=h + ["show_progress"])
        self.add_argument('--input_split_presplit_domains', type=str, default=None,
                          help='Load pre-split domains from a file.',
                          hierarchy=h + ["presplit_domains"])
        self.add_argument('--input_split_skip_getting_worst_domain',
                          action='store_true',
                          help='Skip getting the worst domain at the end of each iteration, to save some time cost when the domain list is long.',
                          hierarchy=h + ["skip_getting_worst_domain"])

        h = ["bab", "attack"]  # BaB-Attack options.
        self.add_argument("--enable_bab_attack", action='store_true',
                          help='Enable beam search based BaB-attack.',
                          hierarchy=h + ["enabled"])
        self.add_argument("--beam_candidates_number", type=int, default=8,
                          help='Number of candidates in beam search.',
                          hierarchy=h + ["beam_candidates"])
        self.add_argument("--beam_split_depth", type=int, default=7,
                          help='Max additional level of splits to expand during beam search in BaB-Attack.',
                          hierarchy=h + ["beam_depth"])
        self.add_argument('--max_dive_fix_ratio', type=float, default=0.8,
                          help='Maximum ratio of fixed neurons during diving in BaB-Attack.',
                          hierarchy=h + ['max_dive_fix_ratio'])
        self.add_argument('--min_local_free_ratio', type=float, default=0.2,
                          help='Minimum ratio of free neurons during local search in BaB-Attack.',
                          hierarchy=h + ['min_local_free_ratio'])
        self.add_argument('--submip_start_iteration', type=int, default=5,
                          help='Iteration number to start sub-MIPs in BaB-Attack.',
                          hierarchy=h + ["mip_start_iteration"])
        self.add_argument('--submip_timeout', type=float, default=30.0,
                          help='Sub-MIP timeout threshold.',
                          hierarchy=h + ["mip_timeout"])
        self.add_argument('--adv_pool_threshold', type=float, default=None,
                          help='Minimum value of difference when adding to adv_pool; default `None` means auto select.',
                          hierarchy=h + ["adv_pool_threshold"])
        self.add_argument("--refined_mip_attacker", action='store_true',
                          help='Use full alpha crown bounds to refined intermediate bounds for sub-MIPs.',
                          hierarchy=h + ["refined_mip_attacker"])
        self.add_argument("--refined_batch_size", type=float, default=None,
                          help='Batch size for full alpha-CROWN to refined intermediate bounds for mip solver attack (to avoid OOM), default None to be the same as mip_multi_proc.',
                          hierarchy=h + ["refined_batch_size"])

        h = ["attack"]
        self.add_argument('--no_general_attack', action='store_false',
                          dest='general_attack',
                          help="Enable/disable the new version of PGD attack which supports general specifications.",
                          hierarchy=h + ["general_attack"])
        self.add_argument('--pgd_order', choices=["before", "after", "middle", "input_bab", "skip"], default="before",
                          help='Run PGD attack before/after/during incomplete verification, only during input bab, or skip it.', hierarchy=h + ["pgd_order"])
        self.add_argument('--pgd_steps', type=int, default=100,
                          help="Steps of PGD attack.",
                          hierarchy=h + ["pgd_steps"])
        self.add_argument('--pgd_restarts', type=int, default=30,
                          help="Number of random PGD restarts.",
                          hierarchy=h + ["pgd_restarts"])
        self.add_argument('--pgd_batch_size', type=int, default=int(1e8),
                          help='Batch size for number of restarts in PGD.',
                          hierarchy=h + ["pgd_batch_size"])
        self.add_argument('--no_pgd_early_stop', action='store_false',
                          dest='pgd_early_stop',
                          help="Enable/disable early stop PGD when an adversarial example is found.",
                          hierarchy=h + ["pgd_early_stop"])
        self.add_argument('--pgd_lr_decay', type=float, default=0.99,
                          help='Learning rate decay factor used in PGD attack.',
                          hierarchy=h + ["pgd_lr_decay"])
        self.add_argument('--pgd_alpha', type=str, default="auto",
                          help='Step size of PGD attack. Default (auto) is epsilon/4.',
                          hierarchy=h + ["pgd_alpha"])
        self.add_argument('--pgd_alpha_scale', action='store_true',
                          help='Scale PGD alpha according to data_max-data_min.',
                          hierarchy=h + ["pgd_alpha_scale"])
        self.add_argument('--pgd_loss_mode', type=str, choices=['hinge', 'sum'],
                          help='Loss mode for choosing the best delta.',
                          hierarchy=h + ["pgd_loss_mode"])
        self.add_argument('--pgd_restart_when_stuck', action='store_true',
                          help='Restart adversarial noise when they do not change over attack iterations.',
                          hierarchy=h + ["pgd_restart_when_stuck"])
        self.add_argument("--mip_attack", action='store_true',
                          help='Use MIP (Gurobi) based attack if PGD cannot find a successful adversarial example.',
                          hierarchy=h + ["enable_mip_attack"])
        self.add_argument("--adv_saver", type=str, default='default_adv_saver',
                          help='Customized saver of adverserial examples.',
                          hierarchy=h + ["adv_saver"])
        self.add_argument("--adv_verifier", type=str, default='default_adv_verifier',
                          help='Customized verifier of adverserial examples.',
                          hierarchy=h + ["adv_verifier"])
        self.add_argument("--early_stop_condition", type=str, default='default_early_stop_condition',
                          help='Customized early stop condition.',
                          hierarchy=h + ["early_stop_condition"])
        self.add_argument("--adv_example_finalizer", type=str, default='default_adv_example_finalizer',
                          help='Customized generation of adversarial examples, margins computation, etc.',
                          hierarchy=h + ["adv_example_finalizer"])
        self.add_argument("--pgd_loss", type=str, default="default_pgd_loss",
                          help='Customized pgd loss.',
                          hierarchy=h + ["pgd_loss"])
        self.add_argument('--cex_path', type=str, default='./test_cex.txt',
                          help='Save path for counter-examples.',
                          hierarchy=h + ["cex_path"])
        self.add_argument('--attack_mode', type=str, default='PGD',
                          help='Attack algorithm, including vanilla PGD and PGD with diversified output (Tashiro et al.), and GAMA loss (Sriramanan et al.).',
                          choices=['diversed_PGD', 'diversed_GAMA_PGD', 'PGD', 'boundary'],
                          hierarchy=h + ["attack_mode"])
        self.add_argument('--attack_tolerance', type=float, default=0.,
                          help='Tolerance when checking whether attack is successful. Positive values relax the check; negative values make it stricter.',
                          hierarchy=h + ["attack_tolerance"])
        self.add_argument('--attack_func', type=str, default='attack_with_general_specs',
                          help='The specific customized attack.',
                          hierarchy=h + ["attack_func"])
        self.add_argument('--attack_gama_lambda', type=float, default=10.,
                          help='Regularization parameter in GAMA attack.',
                          hierarchy=h + ["gama_lambda"])
        self.add_argument('--attack_gama_decay', type=float, default=0.9,
                          help='Decay of regularization parameter in GAMA attack.',
                          hierarchy=h + ["gama_decay"])

        h = ["attack", "input_split"]
        self.add_argument('--input_split_pgd_steps', type=int, default=100,
                          help="Steps of PGD attack in input split before branching starts.",
                          hierarchy=h + ["pgd_steps"])
        self.add_argument('--input_split_pgd_restarts', type=int, default=30,
                          help="Number of random PGD restarts in input split before branching starts.", hierarchy=h + ["pgd_restarts"])
        self.add_argument('--input_split_pgd_alpha', type=str, default="auto",
                          help="Step size (alpha) in input split before branching starts.",
                          hierarchy=h + ["pgd_alpha"])

        h = ["attack", "input_split_check_adv"]
        self.add_argument('--enable_check_adv', choices=["auto", "true", "false"], default='auto',
                          help="Enable or disable counterexample checking during input-space branch-and-bound. Default is 'auto', which disables check_adv when pgd_order is skip.",
                          hierarchy=h + ["enabled"])
        self.add_argument('--input_split_check_adv_auto_enlarge_batch_size',
                          choices=["auto", "true", "false"], default='auto',
                          help='Control automatic increase of PGD attack batch size during input split; "auto" follows solver.auto_enlarge_batch_size.',
                          hierarchy=h + ["auto_enlarge_batch_size"])
        self.add_argument('--input_split_check_adv_pgd_steps', type=int, default=5,
                          help="Steps of PGD attack in input split after each branching.",
                          hierarchy=h + ["pgd_steps"])
        self.add_argument('--input_split_check_adv_pgd_restarts', type=int, default=5,
                          help="Number of random PGD restarts in input split after each branching.",
                          hierarchy=h + ["pgd_restarts"])
        self.add_argument('--input_split_check_adv_pgd_alpha', type=str, default="auto",
                          help="Step size (alpha) in input split after each branching.",
                          hierarchy=h + ["pgd_alpha"])
        self.add_argument('--input_split_check_adv_max_num_domains', type=int, default=10,
                          help="Maximum number of domains for running attack during input split.",
                          hierarchy=h + ["max_num_domains"])

        h = ["debug"]
        self.add_argument('--view_model', action='store_true',
                          help='Print more detailed model information for analysis.',
                          hierarchy=h + ['view_model'])
        self.add_argument("--lp_test", type=str, default=None,
                          choices=["MIP", "LP", None],
                          help='Debugging option. Do not use.',
                          hierarchy=h + ['lp_test'], private=True)
        self.add_argument("--rescale_vnnlib_ptb", type=float, default=None,
                          help='Rescale the perturbation on X in vnnlib files to make them smaller. Used for debugging.',
                          hierarchy=h + ['rescale_vnnlib_ptb'], private=True)
        self.add_argument("--test_optimized_bounds", action='store_true', default=False,
                          help='Debugging option. Compares optimized bounds with those from Gurobi for the relaxed LP formulation',
                          hierarchy=h + ['test_optimized_bounds'], private=True)
        self.add_argument("--test_optimized_bounds_after_n_iterations", type=int, default=0,
                          help='Debugging option. Controls how many optimization steps for all layers are done before bounds are checked against the LP solver',
                          hierarchy=h + ['test_optimized_bounds_after_n_iterations'], private=True)
        self.add_argument("--print_verbose_decisions", action="store_true",
                          help="Print more detailed information about branching decisions",
                          hierarchy=h + ['print_verbose_decisions'], private=True)
        self.add_argument('--biccos_mip_sanity_check', action='store_true',
                          help='Use mip solver to debug BICCOS.',
                          hierarchy=h + ['biccos_mip_sanity_check'])
        self.add_argument("--sanity_check", type=str, default=None,
                          choices=["Full", "Worst", None],
                          help="Using pgd upper bound as rhs-offset to check the feasibility of the method. Warning: If rhs_offset was specified "
                               "in the config or command line, it will get ignored.",
                          hierarchy=h + ['sanity_check'], private=True)
        self.add_argument("--sanity_check_delta", type=float, default=1e-3,
                          help="Slack added on top of the PGD attack margin when --sanity_check "
                               "builds the rhs offset. Was hard-coded to 1e-3.",
                          hierarchy=h + ['sanity_check_delta'], private=True)
        self.add_argument("--lp_sanity_check", action='store_true',
                          help="Using linear programming as the solver to check the feasibility of the method,",
                          hierarchy=h + ['lp_sanity_check'], private=True)
        self.add_argument("--sanity_check_with_graphs", action="store_const", const=2, default=0,
                          help="Sets '--sanity_check' as well as saves visual graphs of the lower bound approaching the rhs.",
                          hierarchy=h + ['sanity_check'], private=True)
        self.add_argument("--z_split", action="store_true",
                          help="The regular BaB split constraints are x >= 0 and x <= 0. This option adds z = 1 and z = 0 to see the difference between x split and z split.",
                          hierarchy=h + ['z_split_enabled'], private=True)
        self.add_argument("--save_minimal_config", type=str, default=None,
                          help="Path to save a minimal config file.",
                          hierarchy=h + ['save_minimal_config'])
        self.add_argument("--save_minimal_config_omit_keys", type=str, nargs="+", default=[],
                          help="Keys to omit from the minimal config file.",
                          hierarchy=h + ['save_minimal_config_omit_keys'])

        h = ["solving"]
        self.add_argument('--fixed_primal_report_only', action="store_true", default=False,
                          help='Treat solving.solver_res as a REPORTING upper bound only: it is '
                               'not written into the objective RHS and cannot prune domains. Use '
                               'this to watch the dual converge toward a deliberately loosened '
                               'incumbent without the incumbent poisoning the certified bound.',
                          hierarchy=h+ ['fixed_primal_report_only'])
        self.add_argument("--solving_mode", action="store_true",
                          help="Enable solving mode.",
                          hierarchy=h + ["solving_mode"])
        self.add_argument('--enable_attack_counter', action="store_true",
                          help='Whether to enable attack_counter or not in domain.',
                          hierarchy=h+ ['enable_attack_counter'])
        self.add_argument('--enable_solution_point', action="store_true", default=False,
                          help='Whether to apply location of the solver_res.',
                          hierarchy=h+["enable_solution_point"])
        self.add_argument('--solution_point', type=str, default="",
                          help='The location of the solution point.',
                          hierarchy=h+ ['solution_point'])
        self.add_argument('--enable_solver_res', action="store_true", default=False,
                          help='Use a torch-saved scalar feasible objective as the fixed primal bound.',
                          hierarchy=h+ ['enable_solver_res'])
        self.add_argument('--solver_res', type=str, default="",
                          help='Path to a torch-saved scalar feasible objective.',
                          hierarchy=h+ ['solver_res'])
        self.add_argument('--obj_index', type=int, default=0,
                          help='The index of the objective in the output.',
                          hierarchy=h+ ['obj_index'])
        self.add_argument('--is_maximization', action='store_true', default=False,
                          help='Whether the optimization problem is a maximization problem. When True, primal and dual solutions are negated for display.',
                          hierarchy=h + ['is_maximization'])

    def update_arguments(self):
        """Adaptively tune arguments."""

        if (self["specification"]["norm"] != torch.inf
            and self["attack"]["pgd_order"] != "skip"):
            print('Only Linf-norm attack is supported, the pgd_order will be changed to skip')
            self["attack"]["pgd_order"] = "skip"

    def add_argument(self, *args, **kwargs):
        """Add a single parameter to the parser. We will check the 'hierarchy' specified and then pass the remaining arguments to argparse."""
        if 'hierarchy' not in kwargs:
            raise ValueError("please specify the 'hierarchy' parameter when using this function.")
        hierarchy = kwargs.pop('hierarchy')
        help = kwargs.get('help', '')
        private_option = kwargs.pop('private', False)
        # Make sure valid help is given
        if not private_option:
            if len(help.strip()) < 10:
                raise ValueError(
                    f'Help message must not be empty, and must be detailed enough. "{help}" is not good enough.')
            elif (not help[0].isupper()) or help[-1] != '.':
                raise ValueError(
                    f'Help message must start with an upper case letter and end with a dot (.); your message "{help}" is invalid.')
            elif help.count('%') != help.count('%%') * 2:
                raise ValueError(
                    f'Please escape "%" in help message with "%%"; your message "{help}" is invalid.')
        self.defaults_parser.add_argument(*args, **kwargs)
        # Build another parser without any defaults.
        if 'default' in kwargs:
            kwargs.pop('default')
        self.no_defaults_parser.add_argument(*args, **kwargs)
        # Determine the variable that will be used to save the argument by argparse.
        if 'dest' in kwargs:
            dest = kwargs['dest']
        else:
            dest = re.sub('^-*', '', args[-1]).replace('-', '_')
        # Also register this parameter to the hierarchy dictionary.
        self.config_file_hierarchies[dest] = hierarchy
        if hierarchy is not None and not private_option:
            self.help_messages[','.join(hierarchy)] = help

    def set_dict_by_hierarchy(self, args_dict, h, value, nonexist_ok=True):
        """Insert an argument into the dictionary of all parameters. The level in this dictionary is determined by list 'h'."""
        # Create all the levels if they do not exist.
        current_level = self.all_args
        assert len(h) != 0
        for config_name in h:
            if config_name not in current_level:
                if nonexist_ok:
                    current_level[config_name] = {}
                else:
                    raise ValueError(f"Config key {h} not found!")
            last_level = current_level
            current_level = current_level[config_name]
        # Add config value to leaf node.
        last_level[config_name] = value

    def construct_config_dict(self, args_dict, nonexist_ok=True):
        """Based on all arguments from argparse, construct the dictionary of all parameters in self.all_args."""
        for arg_name, arg_val in args_dict.items():
            h = self.config_file_hierarchies[arg_name]  # Get levels for this argument.
            if h is not None:
                assert len(h) != 0
                self.set_dict_by_hierarchy(self.all_args, h, arg_val,
                                           nonexist_ok=nonexist_ok)

    def update_config_dict(self, old_args_dict, new_args_dict, levels=None):
        """Recursively update the dictionary of all parameters based on the dict read from config file."""
        if levels is None:
            levels = []
        if isinstance(new_args_dict, dict):
            # Go to the next dict level.
            for k in new_args_dict:
                self.update_config_dict(old_args_dict, new_args_dict[k],
                                        levels=levels + [k])
        else:
            # Reached the leaf level. Set the corresponding key.
            self.set_dict_by_hierarchy(old_args_dict, levels, new_args_dict,
                                       nonexist_ok=False)

    def dump_config(self, args_dict, level=[], out_to_doc=False, show_help=False, omit_defaults=False, omit_keys=[]):
        """Generate a config file based on args_dict with help information."""
        ret_string = ''
        for key, val in args_dict.items():
            if level == ["solver", "nlp"] and key in ("controller", "crown_engine_sigma_main") and val is False:
                continue  # preserve the default-off configuration transcript
            if level == ["solver", "nlp"] and key in self.NLP_GAP_DEFAULTS and type(val) is type(self.NLP_GAP_DEFAULTS[key]) \
                    and val == self.NLP_GAP_DEFAULTS[key]:
                continue  # schema gap key at the code's default: keep the transcript of 62094fd6e
            if level == ["model"] and key == "op_budget" and val == self.OP_BUDGET_DEFAULTS:
                continue  # all four keys at their defaults: keep the transcript of 62094fd6e
            if level == ["bab"] and key == "solve_deadline" and val is None:
                continue  # the GEAR facade's deadline, unset: keep the transcript of 62094fd6e
            if isinstance(val, dict):
                ret = self.dump_config(val, level + [key], out_to_doc, show_help, omit_defaults, omit_keys)
                if len(ret) > 0:
                    # Next level is not empty, print it.
                    ret_string += ' ' * (len(level) * 2) + f'{key}:\n' + ret
            else:
                if omit_defaults:
                    default_value_key = list(self.config_file_hierarchies.keys())[
                        list(self.config_file_hierarchies.values()).index(level + [key])
                    ]
                    if default_value_key in omit_keys:
                        continue
                    if self.defaults_parser.get_default(default_value_key) == val:
                        continue
                if show_help:
                    h = self.help_messages[','.join(level + [key])]
                    if 'debug' in key or 'not use' in h or 'not be use' in h or 'debug' in h or len(h) == 0:
                        # Skip some debugging options.
                        continue
                    h = f'  # {h}'
                else:
                    h = ''
                yaml_line = yaml.safe_dump({key: val}, default_flow_style=None).strip().replace('{', '').replace('}',
                                                                                                                 '')
                ret_string += ' ' * (len(level) * 2) + f'{yaml_line}{h}\n'
        if len(level) > 0:
            return ret_string
        else:
            # Top level, output to file.
            if out_to_doc:
                output_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs',
                                           'abcrown_all_params.yaml')
                with open(output_name, 'w') as f:
                    f.write(ret_string)
            return ret_string

    def parse_config(self, args=None, verbose=True):
        """
        Main function to parse parameter configurations. The commandline arguments have the highest priority;
        then the parameters specified in yaml config file. If a parameter does not exist in either commandline
        or the yaml config file, we use the defaults defined in add_common_options() defined above.
        """
        if args is None:
            args = sys.argv[1:]
        # Create the dictionary of all parameters, all set to their default values.
        self.construct_config_dict(self.default_args)
        # These are arguments specified in command line.
        specified_args = vars(self.no_defaults_parser.parse_args(args))
        if 'root_path' in specified_args and 'config' not in specified_args:
            specified_args['config'] = os.path.join(specified_args['root_path'], 'config.yaml')
        # Read the yaml config files.
        if 'config' in specified_args:
            self.file = specified_args['config']
            with open(self.file, 'r') as file:
                loaded_args = yaml.safe_load(file)
                # Update the defaults with the parameters in the config file.
                self.update_config_dict(self.all_args, loaded_args)
        # Finally, override the parameters based on commandline arguments.
        self.construct_config_dict(specified_args, nonexist_ok=False)
        # For compatibility, we still return all the arguments from argparser.
        parsed_args = self.defaults_parser.parse_args(args)
        self.update_arguments()
        # Print all configuration.
        if verbose:
            print('Configurations:\n')
            print(self.dump_config(self.all_args))
        if self.all_args["debug"]["save_minimal_config"] is not None:
            with open(self.all_args["debug"]["save_minimal_config"], 'w') as f:
                f.write(self.dump_config(self.all_args, omit_defaults=True, omit_keys=self.all_args["debug"]["save_minimal_config_omit_keys"]))
        return parsed_args

    def keys(self):
        return self.all_args.keys()

    def items(self):
        return self.all_args.items()

    def __getitem__(self, key):
        """Read an item from the dictionary of parameters."""
        return self.all_args[key]

    def __setitem__(self, key, value):
        """Set an item from the dictionary of parameters."""
        self.all_args[key] = value


class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("You must register a global parameter in arguments.py.")

    def __setitem__(self, key, value):
        if key not in self:
            raise RuntimeError("You must register a global parameter in arguments.py.")
        else:
            super().__setitem__(key, value)

    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


# Global configuration variable
Config = ConfigHandler()
# Global variables
Globals = ReadOnlyDict({
    # Tightest threshold ever used to discard a domain. Everything discarded satisfied
    # lb > threshold, so its true minimum lies above it; the reported global lower bound
    # must be capped by this or it stops being a bound on the original problem once the
    # domain holding the optimum is discarded.
    "discarded_threshold_min": None,
    "starting_timestamp": int(time.time()),
    "example_idx": -1,
    "sanity_check_offset": None,
    "out": {"idx": None, "pred": None, "attack_margin": None, "pred_adv": None,
            "init_crown_bounds": None, "init_alpha_crown": None,
            "refined_lb": None, "decisions": [], "results": 'timeout',
            "time": None, "domains_visited": None}})


if __name__ == '__main__':
    Config.construct_config_dict(Config.default_args)
    Config.dump_config(Config.all_args, out_to_doc=True, show_help=True)
