#!/usr/bin/env python3
"""Generate every candidate configuration (configs/candidates/<solver>/<name>.json).

One global configuration per solver; a candidate only differs in solver options,
native settings, and the jobs x threads layout (jobs*threads = 16).
Tuning stages: `settings` candidates run with the solver's default layout;
`layout` candidates are derived from the stage-1 winner by campaign.py.
"""
import json
from pathlib import Path

CFG = Path(__file__).resolve().parents[1] / "configs" / "candidates"
OMP = lambda n: {k: str(n) for k in (  # noqa: E731
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def put(solver, name, options, native="", threads=1, jobs=16, notes="",
        stage="settings", select=True, **kw):
    d = dict(name=name, solver=solver, options=options, native=native,
             threads=threads, jobs=jobs, env=OMP(threads), notes=notes,
             stage=stage, select=select, **kw)
    (CFG / solver).mkdir(parents=True, exist_ok=True)
    (CFG / solver / f"{name}.json").write_text(json.dumps(d, indent=2) + "\n")


# ======= single standard (coordinator, 2026-10-12): feasibility 1e-6, rel gap 1e-4,
# abs gap 1e-5 set as every solver's own termination tolerances; other options tuned on top.
# ---------------------------------------------------------------- BARON
BO = "maxtime={time} threads=1 outlev=1 epsr=1e-4 epsa=1e-5"
B0 = "AbsConFeasTol 1e-6\n"
put("baron", "b0_std", BO, B0, notes="standard tolerances only (EpsR 1e-4, EpsA 1e-5, AbsConFeasTol 1e-6)")
b1 = B0 + "NumLoc 20\n"
put("baron", "b1_loc20", BO, b1, notes="std + 20 local searches (prior tuning)")
put("baron", "b2_probe", BO, b1 + "PDo 10\n", notes="b1 + probing (bound-tightening intensity up)")
put("baron", "b3_nobt", BO, b1 + "OBTTDo 0\nLBTTDo 0\n",
    notes="b1 + LP-based/optimality-based tightening off (intensity down)")
put("baron", "b4_nodesel", BO, b1 + "NodeSel 1\n", notes="b1 + best-bound node selection")

# ---------------------------------------------------------------- Couenne
CO = "bonmin.time_limit={time} outlev=1"
C0 = "feas_tolerance 1e-6\nbonmin.allowable_fraction_gap 1e-4\nbonmin.allowable_gap 1e-5\n"
put("couenne", "c0_std", CO, C0, notes="standard tolerances (paper rule: OBBT/aggr FBBT off above 100 vars)")
c1 = C0 + "log_num_local_optimization_per_level 4\n"
put("couenne", "c1_lo4", CO, c1, notes="std + 4 local optimizations per level (prior tuning)")
put("couenne", "c2_bt_off", CO, c1 + "optimality_bt no\naggressive_fbbt no\n",
    couenne_rule="none", notes="c1 with OBBT/aggressive FBBT off for all sizes")
put("couenne", "c3_bt_on", CO, c1 + "optimality_bt yes\naggressive_fbbt yes\n",
    couenne_rule="none", notes="c1 with OBBT/aggressive FBBT on for all sizes")
put("couenne", "c4_branch", CO, c1 + "branch_pt_select balanced\n", notes="c1 + balanced branching point")

# ---------------------------------------------------------------- LindoGlobal
LO = "maxtime={time} threads={threads} outlev=1 epsr=1e-4 epsa=1e-5"
put("lindoglobal", "l0_std", LO, notes="standard gaps, single thread (driver exposes no other tuning options)")
put("lindoglobal", "l9_mt_recheck", LO, threads=16, jobs=1, select=False, stage="recheck",
    notes="re-check of the 16-thread mode with the new license (evidence only: stays single-threaded by rule)")

# ---------------------------------------------------------------- SCIP
SO = ("lim:time={time} lp:threads=1 tech:outlev=1 tech:timing=2 alg:concurrent=1 "
      "par:minnthreads={threads} par:maxnthreads={threads} par:mode=0")
std = "numerics/feastol = 1e-6\nlimits/gap = 1e-4\nlimits/absgap = 1e-5\n"
rlt = std + ("separating/rlt/freq = 10\nseparating/rlt/detecthidden = TRUE\n"
             "separating/rlt/onlyoriginal = FALSE\n")
put("scip", "s0_std", SO, std, threads=16, jobs=1, notes="standard tolerances, 16-way concurrent")
put("scip", "s1_rlt", SO, rlt, threads=16, jobs=1, notes="std + RLT cuts (prior tuning)")
put("scip", "s2_obbt", SO, rlt + "propagating/obbt/onlynonconvexvars = FALSE\n"
    "propagating/obbt/freq = 10\n", threads=16, jobs=1, notes="s1 + OBBT in the tree (bound tightening up)")
put("scip", "s3_heur", SO, rlt + "heuristics/multistart/freq = 10\nheuristics/nlpdiving/freq = 1\n",
    threads=16, jobs=1, notes="s1 + multistart heuristic and NLP diving every node (local-solver heuristics)")
put("scip", "s4_branch", SO, rlt + "branching/midpull = 0.5\n", threads=16, jobs=1,
    notes="s1 + branching midpoint pull 0.5")
sets = {
    1: rlt, 2: rlt + "branching/midpull = 0.5\n",
    3: rlt + "propagating/obbt/onlynonconvexvars = FALSE\npropagating/obbt/freq = 10\n",
    4: rlt + "heuristics/subnlp/freq = 1\nheuristics/multistart/freq = 10\n",
    5: rlt + "branching/scorefac = 0.9\n", 6: rlt + "separating/rlt/freq = 1\n",
    7: std, 8: rlt + "nlhdlr/convex/enabled = TRUE\n",
}
put("scip", "s5_portfolio", SO, rlt, threads=16, jobs=1, pfx={f"scip-{i}.set": t for i, t in sets.items()},
    notes="s1 base + per-thread parameter-set portfolio (threads 1-8 distinct, 9-16 the s1 base with new seeds)")

# ---------------------------------------------------------------- Gurobi
GT = " mip:gap=1e-4 mip:gapabs=1e-5 alg:feastol=1e-6"
GB = "lim:time={time} tech:threads={threads} tech:outlev=1" + GT
GN = GB + " pre:funcnonlinear=1 qp:nonconvex=2"
put("gurobi", "g0_std", GB, threads=16, jobs=1, notes="standard tolerances, automatic function handling, 16 threads")
put("gurobi", "g1_nonlin", GN, threads=16, jobs=1, notes="std + true nonlinear/global handling (funcnonlinear=1, nonconvex=2)")
put("gurobi", "g2_focus3", GN + " mip:focus=3", threads=16, jobs=1, notes="g1 + MIPFocus 3 (bound)")
put("gurobi", "g3_pre_cuts", GN + " pre:solve=2 cut:cuts=2", threads=16, jobs=1, notes="g1 + aggressive presolve and cuts")
put("gurobi", "g4_obbt", GN + " mip:obbt=3", threads=16, jobs=1, notes="g1 + OBBT aggressiveness 3")
put("gurobi", "g5_heur", GN + " alg:nlpheur=3 mip:heurfrac=0.2", threads=16, jobs=1,
    notes="g1 + aggressive NLP heuristic, 20% heuristic time")

# ---------------------------------------------------------------- Xpress (global)
XT = " mip:gap=1e-4 mip:gapabs=1e-5 alg:feastol=1e-6"
XB = "lim:time={time} tech:threads={threads} tech:outlev=1" + XT
XG = XB + " alg:nlpsolver=2"
put("xpress", "x0_std", XB, threads=16, jobs=1, notes="standard tolerances, solver-chosen global handling, 16 threads")
put("xpress", "x1_global", XG, threads=16, jobs=1, notes="std + explicit Xpress Global")
put("xpress", "x2_obbt3", XG + " global:xprs_globalpresolveobbt=3", threads=16, jobs=1, notes="x1 + OBBT on all variables")
put("xpress", "x3_cuts", XG + " cut:rltcuts=1 global:xprs_globalspatialbranchcuttingeffort=0.5",
    threads=16, jobs=1, notes="x1 + RLT cuts and more branching cuts")
put("xpress", "x4_heur", XG + " global:xprs_globallsheurstrategy=3 heur:xslp_heurstrategy=3",
    threads=16, jobs=1, notes="x1 + aggressive local-solver heuristics")
put("xpress", "x5_nobt", XG + " global:xprs_globalpresolveobbt=0 pre:solve_nlp=1",
    threads=16, jobs=1, notes="x1 + OBBT off, NLP presolve on")

# ---------------------------------------------------------------- local solvers: prior tuned options,
# feasibility tolerances set to the standard 1e-6; not tuned again
put("ipopt", "ipopt_final",
    "max_wall_time={time} print_level=5 timing_statistics=yes honor_original_bounds=yes "
    "constr_viol_tol=1e-6 acceptable_constr_viol_tol=1e-6 bound_relax_factor=0 max_iter=100000",
    stage="final", notes="prior tuned options, constr_viol_tol 1e-6; 16 serial jobs")
KN = ("maxtime={time} numthreads={threads} nlp_algorithm=4 ms_enable=1 ms_numthreads={threads} "
      "ms_deterministic=0 ms_maxsolves=200 ms_terminate=0 concurrent_evals=1 soltype=1 outlev=2 "
      "timing=1 feastol=1e-6 feastol_abs=1e-6")
put("knitro", "knitro_final", KN, stage="final", notes="prior tuned (SQP + multistart), feastol 1e-6; 16 serial jobs")
put("knitro", "knitro_ms16", KN, threads=16, jobs=1, stage="final-alt", select=False,
    notes="native multistart with 16 threads, one instance at a time (alternative layout)")
put("snopt", "snopt_final",
    "outlev=1 timing=1 major_feasibility_tolerance=1e-6 minor_feasibility_tolerance=1e-6",
    stage="final", notes="feasibility 1e-6 (SNOPT default optimality)")
put("minos", "minos_final",
    "timelim={time} outlev=1 timing=1 feasibility_tolerance=1e-6 row_tolerance=1e-6 "
    "major_iterations=1000 minor_iterations=1000 iterations_limit=10000000 "
    "superbasics_limit=10000 scale_option=0 completion=full", stage="final",
    notes="prior tuned options, feasibility 1e-6")
put("lgo", "lgo_final",
    "timelim={time} outlev=1 timing=1 g_maxfct=10000000 maxnosuc=10000000 penmult=100 con_tol=1e-6",
    stage="final", notes="prior tuned options, con_tol 1e-6 (global-search primal only)")
put("conopt", "conopt_default", "maxftime={time}", stage="final",
    notes="CONOPT defaults (feasibility tolerance 1e-7 default)")
