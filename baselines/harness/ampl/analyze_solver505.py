#!/usr/bin/env python3
"""Parse progress and primal-feasibility results from solver505 logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = EXPERIMENT_DIR / "runs"
SOLVERS = [
    "baron",
    "couenne",
    "ipopt",
    "knitro",
    "lgo",
    "lindoglobal",
    "minos",
    "scip",
    "snopt",
]
GLOBAL_SOLVERS = ["baron", "couenne", "lgo", "lindoglobal", "scip"]
LOCAL_SOLVERS = ["ipopt", "knitro", "minos", "snopt"]
NATIVE_CERTIFICATE_SOLVERS = ["baron", "couenne", "lindoglobal", "scip"]
OFFICIAL_REFERENCE = (
    EXPERIMENT_DIR.parent / "reference/minlplib_official_reference_240.tsv"
)
COMPARISON_FEASIBILITY_FIELD = "feasible_1e8"
OBJECTIVE_TIE_REL_TOL = 1e-4
VERIFIED_DUAL_GAP_REL_TOL = 1e-6
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_kv(line: str) -> dict[str, str]:
    return dict(KV_RE.findall(line))


def last_line(text: str, prefix: str) -> str:
    found = [line for line in text.splitlines() if line.startswith(prefix)]
    return found[-1] if found else ""


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def objective_sign(sense: str) -> float:
    if sense in {"min", "minimize"}:
        return 1.0
    if sense in {"max", "maximize"}:
        return -1.0
    raise ValueError(f"Unsupported objective sense: {sense!r}")


def signed_relative_quality(
    objective: float,
    reference: float,
    sense: str,
) -> float:
    """Negative improves the reference; positive is worse."""
    return (
        objective_sign(sense)
        * (objective - reference)
        / max(1.0, abs(reference))
    )


def comparison_relation(signed_quality: float) -> str:
    if signed_quality < -OBJECTIVE_TIE_REL_TOL:
        return "improves_current_global_incumbent"
    if signed_quality > OBJECTIVE_TIE_REL_TOL:
        return "worse_than_current_global_incumbent"
    return "matches_current_global_incumbent"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_official_reference() -> dict[str, dict[str, object]]:
    if not OFFICIAL_REFERENCE.is_file():
        return {}
    with OFFICIAL_REFERENCE.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    return {
        record["problem"]: {
            "sense": record["objsense"],
            "official_primal": as_float(record.get("official_primal")),
            "official_dual": as_float(record.get("official_dual")),
            "official_gap": as_float(record.get("official_gap")),
            "source_url": record.get("source_url", ""),
        }
        for record in records
    }


def last_regex_float(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return as_float(matches[-1]) if matches else None


def native_global_certificate(
    solver: str,
    attempt_text: str,
    solve_result: str | None,
    strict_feasible: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "native_global_certificate": False,
        "native_certificate_type": "",
        "native_certificate_gap": None,
    }
    if (
        solver not in NATIVE_CERTIFICATE_SOLVERS
        or solve_result != "solved"
        or not strict_feasible
    ):
        return result

    if solver == "baron":
        certified = bool(
            re.search(
                r"optimal within tolerances",
                attempt_text,
                flags=re.IGNORECASE,
            )
        )
        result.update(
            {
                "native_global_certificate": certified,
                "native_certificate_type": (
                    "BARON optimal-within-tolerances termination"
                    if certified
                    else ""
                ),
            }
        )
        return result

    if solver == "couenne":
        lower = last_regex_float(
            attempt_text, rf"^\s*Lower bound:\s*({NUMBER_PATTERN})"
        )
        upper = last_regex_float(
            attempt_text, rf"^\s*Upper bound:\s*({NUMBER_PATTERN})"
        )
        gap = (
            abs(upper - lower) / max(1.0, abs(upper), abs(lower))
            if lower is not None and upper is not None
            else None
        )
        certified = (
            "couenne: Optimal" in attempt_text
            and gap is not None
            and gap <= VERIFIED_DUAL_GAP_REL_TOL
        )
        result.update(
            {
                "native_global_certificate": certified,
                "native_certificate_type": (
                    "Couenne closed lower-upper bound"
                    if certified
                    else ""
                ),
                "native_certificate_gap": gap,
            }
        )
        return result

    if solver == "lindoglobal":
        gap = last_regex_float(
            attempt_text, rf"Rel\.\s*GOP Gap\s*:\s*({NUMBER_PATTERN})"
        )
        certified = (
            "Global optimum found" in attempt_text
            and gap is not None
            and abs(gap) <= VERIFIED_DUAL_GAP_REL_TOL
        )
        result.update(
            {
                "native_global_certificate": certified,
                "native_certificate_type": (
                    "LindoGlobal closed GOP bound"
                    if certified
                    else ""
                ),
                "native_certificate_gap": abs(gap) if gap is not None else None,
            }
        )
        return result

    primal = last_regex_float(
        attempt_text, rf"^\s*Primal Bound\s*:\s*({NUMBER_PATTERN})"
    )
    dual = last_regex_float(
        attempt_text, rf"^\s*Dual Bound\s*:\s*({NUMBER_PATTERN})"
    )
    gap = (
        abs(primal - dual) / max(1.0, abs(primal), abs(dual))
        if primal is not None and dual is not None
        else None
    )
    certified = (
        bool(
            re.search(
                r"problem is solved \[optimal solution found\]",
                attempt_text,
                flags=re.IGNORECASE,
            )
        )
        and gap is not None
        and gap <= VERIFIED_DUAL_GAP_REL_TOL
    )
    result.update(
        {
            "native_global_certificate": certified,
            "native_certificate_type": (
                "SCIP closed primal-dual bound" if certified else ""
            ),
            "native_certificate_gap": gap,
        }
    )
    return result


def best_objective(
    candidates: list[dict[str, object]],
    sense: str,
) -> dict[str, object] | None:
    feasible = [
        row
        for row in candidates
        if bool(row[COMPARISON_FEASIBILITY_FIELD])
        and isinstance(row.get("objective"), float)
    ]
    if not feasible:
        return None
    key = lambda row: objective_sign(sense) * float(row["objective"])
    return min(feasible, key=key)


def build_local_global_comparison(
    rows: list[dict[str, object]],
    manifest: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key = {
        (str(row["family"]), str(row["instance"]), str(row["solver"])): row
        for row in rows
    }
    official = read_official_reference()
    comparison_rows: list[dict[str, object]] = []
    for record in manifest:
        family = record["family"]
        instance = record["instance"]
        sense = record["sense"]
        global_candidates = [
            by_key[(family, instance, solver)] for solver in GLOBAL_SOLVERS
        ]
        global_best = best_objective(global_candidates, sense)
        global_objective = (
            float(global_best["objective"]) if global_best is not None else None
        )
        global_solver = (
            str(global_best["solver"]) if global_best is not None else ""
        )
        reference = official.get(instance) if family == "minlplib_linear" else None
        for solver in LOCAL_SOLVERS:
            local = by_key[(family, instance, solver)]
            local_feasible = bool(local[COMPARISON_FEASIBILITY_FIELD])
            local_objective = (
                float(local["objective"])
                if local_feasible and isinstance(local.get("objective"), float)
                else None
            )
            signed_quality = (
                signed_relative_quality(local_objective, global_objective, sense)
                if local_objective is not None and global_objective is not None
                else None
            )
            if local_objective is None:
                relation = "no_strict_feasible_local_primal"
            elif global_objective is None:
                relation = "no_strict_feasible_global_incumbent"
            else:
                relation = comparison_relation(signed_quality)

            official_dual = (
                float(reference["official_dual"])
                if reference is not None
                and isinstance(reference.get("official_dual"), float)
                else None
            )
            trusted_dual_gap = (
                objective_sign(sense)
                * (local_objective - official_dual)
                / max(1.0, abs(local_objective), abs(official_dual))
                if local_objective is not None and official_dual is not None
                else None
            )
            matches_verified_dual = (
                trusted_dual_gap is not None
                and -VERIFIED_DUAL_GAP_REL_TOL
                <= trusted_dual_gap
                <= VERIFIED_DUAL_GAP_REL_TOL
            )
            comparison_rows.append(
                {
                    "family": family,
                    "instance": instance,
                    "sense": sense,
                    "local_solver": solver,
                    "local_strict_feasible": local_feasible,
                    "local_objective": local_objective,
                    "global_incumbent_available": global_best is not None,
                    "global_incumbent_objective": global_objective,
                    "global_incumbent_solver": global_solver,
                    "signed_relative_to_global_incumbent": signed_quality,
                    "relation_to_global_incumbent": relation,
                    "official_dual": official_dual,
                    "signed_relative_gap_to_official_dual": trusted_dual_gap,
                    "matches_verified_official_dual_1e6": matches_verified_dual,
                    "terminology": (
                        "verified_reference_match"
                        if matches_verified_dual
                        else "feasible_primal_only"
                    ),
                }
            )

    summary: dict[str, object] = {}
    for solver in LOCAL_SOLVERS:
        selected = [
            row for row in comparison_rows if row["local_solver"] == solver
        ]
        paired = [
            row
            for row in selected
            if row["local_strict_feasible"]
            and row["global_incumbent_available"]
        ]
        relations = Counter(
            str(row["relation_to_global_incumbent"]) for row in paired
        )
        losses = [
            max(0.0, float(row["signed_relative_to_global_incumbent"]))
            for row in paired
            if isinstance(row.get("signed_relative_to_global_incumbent"), float)
        ]
        summary[solver] = {
            "strict_feasible_local_primals": sum(
                bool(row["local_strict_feasible"]) for row in selected
            ),
            "paired_with_strict_feasible_global_incumbent": len(paired),
            "improves_current_global_incumbent": relations.get(
                "improves_current_global_incumbent", 0
            ),
            "matches_current_global_incumbent": relations.get(
                "matches_current_global_incumbent", 0
            ),
            "worse_than_current_global_incumbent": relations.get(
                "worse_than_current_global_incumbent", 0
            ),
            "strict_feasible_without_global_incumbent": sum(
                bool(row["local_strict_feasible"])
                and not bool(row["global_incumbent_available"])
                for row in selected
            ),
            "matches_verified_official_dual_1e6": sum(
                bool(row["matches_verified_official_dual_1e6"])
                for row in selected
            ),
            "nonnegative_loss_to_global_incumbent": {
                "median": statistics.median(losses) if losses else None,
                "p90": percentile(losses, 0.90),
                "within_1e-4": sum(value <= 1e-4 for value in losses),
                "within_1e-3": sum(value <= 1e-3 for value in losses),
                "within_1e-2": sum(value <= 1e-2 for value in losses),
            },
        }
    return comparison_rows, summary


def parse_log(path: Path, solver: str) -> dict[str, object]:
    if not path.is_file():
        return {
            "present": False,
            "runner_status": "missing",
            "objective": None,
            "max_violation": None,
            "feasible_1e8": False,
            "feasible_1e6": False,
            "native_global_certificate": False,
            "native_certificate_type": "",
            "native_certificate_gap": None,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    runner = parse_kv(last_line(text, "RUNNER_RESULT "))
    harness = parse_kv(last_line(text, "HARNESS_RESULT "))
    objective = as_float(harness.get("objective"))
    con_viol = as_float(harness.get("constraint_violation"))
    var_viol = as_float(harness.get("variable_violation"))
    violations = [v for v in (con_viol, var_viol) if v is not None]
    max_violation = max(violations) if len(violations) == 2 else None
    runner_status = runner.get("status", "missing_terminal")
    accepted = runner_status in {"completed", "completed_after_interrupt"}
    strict_feasible = (
        accepted
        and objective is not None
        and max_violation is not None
        and max_violation <= 1e-8
    )
    attempt_text = text.rsplit("=== ATTEMPT", 1)[-1]
    certificate = native_global_certificate(
        solver,
        attempt_text,
        harness.get("solve_result"),
        strict_feasible,
    )
    return {
        "present": True,
        "runner_status": runner_status,
        "exit_code": runner.get("exit_code"),
        "timed_out": runner.get("timed_out"),
        "wall_time": as_float(runner.get("wall_time")),
        "solve_result": harness.get("solve_result"),
        "solve_result_num": harness.get("solve_result_num"),
        "objective": objective,
        "constraint_violation": con_viol,
        "variable_violation": var_viol,
        "max_violation": max_violation,
        "feasible_1e8": strict_feasible,
        "feasible_1e6": (
            accepted
            and objective is not None
            and max_violation is not None
            and max_violation <= 1e-6
        ),
        **certificate,
        "log": str(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.runs / "manifest.tsv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    official = read_official_reference()
    rows: list[dict[str, object]] = []
    for solver in SOLVERS:
        for record in manifest:
            path = (
                args.runs
                / "logs"
                / solver
                / record["family"]
                / f"{record['instance']}.log"
            )
            parsed = parse_log(path, solver)
            reference = (
                official.get(record["instance"])
                if record["family"] == "minlplib_linear"
                else None
            )
            official_dual = (
                float(reference["official_dual"])
                if reference is not None
                and isinstance(reference.get("official_dual"), float)
                else None
            )
            objective = (
                float(parsed["objective"])
                if isinstance(parsed.get("objective"), float)
                else None
            )
            trusted_dual_gap = (
                objective_sign(record["sense"])
                * (objective - official_dual)
                / max(1.0, abs(objective), abs(official_dual))
                if bool(parsed["feasible_1e8"])
                and objective is not None
                and official_dual is not None
                else None
            )
            matches_official_dual = (
                trusted_dual_gap is not None
                and -VERIFIED_DUAL_GAP_REL_TOL
                <= trusted_dual_gap
                <= VERIFIED_DUAL_GAP_REL_TOL
            )
            rows.append(
                {
                    "solver": solver,
                    "family": record["family"],
                    "instance": record["instance"],
                    "sense": record["sense"],
                    "nvars": record["nvars"],
                    "ncons": record["ncons"],
                    "official_dual": official_dual,
                    "signed_relative_gap_to_official_dual": trusted_dual_gap,
                    "matches_verified_official_dual_1e6": (
                        matches_official_dual
                    ),
                    **parsed,
                }
            )

    summary: dict[str, object] = {
        "expected_instances": len(manifest),
        "expected_records": len(manifest) * len(SOLVERS),
        "solvers": {},
        "solver_classes": {
            "global": GLOBAL_SOLVERS,
            "local": LOCAL_SOLVERS,
            "native_global_certificate_capable": (
                NATIVE_CERTIFICATE_SOLVERS
            ),
            "global_search_primal_without_native_certificate": ["lgo"],
        },
        "terminology_policy": {
            "global_incumbent": (
                "best AMPL-audited strict-feasible primal returned by a global "
                "solver; this is not called an optimum"
            ),
            "verified_reference_match": (
                "AMPL-audited at 1e-8 and within normalized 1e-6 of the "
                "published trusted MINLPLib dual"
            ),
            "optimal": (
                "never inferred from a local solve, a best incumbent, or a "
                "solver's solved label without an independently auditable "
                "primal-dual certificate"
            ),
            "native_global_certificate": (
                "strict-feasible returned primal plus an auditable solver-native "
                "closed primal-dual/global bound at normalized gap <=1e-6"
            ),
        },
    }
    for solver in SOLVERS:
        selected = [row for row in rows if row["solver"] == solver]
        status_counts = Counter(str(row["runner_status"]) for row in selected)
        summary["solvers"][solver] = {
            "logs_present": sum(bool(row["present"]) for row in selected),
            "terminal_records": sum(
                str(row["runner_status"]) not in {"missing", "missing_terminal"}
                for row in selected
            ),
            "strict_feasible": sum(bool(row["feasible_1e8"]) for row in selected),
            "loose_feasible": sum(bool(row["feasible_1e6"]) for row in selected),
            "native_global_certificates": sum(
                bool(row["native_global_certificate"]) for row in selected
            ),
            "matches_verified_official_dual_1e6": sum(
                bool(row["matches_verified_official_dual_1e6"])
                for row in selected
            ),
            "status_counts": dict(sorted(status_counts.items())),
        }

    certificate_rows: list[dict[str, object]] = []
    for record in manifest:
        selected = [
            row
            for row in rows
            if row["family"] == record["family"]
            and row["instance"] == record["instance"]
        ]
        native = [
            str(row["solver"])
            for row in selected
            if bool(row["native_global_certificate"])
        ]
        official_matches = [
            str(row["solver"])
            for row in selected
            if bool(row["matches_verified_official_dual_1e6"])
        ]
        certificate_rows.append(
            {
                "family": record["family"],
                "instance": record["instance"],
                "sense": record["sense"],
                "native_certificate_available": bool(native),
                "native_certificate_solvers": ",".join(native),
                "official_dual_match_available": bool(official_matches),
                "official_dual_match_solvers": ",".join(official_matches),
                "verified_global_optimum": bool(native or official_matches),
                "verification_sources": ",".join(
                    source
                    for source, present in (
                        ("solver_native_primal_dual", bool(native)),
                        ("minlplib_trusted_dual", bool(official_matches)),
                    )
                    if present
                ),
            }
        )
    summary["global_optimal_certification"] = {
        "native_certificate_gap_tolerance": VERIFIED_DUAL_GAP_REL_TOL,
        "native_certificate_capable_solvers": (
            NATIVE_CERTIFICATE_SOLVERS
        ),
        "lgo_treatment": (
            "global-search primal only; its AMPL solved statuses do not expose "
            "a closed dual bound and are not counted as global-optimal proofs"
        ),
        "instances_with_native_certificate": sum(
            bool(row["native_certificate_available"])
            for row in certificate_rows
        ),
        "instances_matching_minlplib_trusted_dual": sum(
            bool(row["official_dual_match_available"])
            for row in certificate_rows
        ),
        "verified_global_optimum_instances": sum(
            bool(row["verified_global_optimum"]) for row in certificate_rows
        ),
        "verified_by_family": {
            family: sum(
                row["family"] == family
                and bool(row["verified_global_optimum"])
                for row in certificate_rows
            )
            for family in ("gams_world", "minlplib_linear")
        },
        "per_solver": {
            solver: {
                "native_certificates": summary["solvers"][solver][
                    "native_global_certificates"
                ],
                "matches_minlplib_trusted_dual": summary["solvers"][solver][
                    "matches_verified_official_dual_1e6"
                ],
            }
            for solver in SOLVERS
        },
    }

    comparison_rows, comparison_summary = build_local_global_comparison(
        rows, manifest
    )
    summary["local_vs_global"] = {
        "comparison_feasibility_tolerance": 1e-8,
        "objective_tie_relative_tolerance": OBJECTIVE_TIE_REL_TOL,
        "verified_official_dual_relative_tolerance": (
            VERIFIED_DUAL_GAP_REL_TOL
        ),
        "global_ensemble_complete": all(
            summary["solvers"][solver]["terminal_records"] == len(manifest)
            for solver in GLOBAL_SOLVERS
        ),
        "global_ensemble_strict_feasible_instances": len(
            {
                (row["family"], row["instance"])
                for row in comparison_rows
                if row["global_incumbent_available"]
            }
        ),
        "local_solvers": comparison_summary,
    }

    out_dir = args.runs / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (out_dir / "per_instance.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    comparison_fields = list(
        dict.fromkeys(key for row in comparison_rows for key in row)
    )
    with (out_dir / "local_vs_global.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=comparison_fields, delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    certificate_fields = list(certificate_rows[0])
    with (out_dir / "global_optimal_certificates.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=certificate_fields, delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(certificate_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "solver\tlogs\tterminal\tfeasible@1e-8\tfeasible@1e-6\tstatuses"
    )
    for solver in SOLVERS:
        item = summary["solvers"][solver]
        print(
            f"{solver}\t{item['logs_present']}\t{item['terminal_records']}\t"
            f"{item['strict_feasible']}\t{item['loose_feasible']}\t"
            f"{item['status_counts']}"
        )
    certification = summary["global_optimal_certification"]
    print(
        "global_optimal_certification\t"
        f"native_union={certification['instances_with_native_certificate']}\t"
        f"official_dual_union="
        f"{certification['instances_matching_minlplib_trusted_dual']}\t"
        f"verified_union="
        f"{certification['verified_global_optimum_instances']}\t"
        f"by_family={certification['verified_by_family']}"
    )
    print("solver\tnative_global_certificates\tofficial_dual_matches")
    for solver in SOLVERS:
        item = certification["per_solver"][solver]
        print(
            f"{solver}\t{item['native_certificates']}\t"
            f"{item['matches_minlplib_trusted_dual']}"
        )
    ensemble = summary["local_vs_global"]
    print(
        "local_vs_global\t"
        f"global_complete={ensemble['global_ensemble_complete']}\t"
        f"global_strict_coverage="
        f"{ensemble['global_ensemble_strict_feasible_instances']}"
    )
    print(
        "local_solver\tstrict_feasible\tpaired_global\timproves\tmatches\t"
        "worse\tverified_official_dual"
    )
    for solver in LOCAL_SOLVERS:
        item = ensemble["local_solvers"][solver]
        print(
            f"{solver}\t{item['strict_feasible_local_primals']}\t"
            f"{item['paired_with_strict_feasible_global_incumbent']}\t"
            f"{item['improves_current_global_incumbent']}\t"
            f"{item['matches_current_global_incumbent']}\t"
            f"{item['worse_than_current_global_incumbent']}\t"
            f"{item['matches_verified_official_dual_1e6']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
