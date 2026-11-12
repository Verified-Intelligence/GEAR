# Original-model solve followed by VA-PINN refinement

`gear.solve_dynamics` implements the paper's two-level procedure in the public
Python API. It first calls the existing GEAR solver on the supplied original
graph. An optimal incumbent is checked by reference-model simulation before it
is accepted. A proof of original infeasibility returns immediately.
For an ordinary unsolved or unsupported original graph,
the pipeline trains a verification-aware physics-informed neural network
(VA-PINN), calibrates constraint margins, solves the fixed surrogate program and
checks the optimization and validation criteria. Failed checks cause new
reference simulations and another training/optimization round.

The ordinary `gear.solve`, its defaults and its `Result` schema are unchanged.

## Run the complete example

The example models two discrete steps, `x[0]=0`, `x[k+1]=x[k]+z`, with control
`0 <= z <= 1`, objective `-x[2]` and constraint `x[2] <= 0.8`. Its solution is
`z=0.4`, objective `-0.8`.

```bash
export PYTHONPATH=$PWD/solver:$PWD/solver/complete_verifier
python solver/examples/dynamics.py --device cpu
python solver/examples/dynamics.py --device cpu --force-surrogate
```

The first command exercises L1. The second gives L1 a zero solve budget to
exercise L2 on this small example. Tests run both commands through the actual
GEAR solver and check the returned point, scope and validation records.

## One budget for the complete pipeline

Set `DynamicsOptions(total_time_limit=10000)` to share one wall-clock deadline
across original compilation/solve, reference simulations, all training rounds,
surrogate compilation/solve and independent validation. The default `None`
preserves the previous per-solve behavior. Each native solve receives the smaller
of its configured limit and the remaining shared time. Each built-in training
fit uses at most `training_time_fraction` (default 0.7) of the remaining time,
reserving time for optimization and validation; this is a scheduling allocation,
not an acceptance criterion. Completed Adam updates and training-budget stops
are recorded per round.

This deadline is cooperative. A single long native kernel or a caller callback
needs a process watchdog for strict wall-clock enforcement. Start that watchdog
before imports/problem construction if those must count toward the experiment.
GPU queue waiting precedes the experiment's clock. A separately timed L1 wall
allocation must also cover original compilation if that is the chosen protocol.

`on_progress(snapshot)` receives a JSON-compatible snapshot at phase boundaries.
An exhausted budget returns `accepted=False`, `budget_exhausted=True`, the last
phase, the original result and completed rounds. A native L2 result awaiting
validation is retained in `pending_surrogate`; it has no accepted certificate.
Bounds retain their original or surrogate scope. A validation check that has not
finished by the deadline never earns acceptance.

## Define a dynamics problem

```python
import gear as gr

problem = gr.DynamicsProblem(
    original=lambda options: gr.compile(
        original_graph, box=control_box, objective=0, le=constraint_indices,
        options=options),
    box=control_box,
    simulate=simulate_reference,
    initial_state=initial_state,
    outputs=objective_and_constraints,
    residual=discrete_residual,
)
result = gr.solve_dynamics(problem, gr.DynamicsOptions(
    total_time_limit=10000,
    original=gr.Options(device="cuda", time_limit=600),
    surrogate=gr.Options(device="cuda", time_limit=600),
))
```

The controls have a finite continuous box, including any fixed coordinates.
Callbacks accept batches of float64 tensors:

| Callback | Result |
|---|---|
| `simulate(z)` | reference trajectories `[B,T+1,S]` |
| `initial_state(z)` | exact initial states `[B,S]` |
| `outputs(trajectory,z)` | `[B,1+R]`, objective in column 0, constraint rows <= 0 |
| `residual(trajectory,z)` | discrete-equation residual `[B,T,S]`, in state units |

`simulate` runs on the CPU. `initial_state`, `outputs` and `residual` must also
support differentiable evaluation of predicted trajectories on the training
device. Registered buffers of callback modules are copied to that device;
caller-owned modules are not modified. A reference trajectory must agree with
the initial-state callback and satisfy the declared residual to
`reference_tolerance` (default 1e-6).

Set `sense=gr.MAX` for maximization; both graphs must use the same objective
sense. All returned bounds remain in that sense. If the original graph has
state inputs as well as controls, supply `project_original(result_x)` to recover
the controls for physical replay. The original graph must represent the
reference problem, or a known relaxation whose feasible incumbents can be
replayed on it; the adapter does not prove that relationship.

`original` may be an already `Compiled`/`Legacy` source or a factory taking the
L1 `Options`. A factory allows an unsupported original compilation to fall back
to L2. `original=None` explicitly records that no original graph was supplied;
it is not reported as an original-model solve. Malformed input and missing
artifacts remain errors. Interruptions do not start training. A native solver
`INTERNAL_ERROR` also stops immediately: `phase` is `original_error` or
`surrogate_error`, the failed native result is retained, and no subsequent
training/refinement is attempted. An implementation error is never treated as
an ordinary unsuccessful L1 solve. Correct the cause before restarting the
instance with a fresh experiment budget.

For DAEs with nonexistence, `exists(z,trajectory)` must mark finite placeholder
paths as false. These points are reference-infeasible; they cannot be accepted
as incumbents. `relevant_nodes(z,trajectory)` may return the future prefix to
use in trajectory-error metrics, for example through the first constraint
violation or failed time step. Without it, existing trajectories use the full
horizon and nonexistent trajectories have no valid prefix.

The paper's existing `Case` objects can be adapted directly:

```python
problem = gr.DynamicsProblem.from_case(case, original=original_factory)
result = gr.solve_dynamics(problem, options)
```

This reads the existing affine/nonlinear objective, optional exact control
objective and constraint groups. It uses the case's simulator, residual,
initial-state map and nonexistence/prefix callbacks. It imports no benchmark
package, checkpoint, optimum, solver result or machine-specific path.

## Training, calibration and refinement

The built-in `VAPINNTrainer` uses shared affine, quadratic and tanh features for
trajectory and functional output maps. Initial states are imposed exactly;
constant future coordinates are fixed from the initial training set. The loss
includes normalized trajectory error, discrete residual, functional prediction
error, consistency with F/G of the predicted trajectory, a tanh relaxation-slack
penalty and a sampled sub-box bound-gap penalty. Output maps are fitted by ridge
regression every 250 updates. There are three width-64 hidden layers, 150,000
initial updates and 20,000 updates per refinement by default. Complete quadratic
features are used for up to 64 controls; larger inputs use learned low-rank
squares. `VAPINNOptions` configures these choices.

Normalization is fixed from the initial training set (`4096 + 256d` samples by
default). Training sees that set and separate validation data. It never receives
either independent test set. Validation selects a checkpoint and calibrates each
nonnegative constraint margin as the 99th percentile of `G_true - G_hat`.
The solver then receives the fixed program `[F_hat, G_hat + margin]`.

Every solve has its own native `Options` budget. Simulation, training and
reference checks are recorded separately from the optimization budget. The
current GEAR API solves in float64; this implementation also trains in float64.
It is a new native implementation, not a byte-for-byte replay of the paper's
older float32 L2 experiment harness, branching presets or checkpoint recipes.

Failed rounds acquire reference simulations in open branch-and-bound boxes
and near available incumbents. When either is unavailable they sample the
global box. One fifth of new samples is held out; the rest augments training.
Each refinement starts from the initial saved model. The two independent test
sets remain fixed and never enter fitting or margin calibration.

`Solver.on_domains(callback, limit=32)` supplies independent CPU copies of up to
32 open boxes at the end of input branch-and-bound, in search order. Observing
does not remove domains or change the search. Root-only solves can supply no
boxes. Without a subscriber the solver does not copy or inspect these boxes.

An alternative trainer can implement
`fit(problem, training, validation, normalization, initial_model, round_index,
tau_f, device)` and be supplied as `trainer=`. The returned `nn.Module` must
provide `forward(z)->[F_hat,G_hat]` and `trajectory(z)->[B,T+1,S]`, be exportable
through the public compiler, and leave caller-owned data and `initial_model`
unchanged. The cascade owns test evaluation and acceptance independently of
this trainer.

## Acceptance and result scope

For L2, the optimization criterion requires a replayed surrogate incumbent,
a finite correctly oriented gap, reference-model feasibility, objective
prediction error <= `tau_f`, and no simulated reference-feasible point below
`D - tau_f` (directions are reversed for maximization). Surrogate infeasibility
alone cannot prove original infeasibility. A closed surrogate gap alone is not
enough to accept a result.

Both independent test sets must pass: median objective error <= `tau_f`, 95th
percentile error <= `5 tau_f`, median normalized trajectory error <= 1%, and
zero false feasible predictions. These are empirical checks, not a uniform
error bound or an original-model global-optimality certificate.

`DynamicsOptions(tolerance="standard")` sets `tau_f=.01 sigma_F` and uses it for
the surrogate gap. The default `"strict"` uses `tau_f=.001 sigma_F` and gap
1e-4. Physical feasibility defaults to 1e-6. `max_refinements=6` means at most
seven models including the initial one.

`DynamicsResult` retains:

- `original` and `original_validation`: the native L1 result and physical replay;
- every round's own native `result`, compiled hash, margins, checks and timings;
- `accepted`, `level` (1, 2 or None), and `certificate_scope` (`"original"`,
  `"surrogate"` or None);
- `surrogate_model`: the final CPU float64 predictor, for inspection or saving its
  state dict; `surrogate` is its native optimization result;
- `to_json()`: a serializable audit record, excluding predictor weights.

Bounds from L1 and L2, or from different refinement models, are never merged.
An L2 acceptance leaves an unsuccessful L1 result unchanged and cannot mark the
original problem optimal. Each round is accepted only if the optimization and
both test-set checks pass for that same fixed model.
