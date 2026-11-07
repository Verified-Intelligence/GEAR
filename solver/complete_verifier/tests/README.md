# Complete Verifier Unit Tests

This directory contains the unit test suite for the `complete_verifier` package (α,β-CROWN verifier).

## Prerequisites

Activate the `alpha-beta-crown` conda environment:

```bash
conda activate alpha-beta-crown
```

To update your existing conda environment with additional test dependencies:

```bash
conda env update --file environment.yaml
```

## Running Tests

### Run All Tests

From the `complete_verifier/tests` directory:

```bash
pytest
```

Or from the repository root:

```bash
pytest complete_verifier/tests/
```

### Run a Specific Test File

```bash
pytest complete_verifier/tests/test_utils.py
```

### Run a Specific Test Class or Method

```bash
# Run a specific test class
pytest complete_verifier/tests/test_utils.py::TestTimerClass

# Run a specific test method
pytest complete_verifier/tests/test_utils.py::TestTimerClass::test_timer_can_be_used
```

### Run Tests with Verbose Output

```bash
pytest -v complete_verifier/tests/
```

### Run Tests with Duration Information

```bash
pytest --durations=0 complete_verifier/tests/
```

## Coverage Reports

### Generate Coverage Report

Run tests with coverage collection:

```bash
pytest --cov=complete_verifier complete_verifier/tests/
```

### Generate HTML Coverage Report

```bash
pytest --cov=complete_verifier --cov-report=html complete_verifier/tests/
```

This creates an `htmlcov/` directory. Open `htmlcov/index.html` in a browser to view the interactive coverage report.

### Generate XML Coverage Report (for CI)

```bash
pytest --cov=complete_verifier --cov-report=xml complete_verifier/tests/
```

### Generate Terminal Coverage Report with Missing Lines

```bash
pytest --cov=complete_verifier --cov-report=term-missing complete_verifier/tests/
```

### Combined Coverage Options

```bash
pytest --cov=complete_verifier \
       --cov-report=term-missing \
       --cov-report=html \
       complete_verifier/tests/
```

## Test Directory Structure

```
complete_verifier/tests/
├── conftest.py              # Pytest configuration and fixtures
├── pytest.ini               # Pytest settings and warning filters
├── fixtures/                # Test fixtures (models and specifications)
│   ├── create_fixtures.py   # Script to regenerate fixtures
│   ├── simple_mlp.onnx      # Simple MLP model for testing
│   ├── simple_cnn.onnx      # Simple CNN model for testing
│   ├── robustness_mlp.vnnlib
│   ├── robustness_cnn.vnnlib
│   ├── targeted_mlp.vnnlib
│   └── disjunctive_mlp.vnnlib
└── test_*.py                # Test modules
```

## Test Modules

| Module | Description |
|--------|-------------|
| `test_abcrown.py` | ABCROWN class initialization tests |
| `test_abcrown_api.py` | API interface tests |
| `test_alpha_beta.py` | Alpha-beta optimization tests |
| `test_arguments.py` | Argument configuration tests |
| `test_bab.py` | Branch and bound tests |
| `test_beta_CROWN_solver.py` | Beta-CROWN solver tests |
| `test_branching_domains.py` | Branching domain logic tests |
| `test_check_counterexample.py` | Counterexample validation tests |
| `test_complete_verifier_func.py` | Complete verification function tests |
| `test_data_utils.py` | Data utility tests |
| `test_domain_clipper.py` | Domain clipping tests |
| `test_domain_updater.py` | Domain update tests |
| `test_incomplete_verifier_func.py` | Incomplete verification tests |
| `test_integration.py` | End-to-end integration tests |
| `test_jit_precompile.py` | JIT precompilation tests |
| `test_load_model.py` | Model loading tests |
| `test_loading.py` | General loading tests |
| `test_model_defs.py` | Model definition tests |
| `test_onnx_opt.py` | ONNX optimization tests |
| `test_prune.py` | Pruning strategy tests |
| `test_read_vnnlib.py` | VNNLIB parser tests |
| `test_scip_model.py` | SCIP MIP solver integration tests |
| `test_specifications.py` | Specification handling tests |
| `test_tensor_storage.py` | Tensor storage tests |
| `test_utils.py` | Utility function tests |

## Test Fixtures

The `fixtures/` directory contains pre-generated ONNX models and VNNLIB specifications used for testing.

### Regenerating Fixtures

If you need to regenerate the test fixtures:

```bash
cd complete_verifier/tests/fixtures
python create_fixtures.py
```

### Available Fixtures

- **`simple_mlp.onnx`**: A 2-layer MLP (4 inputs → 8 hidden → 2 outputs)
- **`simple_cnn.onnx`**: A simple CNN with convolution, pooling, and fully connected layers
- **`robustness_mlp.vnnlib`**: L-infinity robustness specification for the MLP
- **`robustness_cnn.vnnlib`**: L-infinity robustness specification for the CNN
- **`targeted_mlp.vnnlib`**: Targeted attack specification
- **`disjunctive_mlp.vnnlib`**: Disjunctive output constraint specification

## Configuration

### pytest.ini

The `pytest.ini` file configures warning filters to suppress known non-critical warnings from third-party libraries:

- NumPy array writability warnings from `onnx2pytorch`
- Experimental implementation warnings
- Model conversion correctness check warnings

### conftest.py

The `conftest.py` file provides:

- Pytest configuration hooks
- Additional warning filters
- Shared fixtures (if any)
