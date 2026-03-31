#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running local CI pipeline"

if ! command -v julia >/dev/null 2>&1; then
  echo "Julia is not installed or not in PATH"
  exit 1
fi

echo "Julia version:"
julia --version

echo "Installing dependencies"
julia -e 'using Pkg; Pkg.add("Plots"); Pkg.add("XLSX")'

mkdir -p artifacts
mkdir -p artifacts/ci_upload

run_step() {
  local name="$1"
  shift
  echo "Running: $name"
  "$@"
  echo "Finished: $name"
}

run_step "Test core types" julia test/tests_for_core_types.jl
run_step "Test main" julia test/runtests.jl
run_step "Test data utils" julia test/test_data_utils.jl

run_step "Example XOR" julia examples/xor_example.jl
run_step "Example Spiral" julia examples/spiral_example.jl
run_step "Example MNIST" julia examples/mnist_example.jl
run_step "Example Fire Extinguisher" julia examples/fire_extinguisher_example.jl

echo "Collecting png files"
find . -type f -name "*.png" -exec cp {} artifacts/ci_upload/ \; 2>/dev/null || true

echo "Done"
echo "Collected files:"
ls -la artifacts/ci_upload || true